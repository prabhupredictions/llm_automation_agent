from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
import os
import json
import pandas as pd
from pathlib import Path
from llm import process_task
from business_tasks import BusinessTasks

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

def validate_path(path: str) -> bool:
    """Validate path is within /data directory"""
    try:
        path = Path(path).resolve()
        data_dir = Path("/data").resolve()
        return data_dir in path.parents
    except:
        return False

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.post("/run")
async def run_task(task: str):
    """
    Execute task based on description
    
    Args:
        task: Task description in plain English
        
    Returns:
        Task execution result
    """
    try:
        # Process task using LLM
        result = await process_task(task)
        return {
            "status": "success",
            "result": result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str):
    """
    Read file content
    
    Args:
        path: File path (must be within /data directory)
        
    Returns:
        File content as plain text
    """
    if not path.startswith("/data/"):
        raise HTTPException(
            status_code=400,
            detail="Can only access files in /data directory"
        )
    
    try:
        # Convert /data path to actual path
        actual_path = path.replace("/data/", "data/", 1)
        
        # Additional security check
        if not validate_path(actual_path):
            raise HTTPException(
                status_code=400,
                detail="Invalid path"
            )
        
        # Read file content
        with open(actual_path, 'r') as file:
            content = file.read()
            return content
            
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/filter-csv")
async def filter_csv(
    csv_path: str = Query(..., description="Path to CSV file"),
    column: str = Query(..., description="Column to filter on"),
    value: str = Query(..., description="Value to filter for")
):
    """
    Filter CSV data (B10 requirement)
    
    Args:
        csv_path: Path to CSV file (must be in /data directory)
        column: Column name to filter on
        value: Value to filter for
        
    Returns:
        Filtered data as JSON
    """
    try:
        if not validate_path(csv_path):
            raise HTTPException(
                status_code=400,
                detail="CSV file must be in /data directory"
            )
        
        # Read and filter CSV
        df = pd.read_csv(csv_path)
        if column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{column}' not found"
            )
        
        filtered_df = df[df[column] == value]
        return {
            "status": "success",
            "data": filtered_df.to_dict(orient='records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)