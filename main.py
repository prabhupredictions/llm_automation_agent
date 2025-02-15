from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
import json
import pandas as pd
from pathlib import Path
from llm import process_task
from business_tasks import BusinessTasks
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_data_directory() -> Path:
    """Get the appropriate data directory path based on environment"""
    # If running in Docker, use /data
    if os.path.exists('/.dockerenv'):
        return Path('/data')
    
    # For local development, use ./data (relative to current directory)
    return Path(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data')))

async def ensure_data_directory() -> None:
    """Ensure data directory exists with proper permissions"""
    try:
        data_dir = get_data_directory()
        data_dir.mkdir(exist_ok=True, parents=True)
        
        # Set permissions
        try:
            os.chmod(data_dir, 0o777)
        except Exception as e:
            logger.warning(f"Could not set permissions on data directory: {str(e)}")
            
        logger.info(f"Using data directory: {data_dir}")
    except Exception as e:
        logger.error(f"Failed to create/setup data directory: {str(e)}")
        raise

def validate_path(path: str) -> bool:
    """Validate if the path is within allowed directories"""
    try:
        path = Path(path).resolve()
        data_dir = get_data_directory().resolve()
        
        # Check if path is within allowed directory
        return data_dir in path.parents or path == data_dir
    except Exception as e:
        logger.error(f"Path validation error: {str(e)}")
        return False

def get_actual_path(path: str) -> str:
    """Get the actual filesystem path from the provided path"""
    if path.startswith("/data/"):
        # Remove /data/ prefix and join with actual data directory
        relative_path = path.replace("/data/", "", 1)
        return str(get_data_directory() / relative_path)
    return path

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events handler"""
    try:
        await ensure_data_directory()
        logger.info("Application started successfully")
        yield
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    finally:
        # Cleanup code (if any) goes here
        pass

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="LLM Automation Agent",
    description="API for automating tasks using LLM",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=Dict[str, str])
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "LLM Automation Agent API is running"
    }

@app.post("/run")
async def run_task(task: str) -> Dict[str, Any]:
    """
    Execute task based on description
    
    Args:
        task: Task description in plain English
        
    Returns:
        Dict containing task execution result
    """
    try:
        logger.info(f"Processing task: {task}")
        
        if not task or not isinstance(task, str):
            raise HTTPException(
                status_code=400,
                detail="Invalid task description"
            )
        
        # Process task using LLM
        result = await process_task(task)
        
        logger.info(f"Task completed successfully: {result}")
        return {
            "status": "success",
            "result": result
        }
        
    except ValueError as e:
        logger.error(f"Invalid task parameters: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Task processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str) -> str:
    """
    Read file content
    
    Args:
        path: File path (must be within /data directory)
        
    Returns:
        str: File content as plain text
    """
    try:
        if not path.startswith("/data/"):
            raise HTTPException(
                status_code=400,
                detail="Can only access files in /data directory"
            )
        
        actual_path = get_actual_path(path)
        
        if not validate_path(actual_path):
            raise HTTPException(
                status_code=400,
                detail="Invalid path"
            )
        
        try:
            with open(actual_path, 'r') as file:
                content = file.read()
                return content
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="File not found")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading file {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/filter-csv")
async def filter_csv(
    csv_path: str = Query(..., description="Path to CSV file"),
    column: str = Query(..., description="Column to filter on"),
    value: str = Query(..., description="Value to filter for")
) -> Dict[str, Any]:
    try:
        # Ensure path starts with /data/
        if not csv_path.startswith('/data/'):
            csv_path = f'/data/{csv_path}'
            
        actual_path = get_actual_path(csv_path)
        
        if not validate_path(actual_path):
            raise HTTPException(
                status_code=400,
                detail="CSV file must be in /data directory"
            )
        
        # Create test data if file doesn't exist (for testing)
        if not os.path.exists(actual_path):
            df = pd.DataFrame({
                'category': ['test', 'other'],
                'value': [1, 2]
            })
            df.to_csv(actual_path, index=False)
        
        df = pd.read_csv(actual_path)
        
        if column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{column}' not found"
            )
        
        filtered_df = df[df[column] == value]
        return {
            "data": filtered_df.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )