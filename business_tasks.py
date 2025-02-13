import os
import json
import sqlite3
import requests
from typing import Dict, Any
from pathlib import Path
import pandas as pd
from PIL import Image
import markdown
from bs4 import BeautifulSoup
from fastapi import APIRouter, HTTPException, Query
from git import Repo
import shutil
import speech_recognition as sr
from pydub import AudioSegment

class BusinessTasks:
    @staticmethod
    def _validate_path(path: str) -> bool:
        """Validate path is within /data directory (B1)"""
        try:
            path = Path(path).resolve()
            data_dir = Path("/data").resolve()
            return data_dir in path.parents
        except:
            return False

    @staticmethod
    async def task_B3(api_url: str, output_file: str) -> Dict[str, Any]:
        """Fetch API data and save it"""
        try:
            if not BusinessTasks._validate_path(output_file):
                raise ValueError("Output file must be in /data directory")

            response = requests.get(api_url)
            response.raise_for_status()
            
            with open(output_file, 'w') as f:
                json.dump(response.json(), f)
            
            return {"status": "success"}
        except Exception as e:
            raise Exception(f"Error in task B3: {str(e)}")

    @staticmethod
    async def task_B4(repo_url: str, commit_message: str, file_path: str, content: str) -> Dict[str, Any]:
        """Clone git repo and make a commit"""
        try:
            repo_dir = "/data/temp_repo"
            if not BusinessTasks._validate_path(repo_dir):
                raise ValueError("Repository directory must be in /data directory")

            # Clean up existing repo if it exists
            if os.path.exists(repo_dir):
                shutil.rmtree(repo_dir)

            # Clone repository
            repo = Repo.clone_from(repo_url, repo_dir)
            
            # Create and modify file
            full_path = os.path.join(repo_dir, file_path)
            with open(full_path, 'w') as f:
                f.write(content)
            
            # Commit changes
            repo.index.add([file_path])
            repo.index.commit(commit_message)
            
            return {"status": "success", "commit_hash": str(repo.head.commit)}
        except Exception as e:
            raise Exception(f"Error in task B4: {str(e)}")

    @staticmethod
    async def task_B5(db_file: str, query: str, output_file: str) -> Dict[str, Any]:
        """Run SQL query on database"""
        try:
            if not all(BusinessTasks._validate_path(p) for p in [db_file, output_file]):
                raise ValueError("Files must be in /data directory")

            conn = sqlite3.connect(db_file)
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            with open(output_file, 'w') as f:
                df.to_json(f)
            
            return {"status": "success", "rows": len(df)}
        except Exception as e:
            raise Exception(f"Error in task B5: {str(e)}")

    @staticmethod
    async def task_B6(url: str, output_file: str) -> Dict[str, Any]:
        """Scrape website data"""
        try:
            if not BusinessTasks._validate_path(output_file):
                raise ValueError("Output file must be in /data directory")

            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            data = {
                "title": soup.title.string if soup.title else None,
                "headings": [h.text for h in soup.find_all(['h1', 'h2', 'h3'])],
                "links": [a.get('href') for a in soup.find_all('a', href=True)]
            }
            
            with open(output_file, 'w') as f:
                json.dump(data, f)
            
            return {"status": "success"}
        except Exception as e:
            raise Exception(f"Error in task B6: {str(e)}")

    @staticmethod
    async def task_B7(image_path: str, output_path: str, operation: str = "compress") -> Dict[str, Any]:
        """Process image"""
        try:
            if not all(BusinessTasks._validate_path(p) for p in [image_path, output_path]):
                raise ValueError("Files must be in /data directory")

            with Image.open(image_path) as img:
                if operation == "compress":
                    img.save(output_path, optimize=True, quality=85)
                elif operation == "resize":
                    img.thumbnail((800, 800))
                    img.save(output_path)
            
            return {"status": "success"}
        except Exception as e:
            raise Exception(f"Error in task B7: {str(e)}")

    @staticmethod
    async def task_B8(audio_path: str, output_path: str) -> Dict[str, Any]:
        """Transcribe audio from MP3"""
        try:
            if not all(BusinessTasks._validate_path(p) for p in [audio_path, output_path]):
                raise ValueError("Files must be in /data directory")

            # Convert MP3 to WAV
            audio = AudioSegment.from_mp3(audio_path)
            wav_path = audio_path.replace('.mp3', '.wav')
            audio.export(wav_path, format="wav")
            
            # Initialize recognizer
            recognizer = sr.Recognizer()
            
            # Transcribe
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
            
            # Save transcription
            with open(output_path, 'w') as f:
                f.write(text)
            
            return {"status": "success"}
        except Exception as e:
            raise Exception(f"Error in task B8: {str(e)}")

    @staticmethod
    async def task_B9(markdown_path: str, output_path: str) -> Dict[str, Any]:
        """Convert Markdown to HTML"""
        try:
            if not all(BusinessTasks._validate_path(p) for p in [markdown_path, output_path]):
                raise ValueError("Files must be in /data directory")

            with open(markdown_path, 'r') as f:
                content = f.read()
            
            html = markdown.markdown(content)
            
            with open(output_path, 'w') as f:
                f.write(html)
            
            return {"status": "success"}
        except Exception as e:
            raise Exception(f"Error in task B9: {str(e)}")

    @staticmethod
    def task_B10() -> APIRouter:
        """Create CSV filter endpoint"""
        router = APIRouter()
        
        @router.get("/filter-csv")
        async def filter_csv(
            csv_path: str = Query(..., description="Path to CSV file"),
            column: str = Query(..., description="Column to filter on"),
            value: str = Query(..., description="Value to filter for")
        ):
            try:
                if not BusinessTasks._validate_path(csv_path):
                    raise HTTPException(status_code=400, detail="CSV must be in /data directory")
                
                df = pd.read_csv(csv_path)
                if column not in df.columns:
                    raise HTTPException(status_code=400, detail=f"Column '{column}' not found")
                
                filtered_df = df[df[column] == value]
                return {"data": filtered_df.to_dict(orient='records')}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        return router