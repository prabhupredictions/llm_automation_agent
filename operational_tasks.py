import json
import os
from datetime import datetime
import aiofiles
import sqlite3
import requests
from typing import Dict, Any, List
import base64
from PIL import Image  # This should now work with Pillow installed
import subprocess

class OperationalTasks:
    @staticmethod
    async def task_A1(email: str) -> Dict[str, Any]:
        """Install uv and run datagen.py"""
        try:
            # Check if uv is installed
            try:
                subprocess.run(['uv', '--version'], check=True)
            except:
                subprocess.run(['pip', 'install', 'uv'], check=True)
            
            # Download and run datagen.py
            url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
            response = requests.get(url)
            
            with open("datagen.py", "w") as f:
                f.write(response.text)
            
            result = subprocess.run(['python', 'datagen.py', email], capture_output=True, text=True, check=True)
            return {"status": "success", "output": result.stdout}
        except Exception as e:
            raise Exception(f"Error in task A1: {str(e)}")

    @staticmethod
    async def task_A2(file_path: str) -> Dict[str, Any]:
        """Format markdown using prettier"""
        try:
            # Install prettier if not present
            subprocess.run(['npm', 'install', 'prettier@3.4.2'], check=True)
            
            # Format file in-place
            result = subprocess.run(['npx', 'prettier', '--write', file_path], capture_output=True, text=True, check=True)
            return {"status": "success", "output": result.stdout}
        except Exception as e:
            raise Exception(f"Error in task A2: {str(e)}")

    @staticmethod
    async def task_A3(input_file: str, output_file: str, weekday: str) -> Dict[str, Any]:
        """Count weekday occurrences in dates file"""
        try:
            async with aiofiles.open(input_file, 'r') as f:
                dates = await f.readlines()
            
            count = 0
            for date_str in dates:
                date = datetime.strptime(date_str.strip(), '%Y-%m-%d')
                if date.strftime('%A').lower() == weekday.lower():
                    count += 1
            
            async with aiofiles.open(output_file, 'w') as f:
                await f.write(str(count))
            return {"status": "success", "count": count}
        except Exception as e:
            raise Exception(f"Error in task A3: {str(e)}")

    @staticmethod
    async def task_A4(input_file: str, output_file: str) -> Dict[str, Any]:
        """Sort contacts by last_name, first_name"""
        try:
            async with aiofiles.open(input_file, 'r') as f:
                contacts = json.loads(await f.read())
            
            sorted_contacts = sorted(
                contacts,
                key=lambda x: (x['last_name'], x['first_name'])
            )
            
            async with aiofiles.open(output_file, 'w') as f:
                await f.write(json.dumps(sorted_contacts, indent=2))
            return {"status": "success", "contacts_sorted": len(sorted_contacts)}
        except Exception as e:
            raise Exception(f"Error in task A4: {str(e)}")

    @staticmethod
    async def task_A5(log_dir: str, output_file: str) -> Dict[str, Any]:
        """Extract first lines from recent log files"""
        try:
            log_files = []
            for file in os.listdir(log_dir):
                if file.endswith('.log'):
                    path = os.path.join(log_dir, file)
                    log_files.append((path, os.path.getmtime(path)))
            
            recent_logs = sorted(log_files, key=lambda x: x[1], reverse=True)[:10]
            first_lines = []
            
            for log_path, _ in recent_logs:
                async with aiofiles.open(log_path, 'r') as f:
                    first_line = await f.readline()
                    first_lines.append(first_line.strip())
            
            async with aiofiles.open(output_file, 'w') as f:
                await f.write('\n'.join(first_lines))
            return {"status": "success", "logs_processed": len(first_lines)}
        except Exception as e:
            raise Exception(f"Error in task A5: {str(e)}")

    @staticmethod
    async def task_A6(docs_dir: str, output_file: str) -> Dict[str, Any]:
        """Create index of markdown H1 headings"""
        try:
            index = {}
            for root, _, files in os.walk(docs_dir):
                for file in files:
                    if file.endswith('.md'):
                        relative_path = os.path.relpath(os.path.join(root, file), docs_dir)
                        async with aiofiles.open(os.path.join(root, file), 'r') as f:
                            content = await f.read()
                            
                            # Find first H1 heading
                            for line in content.split('\n'):
                                if line.startswith('# '):
                                    index[relative_path] = line[2:].strip()
                                    break
            
            async with aiofiles.open(output_file, 'w') as f:
                await f.write(json.dumps(index, indent=2))
            return {"status": "success", "files_indexed": len(index)}
        except Exception as e:
            raise Exception(f"Error in task A6: {str(e)}")

    @staticmethod
    async def task_A7(input_file: str, output_file: str, llm_client) -> Dict[str, Any]:
        """Extract email sender using LLM"""
        try:
            async with aiofiles.open(input_file, 'r') as f:
                email_content = await f.read()
            
            # Ask LLM to extract email
            response = llm_client.chat_completion([
                {"role": "system", "content": "Extract the sender's email address from this email message. Return only the email address."},
                {"role": "user", "content": email_content}
            ])
            
            email_address = response["choices"][0]["message"]["content"].strip()
            
            async with aiofiles.open(output_file, 'w') as f:
                await f.write(email_address)
            return {"status": "success", "email": email_address}
        except Exception as e:
            raise Exception(f"Error in task A7: {str(e)}")

    @staticmethod
    async def task_A8(input_file: str, output_file: str, llm_client) -> Dict[str, Any]:
        """Extract credit card number from image using LLM"""
        try:
            # Read image and convert to base64
            with open(input_file, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()
            
            # Ask LLM to extract card number
            response = llm_client.chat_completion([
                {"role": "system", "content": "Extract the credit card number from this image. Return only the numbers without spaces."},
                {"role": "user", "content": f"Base64 image: {image_data}"}
            ])
            
            card_number = ''.join(filter(str.isdigit, response["choices"][0]["message"]["content"]))
            
            async with aiofiles.open(output_file, 'w') as f:
                await f.write(card_number)
            return {"status": "success", "card_number": card_number}
        except Exception as e:
            raise Exception(f"Error in task A8: {str(e)}")

    @staticmethod
    async def task_A9(input_file: str, output_file: str, llm_client) -> Dict[str, Any]:
        """Find similar comments using embeddings"""
        try:
            async with aiofiles.open(input_file, 'r') as f:
                comments = [line.strip() for line in await f.readlines()]
            
            # Get embeddings for all comments
            embeddings = []
            for comment in comments:
                response = requests.post(
                    "https://aiproxy.sanand.workers.dev/openai/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {os.getenv('AIPROXY_TOKEN')}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "text-embedding-3-small",
                        "input": comment
                    }
                )
                embeddings.append(response.json()["data"][0]["embedding"])
            
            # Find most similar pair
            max_similarity = -1
            similar_pair = None
            
            for i in range(len(comments)):
                for j in range(i + 1, len(comments)):
                    similarity = sum(a * b for a, b in zip(embeddings[i], embeddings[j]))
                    if similarity > max_similarity:
                        max_similarity = similarity
                        similar_pair = (comments[i], comments[j])
            
            async with aiofiles.open(output_file, 'w') as f:
                await f.write('\n'.join(similar_pair))
            return {"status": "success", "similarity": max_similarity}
        except Exception as e:
            raise Exception(f"Error in task A9: {str(e)}")

    @staticmethod
    async def task_A10(db_file: str, output_file: str) -> Dict[str, Any]:
        """Calculate total sales for Gold tickets"""
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT SUM(units * price)
                FROM tickets
                WHERE type = 'Gold'
            """)
            
            total_sales = cursor.fetchone()[0]
            conn.close()
            
            async with aiofiles.open(output_file, 'w') as f:
                await f.write(str(total_sales))
            return {"status": "success", "total_sales": total_sales}
        except Exception as e:
            raise Exception(f"Error in task A10: {str(e)}")