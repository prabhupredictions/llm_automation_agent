import os
from typing import Dict, Any, List
import requests
import json
from dotenv import load_dotenv
from operational_tasks import OperationalTasks
from business_tasks import BusinessTasks

load_dotenv()

class LLMClient:
    def __init__(self):
        """Initialize LLM client with API configuration"""
        self.api_url = "https://aiproxy.sanand.workers.dev/openai"
        self.token = os.getenv("AIPROXY_TOKEN")
        if not self.token:
            raise ValueError("AIPROXY_TOKEN not found")

    def chat_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Send chat completion request to AI Proxy"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        
        data = {
            "model": "gpt-4o-mini",
            "messages": messages
        }
        
        response = requests.post(
            f"{self.api_url}/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"Error from AI Proxy: {response.text}")
        
        return response.json()

async def process_task(task_description: str) -> Dict[str, Any]:
    """Process and route tasks to appropriate handlers"""
    try:
        llm_client = LLMClient()
        
        # Ask LLM to analyze task
        system_prompt = """Analyze the task and determine which operation (A1-A10 or B3-B10) is being requested.
        Return a JSON with:
        {
            "task_type": "A1" to "A10" or "B3" to "B10",
            "parameters": {
                // task-specific parameters from the description
            }
        }
        Task types:
        A1: Install uv and run datagen.py with email
        A2: Format markdown with prettier
        A3: Count weekdays in dates file
        A4: Sort contacts by name
        A5: Extract recent log lines
        A6: Create markdown index
        A7: Extract email sender
        A8: Extract credit card number
        A9: Find similar comments
        A10: Calculate ticket sales
        B3: Fetch API data
        B4: Git operations
        B5: Database query
        B6: Web scraping
        B7: Image processing
        B8: Audio transcription
        B9: Markdown to HTML
        B10: CSV filtering API"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_description}
        ]
        
        response = llm_client.chat_completion(messages)
        task_info = json.loads(response["choices"][0]["message"]["content"])
        
        # Map task types to handlers
        handlers = {
            # Operational Tasks
            "A1": OperationalTasks.task_A1,
            "A2": OperationalTasks.task_A2,
            "A3": OperationalTasks.task_A3,
            "A4": OperationalTasks.task_A4,
            "A5": OperationalTasks.task_A5,
            "A6": OperationalTasks.task_A6,
            "A7": lambda *args, **kwargs: OperationalTasks.task_A7(*args, **kwargs, llm_client=llm_client),
            "A8": lambda *args, **kwargs: OperationalTasks.task_A8(*args, **kwargs, llm_client=llm_client),
            "A9": lambda *args, **kwargs: OperationalTasks.task_A9(*args, **kwargs, llm_client=llm_client),
            "A10": OperationalTasks.task_A10,
            
            # Business Tasks
            "B3": BusinessTasks.task_B3,
            "B4": BusinessTasks.task_B4,
            "B5": BusinessTasks.task_B5,
            "B6": BusinessTasks.task_B6,
            "B7": BusinessTasks.task_B7,
            "B8": BusinessTasks.task_B8,
            "B9": BusinessTasks.task_B9
            # B10 is handled separately as it creates an API endpoint
        }
        
        # Handle tasks
        if task_info["task_type"] not in handlers:
            if task_info["task_type"] == "B10":
                return {
                    "status": "success",
                    "message": "CSV filtering API endpoint available at /api/filter-csv"
                }
            raise ValueError(f"Unknown task type: {task_info['task_type']}")
        
        # Execute task
        result = await handlers[task_info["task_type"]](**task_info["parameters"])
        return {
            "status": "success",
            "task_type": task_info["task_type"],
            "result": result
        }
    
    except json.JSONDecodeError:
        raise ValueError("Invalid response format from LLM")
    except Exception as e:
        raise Exception(f"Error processing task: {str(e)}")

if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def test():
        task = "Format the contents of /data/format.md using prettier 3.4.2"
        result = await process_task(task)
        print(json.dumps(result, indent=2))
    
    asyncio.run(test())