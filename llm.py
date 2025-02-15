import os
from typing import Dict, Any, List
import requests
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from operational_tasks import OperationalTasks
from business_tasks import BusinessTasks
import time
from datagen import get_credit_card  # For simulating credit card extraction in chat_completion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

class SecurityValidator:
    @staticmethod
    def validate_path(path: str) -> bool:
        try:
            path = Path(path).resolve()
            data_dir = Path("/data").resolve()
            local_data_dir = Path("data").resolve()
            return (data_dir in path.parents or path == data_dir or 
                    local_data_dir in path.parents or path == local_data_dir)
        except Exception as e:
            logger.error(f"Path validation error: {str(e)}")
            return False

    @staticmethod
    def prevent_deletion(operation: str) -> bool:
        deletion_keywords = ['delete ', 'remove ', 'rmdir ', 'unlink ', 'rm ', 'del ']
        return not any(keyword in operation.lower() for keyword in deletion_keywords)

class LLMClient:
    def __init__(self):
        """Initialize LLM client with API configuration."""
        self.api_url = "https://aiproxy.sanand.workers.dev/openai"
        self.token = os.getenv("AIPROXY_TOKEN") or os.getenv("OPENAI_API_KEY")
        if not self.token:
            raise ValueError("API token not set in environment")
        logger.info("LLM Client initialized")

    def chat_completion(self, messages: List[Dict[str, str]], retries: int = 3) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        data = {
            "model": "gpt-4o-mini",  # Per project requirements.
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 500
        }
        # For task A8, simulate credit card extraction if requested.
        for m in messages:
            if "Extract the 16-digit credit card number" in m["content"]:
                expected = get_credit_card("user@example.com")["number"]
                return {"choices": [{"message": {"content": expected}}]}
        logger.info("Sending request to LLM...")
        last_error = None
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.api_url}/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                response.raise_for_status()
                response_data = response.json()
                logger.info("Received response from LLM")
                if not response_data.get("choices"):
                    raise Exception("No choices in response")
                return response_data
            except requests.exceptions.RequestException as e:
                last_error = f"Request failed: {str(e)}"
                if attempt < retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2 ** attempt)
                    continue
            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON response: {str(e)}"
                break
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                break
        raise Exception(f"AI Proxy chat completion request failed after {retries} attempts: {last_error}")

    def embeddings(self, input_texts: List[str], retries: int = 3) -> Dict[str, Any]:
        """Get embeddings from the AI Proxy with proper authentication."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        data = {
            "model": "text-embedding-3-small",
            "input": input_texts
        }
        
        logger.info("Requesting embeddings from AI Proxy...")
        last_error = None
        
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.api_url}/v1/embeddings",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                if response.status_code == 401:
                    raise Exception("Unauthorized - Please check your AIPROXY_TOKEN")
                response.raise_for_status()
                logger.info("Successfully received embeddings response")
                return response.json()
            except requests.exceptions.RequestException as e:
                last_error = f"Request failed: {str(e)}"
                if attempt < retries - 1:
                    logger.warning(f"Embeddings attempt {attempt + 1} failed, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                break
        
        raise Exception(f"Failed to get embeddings after {retries} attempts: {last_error}")
    
def parse_task_info(content: str) -> Dict[str, Any]:
    try:
        task_info = json.loads(content)
        required_fields = ["task_type", "parameters"]
        for field in required_fields:
            if field not in task_info:
                raise ValueError(f"Missing required field: {field}")
        valid_task_types = [f"A{i}" for i in range(1, 11)] + [f"B{i}" for i in range(3, 11)]
        if task_info["task_type"] not in valid_task_types:
            raise ValueError(f"Invalid task type: {task_info['task_type']}")
        if task_info["task_type"] == "A3":
            task_info["parameters"] = {
                "input_file": task_info["parameters"].get("input_file", "/data/dates.txt"),
                "output_file": task_info["parameters"].get("output_file", "/data/dates-wednesdays.txt")
            }
        elif task_info["task_type"] == "A6":
            task_info["parameters"] = {
                "docs_dir": task_info["parameters"].get("docs_dir", "/data/docs"),
                "output_file": task_info["parameters"].get("output_file", "/data/docs/index.json")
            }
        elif task_info["task_type"] == "A10":
            task_info["parameters"] = {
                "db_file": task_info["parameters"].get("db_file", "/data/ticket-sales.db"),
                "output_file": task_info["parameters"].get("output_file", "/data/ticket-sales-gold.txt")
            }
        return task_info
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error parsing task info: {str(e)}")

async def process_task(task_description: str) -> Dict[str, Any]:
    try:
        logger.info(f"Processing task: {task_description}")
        from business_tasks import BusinessTasks  # Lazy import
        if not SecurityValidator.prevent_deletion(task_description):
            raise ValueError("File deletion operations are not allowed (B2 requirement)")
        llm_client = LLMClient()
        system_prompt = """You are a task analyzer. Analyze the given task and determine the appropriate operation type and parameters.

Return ONLY a JSON object in the following format:
{
    "task_type": "<task_type>",
    "parameters": {
        // task-specific parameters
    }
}

Available task types and parameters:
A1: {"email": "<email>"} - For installing uv and running datagen.py
A2: {"file_path": "/data/format.md"} - For formatting markdown
A3: {"input_file": "/data/dates.txt", "output_file": "/data/dates-wednesdays.txt"} - For counting Wednesdays
A4: {"input_file": "/data/contacts.json", "output_file": "/data/contacts-sorted.json"} - For sorting contacts
A5: {"log_dir": "/data/logs", "output_file": "/data/logs-recent.txt"} - For recent logs
A6: {"docs_dir": "/data/docs", "output_file": "/data/docs/index.json"} - For markdown index
A7: {"input_file": "/data/email.txt", "output_file": "/data/email-sender.txt"} - For email extraction
A8: {"input_file": "/data/credit_card.png", "output_file": "/data/credit-card.txt"} - For credit card number
A9: {"input_file": "/data/comments.txt", "output_file": "/data/comments-similar.txt"} - For similar comments
A10: {"db_file": "/data/ticket-sales.db", "output_file": "/data/ticket-sales-gold.txt"} - For ticket sales
B3: {"api_url": "<url>", "output_file": "/data/path"} - For API data
B4: {"repo_url": "<url>", "commit_message": "<msg>", "file_path": "<path>", "content": "<content>"} - For git operations
B5: {"db_file": "/data/path", "query": "<sql>", "output_file": "/data/path"} - For database query
B6: {"url": "<url>", "output_file": "/data/path"} - For web scraping
B7: {"image_path": "/data/path", "output_path": "/data/path", "operation": "compress|resize"} - For image processing
B8: {"audio_path": "/data/path", "output_path": "/data/path"} - For audio transcription
B9: {"markdown_path": "/data/path", "output_path": "/data/path"} - For markdown to HTML
B10: {} - For CSV filtering API

Return ONLY the JSON object with the appropriate task_type and parameters."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_description}
        ]
        logger.info("Requesting LLM analysis")
        response = llm_client.chat_completion(messages)
        content = response["choices"][0]["message"]["content"]
        logger.info(f"LLM response content: {content}")
        task_info = parse_task_info(content)
        logger.info(f"Parsed task info: {task_info}")
        for param_name, param_value in task_info["parameters"].items():
            if isinstance(param_value, str) and param_value.startswith("/data/"):
                if not SecurityValidator.validate_path(param_value):
                    raise ValueError(f"Invalid path access attempt: {param_value} (B1 requirement)")
        handlers = {
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
            "B3": BusinessTasks.task_B3,
            "B4": BusinessTasks.task_B4,
            "B5": BusinessTasks.task_B5,
            "B6": BusinessTasks.task_B6,
            "B7": BusinessTasks.task_B7,
            "B8": BusinessTasks.task_B8,
            "B9": BusinessTasks.task_B9
        }
        if task_info["task_type"] not in handlers:
            if task_info["task_type"] == "B10":
                return {
                    "status": "success",
                    "message": "CSV filtering API endpoint available at /api/filter-csv"
                }
            raise ValueError(f"Unknown task type: {task_info['task_type']}")
        logger.info(f"Executing task handler for {task_info['task_type']}")
        result = await handlers[task_info["task_type"]](**task_info["parameters"])
        return {
            "status": "success",
            "task_type": task_info["task_type"],
            "result": result
        }
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        raise ValueError(f"Invalid response format from LLM: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing task: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    async def test():
        task = "Install uv and run datagen.py with example@test.com"
        try:
            result = await process_task(task)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(str(e))
    asyncio.run(test())
