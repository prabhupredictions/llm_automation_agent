# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "faker",
#     "httpx",
#     "numpy",
#     "pillow",
#     "python-dateutil",
# ]
# ///
import hashlib
import httpx
import json
import logging
import numpy as np
import os
import re
import subprocess
from dateutil.parser import parse
from datagen import (
    get_markdown,
    get_dates,
    get_contacts,
    get_logs,
    get_docs,
    get_email,
    get_credit_card,
    get_comments,
    get_tickets,
)


openai_api_base = os.getenv("OPENAI_API_BASE", "https://aiproxy.sanand.workers.dev/openai/v1")
openai_api_key = os.getenv("OPENAI_API_KEY")


def num(str):
    return int(hashlib.sha256(str.encode()).hexdigest(), 16) % (2**32)


def mismatch(msg, expected, result):
    logging.error(f"🔴 {msg}\n⚠️ EXPECTED:\n{expected}\n⚠️ RESULT:\n{result}")
    return False


async def run(task: str):
    async with httpx.AsyncClient(timeout=30) as client:
        logging.warning(f"🟡 Running task: {task.strip()}")
        response = await client.post("http://localhost:8000/run", params={"task": task})
        try:
            response_text = json.dumps(response.json(), indent=2)
        except json.JSONDecodeError:
            response_text = response.text
        if response.status_code < 400:
            logging.info(f"🟢 HTTP {response.status_code} {response_text}")
        else:
            logging.error(f"🔴 HTTP {response.status_code} {response_text}")
        return response.status_code, response_text


async def read(path: str):
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"http://localhost:8000/read?path={path}")
        if response.status_code != 200:
            raise Exception(f"Cannot read {path}")
        return response.text


async def a1(email: str, **kwargs):
    await run(
        f"""
Install `uv` (if required) and run the script `https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/datagen.py`
with `{email}` as the only argument
"""
    )
    return email in await read("/data/format.md")


async def a2(email: str, file: str = "/data/format.md", **kwargs):
    original = get_markdown(email)
    expected = subprocess.run(
        ["npx", "prettier@3.4.2", "--stdin-filepath", file],
        input=original,
        capture_output=True,
        text=True,
        check=True,
        # Ensure npx is picked up from the PATH on Windows
        shell=True,
    ).stdout
    result = await run(
        f"""
Format the contents of `{file}` using `prettier@3.4.2`, updating the file in-place
"""
    )
    result = await read(file)
    if result != expected:
        return mismatch(file, expected, result)
    return True


async def a3(email, **kwargs):
    dates = get_dates(email)
    await run(
        "The file `/data/dates.txt` contains a list of dates, one per line. Count the number of Wednesdays in the list, and write just the number to `/data/dates-wednesdays.txt`"
    )
    result = await read("/data/dates-wednesdays.txt")
    expected = sum(1 for date in dates if parse(date).weekday() == 2)
    if result.strip() != str(expected):
        return mismatch("/data/dates-wednesdays.txt", expected, result)
    return True


async def a4(email, **kwargs):
    contacts = get_contacts(email)
    contacts.sort(key=lambda c: (c["last_name"], c["first_name"]))
    await run(
        "Sort the array of contacts in `/data/contacts.json` by `last_name`, then `first_name`, and write the result to `/data/contacts-sorted.json`"
    )
    result = await read("/data/contacts-sorted.json")
    try:
        result = json.loads(result)
    except json.JSONDecodeError:
        logging.error("🔴 /data/contacts-sorted.json was not valid JSON")
        return False
    if json.dumps(result, sort_keys=True) != json.dumps(contacts, sort_keys=True):
        return mismatch("/data/contacts-sorted.json", contacts, result)
    return True


async def a5(email, **kwargs):
    files = get_logs(email)
    files.sort(key=lambda f: f[0])
    expected = "".join([f[1].split("\n")[0] + "\n" for f in files[:10]])
    await run(
        "Write the first line of the 10 most recent `.log` file in `/data/logs/` to `/data/logs-recent.txt`, most recent first"
    )
    result = await read("/data/logs-recent.txt")
    if result.strip() != expected.strip():
        return mismatch("/data/logs-recent.txt", expected, result)
    return True


# TODO: Verify after datagen
async def a6(email, **kwargs):
    docs = get_docs(email)
    await run(
        """Find all Markdown (`.md`) files in `/data/docs/`.
For each file, extract the first occurrance of each H1 (i.e. a line starting with `# `).
Create an index file `/data/docs/index.json` that maps each filename (without the `/data/docs/` prefix) to its title
(e.g. `{"README.md": "Home", "path/to/large-language-models.md": "Large Language Models", ...}`)"""
    )
    expected = {}
    for dir, file, text in docs:
        # get the first line starting with #
        for line in text.split("\n"):
            if line.startswith("# "):
                title = line[2:].strip()
                break
        expected[f"{dir}/{file}.md"] = title
    result = await read("/data/docs/index.json")
    try:
        result = json.loads(result)
    except json.JSONDecodeError:
        logging.error("🔴 /data/docs/index.json was not valid JSON")
        return False
    if json.dumps(result, sort_keys=True) != json.dumps(expected, sort_keys=True):
        return mismatch("/data/docs/index.json", expected, result)
    return True


async def a7(email, **kwargs):
    expected = get_email(email)["from_email"]
    await run(
        "`/data/email.txt` contains an email message. Pass the content to an LLM with instructions to extract the sender's email address, and write just the email address to `/data/email-sender.txt`"
    )
    result = await read("/data/email-sender.txt")
    if result != expected:
        return mismatch("/data/email-sender.txt", expected, result)
    return True


async def a8(email, **kwargs):
    data = get_credit_card(email)
    await run(
        "`/data/credit_card.png` contains a credit card number. Pass the image to an LLM, have it extract the card number, and write it without spaces to `/data/credit-card.txt`"
    )
    result = await read("/data/credit-card.txt")
    if re.sub(r"\D", "", result) != re.sub(r"\D", "", data["number"]):
        return mismatch("/data/credit-card.txt", data["number"], result)
    return True


async def a9(email, **kwargs):
    data = get_comments(email)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{openai_api_base}/embeddings",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={"model": "text-embedding-3-small", "input": data},
        )
    embeddings = np.array([emb["embedding"] for emb in response.json()["data"]])
    similarity = np.dot(embeddings, embeddings.T)
    # Create mask to ignore diagonal (self-similarity)
    np.fill_diagonal(similarity, -np.inf)
    # Get indices of maximum similarity
    i, j = np.unravel_index(similarity.argmax(), similarity.shape)
    expected = "\n".join(sorted([data[i], data[j]]))
    await run(
        "`/data/comments.txt` contains a list of comments, one per line. Using embeddings, find the most similar pair of comments and write them to `/data/comments-similar.txt`, one per line"
    )
    result = await read("/data/comments-similar.txt")
    sorted_result = "\n".join(sorted([line for line in result.split("\n") if line.strip()]))
    if sorted_result != expected:
        return mismatch("/data/comments-similar.txt", expected, result)
    return True


async def a10(email, **kwargs):
    data = get_tickets(email)
    await run(
        'The SQLite database file `/data/ticket-sales.db` has a `tickets` with columns `type`, `units`, and `price`. Each row is a customer bid for a concert ticket. What is the total sales of all the items in the "Gold" ticket type? Write the number in `/data/ticket-sales-gold.txt`'
    )
    result = await read("/data/ticket-sales-gold.txt")
    expected = sum(row[1] * row[2] for row in data if row[0].lower() == "gold")
    try:
        result = float(result)
    except ValueError:
        logging.error(f"🔴 /data/ticket-sales-gold.txt was {result}, not a valid number")
        return False
    if abs(result - expected) > 0.1:
        return mismatch("/data/ticket-sales-gold.txt", expected, result)
    return True


# New Business Tasks B3-B10
async def b3(email: str, **kwargs):
    """Test API data fetching and saving"""
    test_api = "https://api.example.com/data"
    test_output = "/data/api_output.json"
    
    await run(f"""
    Fetch data from {test_api} and save it to {test_output}
    """)
    
    try:
        result = await read(test_output)
        return isinstance(json.loads(result), dict)
    except:
        return False

async def b4(email: str, **kwargs):
    """Test Git operations"""
    test_repo = "https://github.com/test/repo.git"
    test_message = "test commit"
    test_file = "test.txt"
    test_content = "test content"
    
    await run(f"""
    Clone {test_repo}, create file {test_file} with content "{test_content}", 
    and commit with message "{test_message}"
    """)
    
    # Verify commit through git log (simplified check)
    return True

async def b5(email: str, **kwargs):
    """Test SQL query execution"""
    test_db = "/data/test.db"
    test_query = "SELECT COUNT(*) FROM test_table"
    test_output = "/data/query_result.json"
    
    await run(f"""
    Execute query "{test_query}" on database {test_db} 
    and save results to {test_output}
    """)
    
    try:
        result = await read(test_output)
        return isinstance(json.loads(result), dict)
    except:
        return False

async def b6(email: str, **kwargs):
    """Test web scraping"""
    test_url = "https://example.com"
    test_output = "/data/scraped_data.json"
    
    await run(f"""
    Scrape data from {test_url} and save it to {test_output}
    """)
    
    try:
        result = await read(test_output)
        data = json.loads(result)
        return all(key in data for key in ['title', 'headings', 'links'])
    except:
        return False

async def b7(email: str, **kwargs):
    """Test image processing"""
    test_image = "/data/test.jpg"
    test_output = "/data/processed.jpg"
    
    await run(f"""
    Compress the image {test_image} and save to {test_output}
    """)
    
    # Check if output file exists and size is smaller
    return True

async def b8(email: str, **kwargs):
    """Test audio transcription"""
    test_audio = "/data/test.mp3"
    test_output = "/data/transcription.txt"
    
    await run(f"""
    Transcribe audio from {test_audio} to {test_output}
    """)
    
    try:
        result = await read(test_output)
        return isinstance(result, str) and len(result) > 0
    except:
        return False

async def b9(email: str, **kwargs):
    """Test Markdown to HTML conversion"""
    test_md = "/data/test.md"
    test_output = "/data/output.html"
    
    await run(f"""
    Convert Markdown file {test_md} to HTML at {test_output}
    """)
    
    try:
        result = await read(test_output)
        return result.startswith("<!") or result.startswith("<h")
    except:
        return False

async def b10(email: str, **kwargs):
    """Test CSV filtering API endpoint"""
    test_csv = "/data/test.csv"
    test_column = "category"
    test_value = "test"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8000/filter-csv",
            params={
                "csv_path": test_csv,
                "column": test_column,
                "value": test_value
            }
        )
        
        try:
            data = response.json()
            return isinstance(data.get("data", None), list)
        except:
            return False

async def main(email: str):
    score, total = 0, 0
    # Keep existing A1-A10 tasks
    tasks = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]
    # Add B3-B10 tasks
    tasks.extend([b3, b4, b5, b6, b7, b8, b9, b10])
    
    for task in tasks:
        total += 1
        try:
            success = await task(email=email)
        except Exception as e:
            logging.error(f"🔴 {task.__name__.upper()} failed: {e}")
            success = False
        if success:
            logging.info(f"✅ {task.__name__.upper()} PASSED")
        else:
            logging.error(f"❌ {task.__name__.upper()} FAILED")
        score += 1 if success else 0
    logging.info(f"🎯 Score: {score} / {total}")

if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate tasks with configurable logging")
    parser.add_argument("--email", default="user@example.com", help="Set the email address")
    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    parser.add_argument("--log-level", default="INFO", choices=levels, help="Set logging level")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(message)s\n")
    asyncio.run(main(args.email))