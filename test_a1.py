# Create a test script (test_a1.py)
import subprocess
import requests
import os
import logging

logging.basicConfig(level=logging.INFO)

def test_a1():
    try:
        # 1. Check if uv is installed
        logging.info("Checking uv installation...")
        try:
            subprocess.run(['uv', '--version'], check=True, capture_output=True)
            logging.info("uv is installed")
        except FileNotFoundError:
            logging.info("Installing uv...")
            subprocess.run(['pip', 'install', 'uv'], check=True)

        # 2. Download datagen.py
        logging.info("Downloading datagen.py...")
        url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/datagen.py"
        response = requests.get(url)
        
        with open("datagen.py", "w") as f:
            f.write(response.text)
        logging.info("datagen.py downloaded successfully")

        # 3. List current directory and check datagen.py
        logging.info("Current directory contents:")
        subprocess.run(['ls', '-l'], check=True)

        # 4. Check datagen.py content
        with open("datagen.py", 'r') as f:
            logging.info("First few lines of datagen.py:")
            logging.info(f.readlines()[:5])

        # 5. Run datagen.py with debug output
        logging.info("Running datagen.py...")
        result = subprocess.run(
            ['python', 'datagen.py', 'your@email.com'],
            capture_output=True,
            text=True
        )
        logging.info(f"Return code: {result.returncode}")
        logging.info(f"Output: {result.stdout}")
        logging.info(f"Error: {result.stderr}")

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    test_a1()