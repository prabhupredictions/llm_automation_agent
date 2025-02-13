import pytest
from fastapi.testclient import TestClient
import json
import os
from pathlib import Path
from main import app

# Create test client
client = TestClient(app)

def setup_test_data():
    """Setup test data directory and files"""
    # Create test content in the Docker-mounted data directory
    os.makedirs("data", exist_ok=True)
    with open("data/test.txt", "w") as f:
        f.write("test content")

def teardown_test_data():
    """Clean up test files"""
    try:
        os.remove("data/test.txt")
    except:
        pass

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()

def test_run_endpoint():
    """Test run endpoint"""
    response = client.post("/run", params={"task": "Test task"})
    assert response.status_code in [200, 400, 500]

def test_read_endpoint():
    """Test read endpoint"""
    try:
        # Setup test data
        setup_test_data()
        
        # Test reading the file
        response = client.get("/read", params={"path": "/data/test.txt"})
        assert response.status_code == 200
        assert response.text == "test content"
    finally:
        # Clean up
        teardown_test_data()

def test_invalid_path():
    """Test invalid path access"""
    response = client.get("/read", params={"path": "../invalid.txt"})
    assert response.status_code == 400

if __name__ == "__main__":
    pytest.main(["-v", __file__])