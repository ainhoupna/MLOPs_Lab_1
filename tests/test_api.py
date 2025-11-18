"""
Integration testing with the API

"""

# tests/test_api.py

import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient

#  Import the application from your file structure
from api.api import app 

# --- Fixtures for API Tests ---

@pytest.fixture(scope="module")
def test_client():
    """Fixture for the FastAPI TestClient."""
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="module")
def sample_image_path(tmp_path_factory):
    """Creates a temporary JPEG image file for use in tests."""
    img_dir = tmp_path_factory.mktemp("data_api")
    img_path = img_dir / "test_image.jpg"
    
    img = Image.new('RGB', (10, 10), color = 'red')
    img.save(img_path, "jpeg")
    
    return img_path

@pytest.fixture
def image_buffer(sample_image_path):
    """Reads the test image into an io.BytesIO buffer."""
    with open(sample_image_path, "rb") as f:
        img_bytes = io.BytesIO(f.read())
    
    img_bytes.seek(0)
    # Use yield so the buffer can be reset for tests if needed, though for client.post() it's consumed.
    yield img_bytes

# --- Tests ---

def test_api_predict_success(test_client, image_buffer):
    """Tests the /predict endpoint with a valid image upload."""
    files = {"file": ("test_image.jpg", image_buffer, "image/jpeg")}
    
    response = test_client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "predicted_class" in data

def test_api_predict_invalid_file_type(test_client):
    """Tests the /predict endpoint with an invalid content type (text/plain)."""
    text_buffer = io.BytesIO(b"This is not an image.")
    files = {"file": ("not_an_image.txt", text_buffer, "text/plain")}
    
    response = test_client.post("/predict", files=files)
    
    assert response.status_code == 400
    assert "Invalid input. Please upload a JPEG or PNG image." in response.json()["detail"]


def test_api_resize_success(test_client, image_buffer):
    """Tests the /resize endpoint with a valid image and form data."""
    files = {"file": ("test_image.jpg", image_buffer, "image/jpeg")}
    data = {"width": "50", "height": "50"}
    
    response = test_client.post("/resize", files=files, data=data)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    
    img = Image.open(io.BytesIO(response.content))
    assert img.size == (50, 50)


def test_api_resize_missing_form_data(test_client, image_buffer):
    """Tests the /resize endpoint when a required form parameter (height) is missing."""
    files = {"file": ("test_image.jpg", image_buffer, "image/jpeg")}
    data = {"width": "50"}
    
    response = test_client.post("/resize", files=files, data=data)
    
    assert response.status_code == 422
    assert "height" in response.json()["detail"][0]["loc"]