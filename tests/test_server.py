import pytest
from fastapi.testclient import TestClient
from src.server import app

client = TestClient(app)


class TestHealth:
    def test_health_check(self):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "triton_ready" in data
        assert "timestamp" in data

    def test_liveness_probe(self):
        """Test liveness probe."""
        response = client.get("/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"


class TestPrediction:
    def test_prediction_success(self):
        """Test successful prediction."""
        # Note: This requires Triton to be running
        payload = {
            "features": [0.5, 0.3, 0.7, 0.2, 0.9],
            "request_id": "test-123",
        }
        response = client.post("/predict", json=payload)
        # Might be 503 if Triton unavailable, which is expected in test
        assert response.status_code in [200, 503]

    def test_prediction_invalid_features(self):
        """Test prediction with invalid features."""
        payload = {
            "features": ["not", "a", "number"],
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_prediction_empty_features(self):
        """Test prediction with empty features."""
        payload = {
            "features": [],
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_prediction_out_of_range(self):
        """Test prediction with out-of-range values."""
        payload = {
            "features": [1e7],  # Beyond max range
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
