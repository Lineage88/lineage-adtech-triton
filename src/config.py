import os
from typing import Optional

# Triton Configuration
TRITON_HOST = os.getenv("TRITON_HOST", "localhost")
TRITON_PORT = int(os.getenv("TRITON_PORT", "8001"))
TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "ctr_model")
TRITON_MODEL_VERSION = os.getenv("TRITON_MODEL_VERSION", "1")
TRITON_TIMEOUT_MS = int(os.getenv("TRITON_TIMEOUT_MS", "60000"))

# Application Configuration
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
APP_ENV = os.getenv("APP_ENV", "production")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Feature vector configuration
EXPECTED_FEATURE_DIM = int(os.getenv("EXPECTED_FEATURE_DIM", "20"))
MAX_FEATURE_DIM = int(os.getenv("MAX_FEATURE_DIM", "100"))

# Security
API_KEY = os.getenv("API_KEY", None)
ENABLE_CORS = os.getenv("ENABLE_CORS", "false").lower() == "true"

# Metrics
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
METRICS_PORT = int(os.getenv("METRICS_PORT", "8001"))
