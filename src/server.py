import os
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Request, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, start_http_server
import time
import secrets

from src.inference_client import TritonInferenceClient
from src.exceptions import triton_error_to_http, TritonConnectionError, TritonInferenceError
from src.logger import logger
from src import config

# Metrics
request_counter = Counter(
    "ctr_prediction_requests_total",
    "Total prediction requests",
    ["status"],
)
inference_time_histogram = Histogram(
    "ctr_inference_duration_ms",
    "Inference duration in milliseconds",
    buckets=(10, 50, 100, 250, 500, 1000),
)
prediction_value_histogram = Histogram(
    "ctr_prediction_value",
    "Predicted CTR values",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
)

# Global Triton client
triton_client: TritonInferenceClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown."""
    global triton_client
    
    # Startup
    logger.info("Starting CTR Inference Service", extra={"env": config.APP_ENV})
    
    # Start Prometheus metrics server if enabled
    if config.ENABLE_METRICS:
        try:
            start_http_server(config.METRICS_PORT)
            logger.info(f"Metrics server started on port {config.METRICS_PORT}")
        except Exception as exc:
            logger.warning(f"Failed to start metrics server: {exc}")
    
    try:
        triton_client = TritonInferenceClient(
            host=config.TRITON_HOST,
            port=config.TRITON_PORT,
            model_name=config.TRITON_MODEL_NAME,
            model_version=config.TRITON_MODEL_VERSION,
        )
        logger.info("Triton client initialized successfully")
    except TritonConnectionError as exc:
        logger.error(f"Failed to initialize Triton client: {exc}")
        # Don't fail startup in case Triton is warming up
        # The readiness probe will prevent traffic until it's ready
    
    yield
    
    # Shutdown
    logger.info("Shutting down CTR Inference Service")


app = FastAPI(
    title="CTR Inference Service",
    description="NVIDIA Triton-based real-time CTR prediction for ad bidding",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS if enabled
if config.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def verify_api_key(x_api_key: str = Header(None)) -> str:
    """Verify API key if authentication is enabled."""
    if not config.API_KEY:
        return "anonymous"
    
    if not x_api_key or not secrets.compare_digest(x_api_key, config.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key",
        )
    return "authenticated"


class PredictionRequest(BaseModel):
    """Request body for CTR prediction."""
    features: List[float] = Field(
        ...,
        description="Feature vector (typically 20 dimensions)",
        min_items=1,
        max_items=config.MAX_FEATURE_DIM,
    )
    request_id: str = Field(
        default=None,
        description="Optional request tracking ID",
    )
    
    @validator("features")
    def validate_features(cls, v):
        """Validate feature values."""
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Feature {i} must be numeric")
            if not (-1e6 <= val <= 1e6):
                raise ValueError(f"Feature {i} out of valid range [-1e6, 1e6]")
        return v


class PredictionResponse(BaseModel):
    """Response body for CTR prediction."""
    prediction: float = Field(
        description="Predicted CTR (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    confidence: float = Field(
        description="Model confidence (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    inference_time_ms: float = Field(
        description="Inference latency in milliseconds"
    )
    request_id: str = Field(
        description="Request tracking ID"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    triton_ready: bool
    timestamp: float
    version: str


class ErrorResponse(BaseModel):
    """Error response body."""
    error: str
    detail: str
    request_id: str
    timestamp: float


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint",
)
async def health() -> HealthResponse:
    """Check service and Triton readiness."""
    try:
        triton_ready = triton_client.is_ready() if triton_client else False
    except Exception as exc:
        logger.warning(f"Health check failed: {exc}")
        triton_ready = False
    
    return HealthResponse(
        status="healthy" if triton_ready else "degraded",
        triton_ready=triton_ready,
        timestamp=time.time(),
        version="1.0.0",
    )


@app.get(
    "/ready",
    tags=["Health"],
    summary="Kubernetes readiness probe",
)
async def readiness():
    """Kubernetes readiness probe."""
    if not triton_client or not triton_client.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )
    return {"status": "ready"}


@app.get(
    "/live",
    tags=["Health"],
    summary="Kubernetes liveness probe",
)
async def liveness():
    """Kubernetes liveness probe."""
    # Simple check - if the app is responding, it's alive
    return {"status": "alive"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="CTR prediction endpoint",
    status_code=status.HTTP_200_OK,
)
async def predict(
    request: PredictionRequest,
    auth: str = Depends(verify_api_key),
) -> PredictionResponse:
    """Run CTR prediction on Triton.
    
    Accepts a feature vector and returns a CTR prediction with
    confidence score and inference latency.
    
    **Security:** Include `X-API-Key` header if API key is configured.
    """
    request_id = request.request_id or f"req-{int(time.time() * 1000)}"
    
    if not triton_client:
        request_counter.labels(status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Triton client not initialized",
        )
    
    try:
        # Run inference
        result = triton_client.predict(
            features=request.features,
            metadata={
                "request_id": request_id,
                "auth": auth,
            },
        )
        
        prediction = result["prediction"]
        inference_time = result["inference_time_ms"]
        
        # Record metrics
        request_counter.labels(status="success").inc()
        inference_time_histogram.observe(inference_time)
        prediction_value_histogram.observe(prediction)
        
        # Compute confidence as sigmoid derivative approximation
        confidence = abs(prediction - 0.5) * 2  # Maps [0,1] to [0,1]
        
        logger.info(
            "Prediction served",
            extra={
                "request_id": request_id,
                "prediction": prediction,
                "inference_time_ms": inference_time,
                "auth": auth,
            },
        )
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            inference_time_ms=inference_time,
            request_id=request_id,
        )
    
    except (TritonConnectionError, TritonInferenceError) as exc:
        request_counter.labels(status="error").inc()
        logger.error(
            f"Prediction failed: {exc}",
            extra={"request_id": request_id, "auth": auth},
        )
        raise triton_error_to_http(exc)
    
    except ValueError as exc:
        request_counter.labels(status="validation_error").inc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input validation failed: {exc}",
        )


@app.get(
    "/model/info",
    tags=["Model"],
    summary="Get model metadata",
)
async def model_info(
    auth: str = Depends(verify_api_key),
):
    """Fetch model metadata from Triton."""
    if not triton_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Triton client not initialized",
        )
    
    try:
        metadata = triton_client.get_model_metadata()
        return {
            "model_name": config.TRITON_MODEL_NAME,
            "model_version": config.TRITON_MODEL_VERSION,
            "metadata": metadata,
        }
    except Exception as exc:
        logger.error(f"Failed to fetch model metadata: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch model metadata",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    request_id = f"err-{int(time.time() * 1000)}"
    request_counter.labels(status="error").inc()
    logger.error(
        f"Unhandled exception: {exc}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "request_id": request_id,
        },
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "detail": "An unexpected error occurred",
            "request_id": request_id,
            "timestamp": time.time(),
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.APP_HOST,
        port=config.APP_PORT,
        log_config=None,  # Use our JSON logger
    )
