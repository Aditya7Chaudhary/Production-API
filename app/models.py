from pydantic import BaseModel, Field
from datetime import datetime

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The user's message to the agent."
    )
    thread_id: str = Field(
        default = "default",
        description="Conversation thread ID"
    )
    
class ChatResponse(BaseModel):
    response: str = Field(
        ...,
        description="The agent's response to the user's message."
    )
    thread_id: str = Field(
        ...,
        description="Conversation thread ID"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="The time when the response was generated."
    )
    model_used: str
    cached: bool = False
    processing_time_ms: float
    
class HealthkResponse(BaseModel):
    status: str = "healthy"
    environment: str
    version: str = "1.0.0"
    checks: dict = {}

class MetricsResponse(BaseModel):
    total_requests: int
    average_latency_ms: float
    model_usage: dict
    cache_hit_rate: str
    total_errors: int
    error_rate: str
    total_input_tokens: int
    total_output_tokens: int

class ErrorResponse(BaseModel):
    error: str
    details: str | None = None
    request_id: str | None = None