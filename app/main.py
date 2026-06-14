import time
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langsmith import traceable
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app.config import get_settings
from app.models import (
    ChatRequest, HealthResponse,
    ChatResponse, MetricsResponse, ErrorResponse
)
from app.security import SecurityPipeline
from app.cache import ResponseCache
from app.monitoring import MetricsCollector, get_logger, RequestTimer
from app.agent import ProductionAgent

load_dotenv()
cache: ResponseCache = None
logger = get_logger()
security: SecurityPipeline = None
agent: ProductionAgent = None
metrics: MetricsCollector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize components
    global security, cache, metrics, agent
    
    settings = get_settings()
    
    logger.info("Starting up application....", extra={"extra_data": {
        "environment": settings.app_env,
        "primary_model": settings.primary_model,
        "tracing_enabled": settings.langchain_tracing_v2,
    }})
    
    cache = ResponseCache(ttl=settings.cache_ttl_seconds)
    metrics = MetricsCollector()
    security = SecurityPipeline()
    agent = ProductionAgent()

    logger.info("Application startup complete.")
    
    yield

    logger.info("Shutting down application....", extra={"extra_data": metrics.get_summary()})
    
    
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
app = FastAPI(
    title="Production API",
    description="A production-ready API for handling chat requests with security, caching, and monitoring.",
    version="1.0.0",
    lifespan=lifespan
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )
    
    
@app.post("/chat", response_model=ChatResponse, responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}})
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint")
async def chat(request: Request, body: ChatRequest):
    with RequestTimer(metrics_collector=metrics) as timer:
        security_notes = []

        is_allowed, cleaned_message, notes = security.process(body.message)
        security_notes.extend(notes)
        
        if not is_allowed:
            logger.warning("Request blocked by security pipeline.", extra={"extra_data": {"reason": "input_validation", "notes": security_notes}})
            raise HTTPException(status_code=400, detail="Input message failed security checks.")
        
        cache_key = f"chat:{cleaned_message}"
        cached = cache.get(cache_key)
        if cached:
            logger.info("Cache hit for message.", extra={"extra_data": {"message": cleaned_message}})
            if hasattr(metrics, 'record_cache_hit'):
                metrics.record_cache_hit()
            return ChatResponse(
                response=cached, 
                cached=True, 
                thread_id=body.thread_id,
                processing_time_ms=0.0,  # Cache hits are instant
                model_used="cache"
            )
            
        try:
            # Executes the graph asynchronously
            result = await agent.graph.ainvoke({"messages": [HumanMessage(content=cleaned_message)]})
        except Exception as e:
            import traceback
            print("--- CRITICAL AGENT CRASH TRACEBACK ---")
            traceback.print_exc()
            print("--------------------------------------")
            raise HTTPException(status_code=500, detail=f"Agent Crash Details: {str(e)}")
        
        # FIXED: Extract the real message response from the LangGraph message state stream
        final_messages = result.get("messages", [])
        response_used = final_messages[-1].content if final_messages else ""
        model_used = result.get("model_used", "unknown")
        
        is_valid, validated_response, output_warnings = security.validator.validate(response_used)
        if output_warnings:
            security_notes.append(output_warnings)
        
        # FIXED: Save using 'cache_key' (chat:message) instead of raw message text
        cache.set(cache_key, validated_response)
        
        # --- INSIDE THE WITH BLOCK INDENTATION ---
        input_tokens = int(len(cleaned_message.split()) * 1.3)
        output_tokens = int(len(validated_response.split()) * 1.3)
        
        current_latency = (time.time() - timer.start_time) * 1000
        
        metrics.record_request(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=current_latency,
            cache_hit=False,
            model_name=model_used
        )
        
        if security_notes:
            logger.info("Security notes for request.", extra={"extra_data": {
                "notes": security_notes,
                "thread_id": body.thread_id,
            }})
            
        logger.info("Request processed successfully.", extra={"extra_data": {
            "model_used": model_used,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency": current_latency,
            "thread_id": body.thread_id,
        }})
        
        return ChatResponse(
            response=validated_response,
            cached=False, 
            thread_id=body.thread_id,
            processing_time_ms=round(current_latency, 2),
            model_used=model_used
        )
    
@app.get("/")
def read_root():
    return {"message": "Hello World, the API is alive!"}
    
@app.get("/health", response_model=HealthResponse)
async def health():
    checks = {
        "agent": agent.check_health(),
        "cache": cache.check_health(),
        "security": security.check_health()
    }
    
    # FIXED: Properly extract status evaluation metrics
    all_healthy = all(status == "healthy" for status in checks.values())

    return HealthResponse(
        status="ok" if all_healthy else "degraded", 
        environment=get_settings().app_env, 
        checks=checks
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    summary = metrics.get_summary()
    return MetricsResponse(**summary)
    
@app.get("/cache/stats")
async def cache_stats():
    stats = cache.stats
    return JSONResponse(content=stats)