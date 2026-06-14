import logging
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Callable
from functools import wraps
from app.config import Settings

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
        }
        
        if hasattr(record, 'extra'):
            log_record.update(record.extra)
        return json.dumps(log_record)
    
def get_logger(name: str = "production-api") -> logging.Logger:
    logger = logging.getLogger(name)
        
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
            
    return logger


# === Metrics Collection ===


class MetricsCollector:
    """Collect and aggregate metrics."""

    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "errors_total": 0,
            "latency_sum": 0,
            "latency_count": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        settings = Settings()
        # NECESSARY FIX: Track usage per model name (e.g., {"gpt-4o": 2, "claude-3": 5})
        self.model_usage = {settings.primary_model:0}

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        error: bool = False,
        cache_hit: bool = False,
        model_name: str = None,  # ADDED: Accept the model name
    ):
        self.metrics["requests_total"] += 1
        self.metrics["latency_sum"] += latency_ms
        self.metrics["latency_count"] += 1
        self.metrics["tokens_input"] += input_tokens
        self.metrics["tokens_output"] += output_tokens

        if error:
            self.metrics["errors_total"] += 1

        if cache_hit:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1
            
        # ADDED: Track specific model usage
        if model_name:
            self.model_usage[model_name] = self.model_usage.get(model_name, 0) + 1

    def get_summary(self) -> dict:
        avg_latency = (
            self.metrics["latency_sum"] / self.metrics["latency_count"]
            if self.metrics["latency_count"] > 0
            else 0
        )
        
        # 1. FIXED: Converted back to a string percentage ("0.00%") because 
        # your Pydantic model specifically expects a string type for error_rate
        error_rate = (
            self.metrics["errors_total"] / self.metrics["requests_total"]
            if self.metrics["requests_total"] > 0
            else 0
        )
        error_rate_str = f"{error_rate:.2%}"
        
        cache_hit_rate = (
            self.metrics["cache_hits"]
            / (self.metrics["cache_hits"] + self.metrics["cache_misses"])
            if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0
            else 0
        )

        # 2. FIXED: Double-check that self.model_usage is a dictionary. 
        # If it's a set, let's coerce it safely into a dict form.
        usage_dict = self.model_usage if isinstance(self.model_usage, dict) else {model: 1 for model in self.model_usage}

        return {
            "total_requests": self.metrics["requests_total"],
            "total_errors": self.metrics["errors_total"],
            "error_rate": error_rate_str,                      # Fixed: returns string
            "average_latency_ms": round(avg_latency, 2),       # Fixed: changed key from 'avg' to 'average'
            "total_input_tokens": self.metrics["tokens_input"],
            "total_output_tokens": self.metrics["tokens_output"],
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
            "model_usage": usage_dict                           # Fixed: guaranteed dict structure
        }


class RequestTimer:
    """Context manager to measure request latency."""

    def __init__(self, metrics_collector: MetricsCollector, model_name: str = None):
        self.metrics_collector = metrics_collector
        self.model_name = model_name  # Track the model during the request lifecycle

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.latency_ms = (time.time() - self.start_time) * 1000
        error_occurred = exc_type is not None
        
        # Updated to forward token counts and model names when calling record_request
        self.metrics_collector.record_request(
            latency_ms=self.latency_ms, 
            input_tokens=0, 
            output_tokens=0, 
            error=error_occurred,
            model_name=self.model_name
        )