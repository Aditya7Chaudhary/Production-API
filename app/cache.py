import hashlib
import time
from typing import Optional

class ResponseCache:
    def __init__(self,ttl: int = 3600):
        self.ttl = ttl
        self._cache: dict[str, dict] = {}
        self._hit = 0
        self._misses = 0
        
    def check_health(self) -> str:
        """
        Pings the cache server to ensure the connection is alive.
        """
        try:
            # Mocking a cache ping. In real life: self.redis.ping()
            cache_connected = True 
            
            if cache_connected:
                return "healthy"
            return "unhealthy"
        except Exception:
            return "unhealthy"

    def _generate_cache_key(self, query: str) -> str:
        normalized_query = query.strip().lower()
        return hashlib.sha256(normalized_query.encode()).hexdigest()

    def get(self, query: str)  -> Optional[str]:
        cache_key = self._generate_cache_key(query)
        cached_response = self._cache.get(cache_key)
        if cached_response and (time.time() - cached_response['timestamp'] < 3600):  # Cache valid for 1 hour
            self._hit += 1
            return cached_response['data']
        else:
            # Safely removes the key if it exists; does nothing if it's already gone
            self._cache.pop(cache_key, None) # Remove expired cache entry
        self._misses += 1
        return None

    def set(self, query: str, data: str):
        cache_key = self._generate_cache_key(query)
        self._cache[cache_key] = {
            'data': data,
            'timestamp': time.time(),
            'query': query
        }
        
    @property
    def stats(self) -> dict:
        total_requests = self._hit + self._misses
        hit_rate = (self._hit / total_requests) * 100 if total_requests > 0 else 0
        return {
            'hits': self._hit,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'cached_entries': len(self._cache),
        }