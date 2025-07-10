import redis
import json
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from loguru import logger
from functools import lru_cache
import os
from pydantic import BaseModel

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 1))  # Use DB 1 for cache (DB 0 is for Celery)
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

# Cache configuration
CACHE_TTL = 5 * 60  # 5 minutes in seconds
TEMPLATE_CACHE_KEY_PREFIX = "templates:"

logger.add("logs/redis_cache.log", rotation="10 MB", level="DEBUG")

class CacheStats(BaseModel):
    hit_count: int = 0
    miss_count: int = 0
    last_refresh: Optional[datetime] = None
    cache_size: int = 0

class RedisCacheService:
    """Redis cache service for template data"""
    
    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._initialized = False
        self.stats = CacheStats()
    
    async def initialize(self):
        """Initialize Redis connection"""
        if not self._initialized:
            try:
                # Create Redis client
                self._client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    password=REDIS_PASSWORD,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                
                # Test connection
                await asyncio.to_thread(self._client.ping)
                self._initialized = True
                logger.info(f"✅ Redis cache service initialized on {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize Redis cache service: {e}")
                logger.warning("🔄 Falling back to no-cache mode")
                self._client = None
                self._initialized = True  # Mark as initialized to prevent retries
    
    @property
    def client(self) -> redis.Redis:
        """Get Redis client"""
        if not self._initialized:
            raise RuntimeError("RedisCacheService not initialized. Call await initialize() first.")
        if self._client is None:
            # Return a mock client that does nothing
            class MockRedisClient:
                def get(self, key):
                    return None
                def setex(self, key, ttl, value):
                    return True
                def delete(self, key):
                    return 0
                def ping(self):
                    return True
                def info(self, section=None):
                    return {"used_memory": 0, "connected_clients": 0, "used_memory_human": "0B"}
                def dbsize(self):
                    return 0
                def keys(self, pattern):
                    return []
                def ttl(self, key):
                    return -1
                def pipeline(self):
                    class MockPipeline:
                        def setex(self, key, ttl, value):
                            return self
                        def delete(self, key):
                            return self
                        def execute(self):
                            return [0, 0]
                    return MockPipeline()
            
            return MockRedisClient()
        return self._client
    
    def _get_user_templates_key(self, user_id: str) -> str:
        """Get cache key for user's templates"""
        return f"{TEMPLATE_CACHE_KEY_PREFIX}user:{user_id}"
    
    def _get_user_cache_meta_key(self, user_id: str) -> str:
        """Get cache metadata key for user"""
        return f"{TEMPLATE_CACHE_KEY_PREFIX}meta:user:{user_id}"
    
    async def get_user_templates(self, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached templates for a user"""
        try:
            cache_key = self._get_user_templates_key(user_id)
            cached_data = await asyncio.to_thread(self.client.get, cache_key)
            
            if cached_data:
                templates = json.loads(cached_data)
                self.stats.hit_count += 1
                logger.debug(f"Cache HIT for user {user_id}: {len(templates)} templates")
                return templates
            else:
                self.stats.miss_count += 1
                logger.debug(f"Cache MISS for user {user_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting cached templates for user {user_id}: {e}")
            self.stats.miss_count += 1
            return None
    
    async def set_user_templates(self, user_id: str, templates: List[Dict[str, Any]]) -> bool:
        """Cache templates for a user"""
        try:
            cache_key = self._get_user_templates_key(user_id)
            meta_key = self._get_user_cache_meta_key(user_id)
            
            # Serialize templates
            templates_json = json.dumps(templates, default=str)
            
            # Store templates and metadata
            pipe = self.client.pipeline()
            pipe.setex(cache_key, CACHE_TTL, templates_json)
            pipe.setex(meta_key, CACHE_TTL, json.dumps({
                "cached_at": datetime.utcnow().isoformat(),
                "template_count": len(templates),
                "expires_at": (datetime.utcnow() + timedelta(seconds=CACHE_TTL)).isoformat()
            }))
            
            await asyncio.to_thread(pipe.execute)
            
            self.stats.last_refresh = datetime.utcnow()
            self.stats.cache_size = len(templates)
            
            logger.info(f"✅ Cached {len(templates)} templates for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching templates for user {user_id}: {e}")
            return False
    
    async def invalidate_user_cache(self, user_id: str) -> bool:
        """Invalidate cached templates for a user"""
        try:
            cache_key = self._get_user_templates_key(user_id)
            meta_key = self._get_user_cache_meta_key(user_id)
            
            pipe = self.client.pipeline()
            pipe.delete(cache_key)
            pipe.delete(meta_key)
            result = await asyncio.to_thread(pipe.execute)
            
            deleted_count = sum(result)
            if deleted_count > 0:
                logger.info(f"🗑️  Invalidated cache for user {user_id} ({deleted_count} keys deleted)")
            
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error invalidating cache for user {user_id}: {e}")
            return False
    
    async def get_cache_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cache information for a user"""
        try:
            meta_key = self._get_user_cache_meta_key(user_id)
            meta_data = await asyncio.to_thread(self.client.get, meta_key)
            
            if meta_data:
                return json.loads(meta_data)
            return None
            
        except Exception as e:
            logger.error(f"Error getting cache info for user {user_id}: {e}")
            return None
    
    async def is_cache_valid(self, user_id: str) -> bool:
        """Check if cache is valid and not expired"""
        try:
            cache_key = self._get_user_templates_key(user_id)
            ttl = await asyncio.to_thread(self.client.ttl, cache_key)
            return ttl > 0
        except Exception as e:
            logger.error(f"Error checking cache validity for user {user_id}: {e}")
            return False
    
    async def get_cache_stats(self) -> CacheStats:
        """Get cache statistics"""
        try:
            # Update cache size from Redis
            info = await asyncio.to_thread(self.client.info, "memory")
            self.stats.cache_size = info.get('used_memory', 0)
            return self.stats
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return self.stats
    
    async def clear_all_cache(self) -> bool:
        """Clear all template cache (admin function)"""
        try:
            # Get all template cache keys
            pattern = f"{TEMPLATE_CACHE_KEY_PREFIX}*"
            keys = await asyncio.to_thread(self.client.keys, pattern)
            
            if keys:
                deleted = await asyncio.to_thread(self.client.delete, *keys)
                logger.info(f"🗑️  Cleared all template cache: {deleted} keys deleted")
                return deleted > 0
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for Redis cache service"""
        try:
            start_time = datetime.utcnow()
            await asyncio.to_thread(self.client.ping)
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            info = await asyncio.to_thread(self.client.info)
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "hit_rate": self._calculate_hit_rate(),
                "cache_keys": await asyncio.to_thread(self.client.dbsize)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "hit_rate": self._calculate_hit_rate()
            }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_requests = self.stats.hit_count + self.stats.miss_count
        if total_requests == 0:
            return 0.0
        return round((self.stats.hit_count / total_requests) * 100, 2)

# Global cache service instance
cache_service: Optional[RedisCacheService] = None

@lru_cache()
def get_cache_service() -> RedisCacheService:
    """Get the Redis cache service singleton"""
    global cache_service
    if cache_service is None:
        cache_service = RedisCacheService()
    return cache_service

async def initialize_cache_service():
    """Initialize the cache service"""
    service = get_cache_service()
    await service.initialize()
    return service

# Sync wrapper functions for use in Celery tasks
def invalidate_user_cache_sync(user_id: str) -> bool:
    """Synchronous wrapper for cache invalidation (for use in Celery tasks)"""
    try:
        service = get_cache_service()
        if not service._initialized:
            # Initialize synchronously for Celery
            import redis
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            client.ping()  # Test connection
            
            # Create temporary service for this operation
            temp_service = RedisCacheService()
            temp_service._client = client
            temp_service._initialized = True
            
            cache_key = temp_service._get_user_templates_key(user_id)
            meta_key = temp_service._get_user_cache_meta_key(user_id)
            
            pipe = client.pipeline()
            pipe.delete(cache_key)
            pipe.delete(meta_key)
            result = pipe.execute()
            
            deleted_count = sum(result)
            if deleted_count > 0:
                logger.info(f"🗑️  [SYNC] Invalidated cache for user {user_id} ({deleted_count} keys deleted)")
            
            return deleted_count > 0
        else:
            # Use the existing initialized service
            cache_key = service._get_user_templates_key(user_id)
            meta_key = service._get_user_cache_meta_key(user_id)
            
            pipe = service.client.pipeline()
            pipe.delete(cache_key)
            pipe.delete(meta_key)
            result = pipe.execute()
            
            deleted_count = sum(result)
            if deleted_count > 0:
                logger.info(f"🗑️  [SYNC] Invalidated cache for user {user_id} ({deleted_count} keys deleted)")
            
            return deleted_count > 0
            
    except Exception as e:
        logger.error(f"Error in sync cache invalidation for user {user_id}: {e}")
        return False

def refresh_user_cache_sync(user_id: str, templates_data: List[Dict[str, Any]]) -> bool:
    """Synchronous wrapper for cache refresh (for use in Celery tasks)"""
    try:
        service = get_cache_service()
        if not service._initialized:
            # Initialize synchronously for Celery
            import redis
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            client.ping()  # Test connection
            
            # Create temporary service for this operation
            temp_service = RedisCacheService()
            temp_service._client = client
            temp_service._initialized = True
            
            cache_key = temp_service._get_user_templates_key(user_id)
            meta_key = temp_service._get_user_cache_meta_key(user_id)
            
            # Serialize templates
            templates_json = json.dumps(templates_data, default=str)
            
            # Store templates and metadata
            pipe = client.pipeline()
            pipe.setex(cache_key, CACHE_TTL, templates_json)
            pipe.setex(meta_key, CACHE_TTL, json.dumps({
                "cached_at": datetime.utcnow().isoformat(),
                "template_count": len(templates_data),
                "expires_at": (datetime.utcnow() + timedelta(seconds=CACHE_TTL)).isoformat()
            }))
            
            pipe.execute()
            
            logger.info(f"✅ [SYNC] Refreshed cache for user {user_id}: {len(templates_data)} templates")
            return True
        else:
            # Use the existing initialized service
            cache_key = service._get_user_templates_key(user_id)
            meta_key = service._get_user_cache_meta_key(user_id)
            
            # Serialize templates
            templates_json = json.dumps(templates_data, default=str)
            
            # Store templates and metadata
            pipe = service.client.pipeline()
            pipe.setex(cache_key, CACHE_TTL, templates_json)
            pipe.setex(meta_key, CACHE_TTL, json.dumps({
                "cached_at": datetime.utcnow().isoformat(),
                "template_count": len(templates_data),
                "expires_at": (datetime.utcnow() + timedelta(seconds=CACHE_TTL)).isoformat()
            }))
            
            pipe.execute()
            
            logger.info(f"✅ [SYNC] Refreshed cache for user {user_id}: {len(templates_data)} templates")
            return True
            
    except Exception as e:
        logger.error(f"Error in sync cache refresh for user {user_id}: {e}")
        return False 