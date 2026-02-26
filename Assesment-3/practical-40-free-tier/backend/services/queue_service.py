import os

from redis import Redis
from redis.exceptions import RedisError


REDIS_URL = os.getenv("REDIS_URL", "")
QUEUE_NAME = os.getenv("QUEUE_NAME", "message_queue")
USE_REDIS_QUEUE = os.getenv("USE_REDIS_QUEUE", "false").lower() == "true"


class QueueClient:
    def __init__(self) -> None:
        self.enabled = USE_REDIS_QUEUE and bool(REDIS_URL)
        self._client = Redis.from_url(REDIS_URL, decode_responses=True) if self.enabled else None

    def enqueue_message(self, message_id: int) -> bool:
        if not self._client:
            return False

        try:
            self._client.rpush(QUEUE_NAME, str(message_id))
            return True
        except RedisError:
            return False

    def dequeue_message(self, timeout_seconds: int = 5) -> int | None:
        if not self._client:
            return None

        try:
            result = self._client.blpop(QUEUE_NAME, timeout=timeout_seconds)
            if not result:
                return None
            _, value = result
            return int(value)
        except (RedisError, ValueError):
            return None
