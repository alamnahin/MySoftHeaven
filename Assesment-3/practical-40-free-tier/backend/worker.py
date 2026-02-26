import asyncio
import logging
import os

from db import init_db
from services.pipeline_service import process_message
from services.queue_service import QueueClient


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("worker")


async def run_worker() -> None:
    init_db()
    queue = QueueClient()

    if not queue.enabled:
        logger.error("Worker started but Redis queue is disabled. Set USE_REDIS_QUEUE=true and REDIS_URL.")
        return

    logger.info("Worker started. Waiting for messages...")
    while True:
        message_id = queue.dequeue_message(timeout_seconds=5)
        if message_id is None:
            await asyncio.sleep(0.1)
            continue

        try:
            await process_message(message_id)
            logger.info("Processed message_id=%s", message_id)
        except Exception as error:
            logger.exception("Failed processing message_id=%s error=%s", message_id, error)


if __name__ == "__main__":
    asyncio.run(run_worker())
