from db import get_conn
from services.crm_service import queue_crm_sync
from services.llm_service import infer_intent_reply_score


async def process_message(message_id: int) -> None:
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM inbound_messages WHERE id = ?", (message_id,))
        row = cursor.fetchone()

    if not row:
        return

    tenant_id = row["tenant_id"]
    message_text = row["text"]

    result = await infer_intent_reply_score(message_text)
    intent = result["intent"]
    lead_score = result["lead_score"]
    ai_reply = result["reply"]

    with get_conn() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE inbound_messages
            SET status = 'processed', intent = ?, ai_reply = ?, lead_score = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (intent, ai_reply, lead_score, message_id),
        )

        cursor.execute(
            """
            INSERT INTO leads (tenant_id, message_id, intent, score, status)
            VALUES (?, ?, ?, ?, 'new')
            """,
            (tenant_id, message_id, intent, lead_score),
        )
        if cursor.lastrowid is None:
            return
        lead_id = int(cursor.lastrowid)

    queue_crm_sync(
        tenant_id=tenant_id,
        lead_id=lead_id,
        lead_payload={
            "message_id": message_id,
            "intent": intent,
            "score": lead_score,
            "ai_reply": ai_reply,
        },
    )
