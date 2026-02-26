import json
from typing import Any

from db import get_conn


def queue_crm_sync(tenant_id: str, lead_id: int, lead_payload: dict[str, Any]) -> int:
    crm_target = "hubspot" if lead_payload.get("score", 0) >= 70 else "salesforce"

    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO crm_outbox (tenant_id, lead_id, crm_target, payload, status)
            VALUES (?, ?, ?, ?, 'pending')
            """,
            (tenant_id, lead_id, crm_target, json.dumps(lead_payload)),
        )
        return int(cursor.lastrowid)
