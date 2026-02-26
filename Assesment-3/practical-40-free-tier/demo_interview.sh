#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8010}"
TENANT_NAME="${TENANT_NAME:-Interview Demo Tenant}"
PLATFORM="facebook"
EXTERNAL_USER_ID="fb_demo_user"
EXTERNAL_MESSAGE_ID="msg_demo_001"
MESSAGE_TEXT="${MESSAGE_TEXT:-Hi, we need enterprise pricing and a product demo this week.}"

echo "[1/6] Creating tenant..."
TENANT_JSON=$(curl -sS -X POST "$BASE_URL/tenants" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"$TENANT_NAME\"}")

TENANT_ID=$(TENANT_JSON="$TENANT_JSON" python3 - <<'PY'
import json, os
obj = json.loads(os.environ["TENANT_JSON"])
print(obj["id"])
PY
)
API_KEY=$(TENANT_JSON="$TENANT_JSON" python3 - <<'PY'
import json, os
obj = json.loads(os.environ["TENANT_JSON"])
print(obj["api_key"])
PY
)
WEBHOOK_SECRET=$(TENANT_JSON="$TENANT_JSON" python3 - <<'PY'
import json, os
obj = json.loads(os.environ["TENANT_JSON"])
print(obj["webhook_secret"])
PY
)

echo "Tenant ID: $TENANT_ID"

echo "[2/6] Getting JWT token..."
TOKEN_JSON=$(curl -sS -X POST "$BASE_URL/auth/token" \
  -H "Content-Type: application/json" \
  -d "{\"tenant_id\":\"$TENANT_ID\",\"api_key\":\"$API_KEY\"}")

TOKEN=$(TOKEN_JSON="$TOKEN_JSON" python3 - <<'PY'
import json, os
print(json.loads(os.environ["TOKEN_JSON"])["access_token"])
PY
)

echo "[3/6] Creating webhook signature..."
SIGNATURE=$(TENANT_ID="$TENANT_ID" PLATFORM="$PLATFORM" EXTERNAL_USER_ID="$EXTERNAL_USER_ID" EXTERNAL_MESSAGE_ID="$EXTERNAL_MESSAGE_ID" MESSAGE_TEXT="$MESSAGE_TEXT" WEBHOOK_SECRET="$WEBHOOK_SECRET" python3 - <<'PY'
import hmac, hashlib
import os
tenant_id = os.environ["TENANT_ID"]
platform = os.environ["PLATFORM"]
external_user_id = os.environ["EXTERNAL_USER_ID"]
external_message_id = os.environ["EXTERNAL_MESSAGE_ID"]
text = os.environ["MESSAGE_TEXT"]
secret = os.environ["WEBHOOK_SECRET"]
payload = f"{tenant_id}|{platform}|{external_user_id}|{external_message_id}|{text}"
print(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest())
PY
)

echo "[4/6] Sending signed webhook..."
INGEST_JSON=$(curl -sS -X POST "$BASE_URL/webhooks/$PLATFORM?tenant_id=$TENANT_ID" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Webhook-Signature: $SIGNATURE" \
  -H "Content-Type: application/json" \
  -d "{\"external_user_id\":\"$EXTERNAL_USER_ID\",\"external_message_id\":\"$EXTERNAL_MESSAGE_ID\",\"text\":\"$MESSAGE_TEXT\"}")

MESSAGE_ID=$(INGEST_JSON="$INGEST_JSON" python3 - <<'PY'
import json, os
print(json.loads(os.environ["INGEST_JSON"])["message_id"])
PY
)

echo "Message ID: $MESSAGE_ID"

echo "[5/6] Polling message status..."
for _ in {1..12}; do
  MSG_JSON=$(curl -sS "$BASE_URL/messages/$MESSAGE_ID?tenant_id=$TENANT_ID" \
    -H "Authorization: Bearer $TOKEN")
  STATUS=$(MSG_JSON="$MSG_JSON" python3 - <<'PY'
import json, os
print(json.loads(os.environ["MSG_JSON"]).get("status", "unknown"))
PY
)
  if [[ "$STATUS" == "processed" ]]; then
    break
  fi
  sleep 0.4
done

echo "Final message payload:"
echo "$MSG_JSON"

echo "[6/6] Fetching leads and CRM outbox..."
LEADS_JSON=$(curl -sS "$BASE_URL/tenants/$TENANT_ID/leads" \
  -H "Authorization: Bearer $TOKEN")
OUTBOX_JSON=$(curl -sS "$BASE_URL/tenants/$TENANT_ID/crm/outbox" \
  -H "Authorization: Bearer $TOKEN")

echo "Leads: $LEADS_JSON"
echo "CRM Outbox: $OUTBOX_JSON"

echo
echo "Demo complete ✅"
echo "Tenant: $TENANT_ID"
echo "Message: $MESSAGE_ID"
