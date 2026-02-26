import json
import os
import re
from typing import Any

import httpx


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")


def _heuristic_intent(text: str) -> str:
    lower_text = text.lower()
    if any(word in lower_text for word in ["buy", "price", "quote", "demo", "plan", "subscription"]):
        return "sales"
    if any(word in lower_text for word in ["help", "issue", "error", "problem", "support", "not working"]):
        return "support"
    if any(word in lower_text for word in ["win money", "free crypto", "click now", "urgent transfer"]):
        return "spam"
    return "support"


def _heuristic_score(intent: str, text: str) -> int:
    base = {"sales": 75, "support": 45, "spam": 5}.get(intent, 40)
    if len(text) > 200:
        base += 5
    if "budget" in text.lower() or "enterprise" in text.lower():
        base += 10
    return max(0, min(100, base))


def _heuristic_reply(intent: str) -> str:
    if intent == "sales":
        return "Thanks for your interest. Our sales team can share pricing, package options, and a demo schedule shortly."
    if intent == "support":
        return "Thanks for reaching out. Please share your account email and issue details so support can assist quickly."
    return "Thanks for your message. We could not process this request through our normal workflow."


async def _gemini_infer(message_text: str) -> dict[str, Any]:
    prompt = (
        "You are a strict classifier and responder for customer messages. "
        "Return only JSON with keys: intent, lead_score, reply. "
        "intent must be one of sales/support/spam. "
        "lead_score must be integer 0-100. "
        "reply should be <= 40 words and practical.\n\n"
        f"Message: {message_text}"
    )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{LLM_MODEL}:generateContent?key={LLM_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 200,
            "responseMimeType": "application/json",
        },
    }

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    text = data["candidates"][0]["content"]["parts"][0]["text"]
    cleaned = re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE).strip()
    parsed = json.loads(cleaned)

    intent = parsed.get("intent", "support")
    if intent not in {"sales", "support", "spam"}:
        intent = "support"

    score = int(parsed.get("lead_score", 40))
    score = max(0, min(100, score))

    reply = str(parsed.get("reply", "Thanks for your message."))[:400]
    return {"intent": intent, "lead_score": score, "reply": reply}


async def infer_intent_reply_score(message_text: str) -> dict[str, Any]:
    if LLM_PROVIDER == "gemini" and LLM_API_KEY:
        try:
            return await _gemini_infer(message_text)
        except Exception:
            pass

    intent = _heuristic_intent(message_text)
    return {
        "intent": intent,
        "lead_score": _heuristic_score(intent, message_text),
        "reply": _heuristic_reply(intent),
    }
