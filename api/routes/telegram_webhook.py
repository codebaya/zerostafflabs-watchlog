"""Telegram webhook — handles inline button callbacks for alert feedback."""
import logging

from fastapi import APIRouter, Request
from sqlalchemy.ext.asyncio import AsyncSession

from storage.database import async_session, mark_alert_feedback

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/telegram", tags=["telegram"])


@router.post("/webhook")
async def telegram_webhook(request: Request):
    """Receive Telegram bot updates and process callback_query."""
    data = await request.json()
    cq = data.get("callback_query")
    if not cq:
        return {"ok": True}

    callback_data: str = cq.get("data", "")
    message_id = cq.get("message", {}).get("message_id")

    try:
        action, alert_id_str = callback_data.split(":", 1)
        alert_id = int(alert_id_str)
        is_fp = action == "fp"

        async with async_session() as db:
            await mark_alert_feedback(db, alert_id, is_fp)

        logger.info(
            "Telegram feedback: alert_id=%d false_positive=%s (msg_id=%s)",
            alert_id, is_fp, message_id,
        )
    except (ValueError, KeyError) as exc:
        logger.warning("Unexpected callback data '%s': %s", callback_data, exc)

    return {"ok": True}
