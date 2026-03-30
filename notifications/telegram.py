"""Telegram bot notification sender.

Sends alert snapshots with inline feedback buttons.
Buttons: "오탐이에요 ✓" / "실제 이상 ⚠" — webhook handler in api/routes/telegram.py.
"""
import logging
from pathlib import Path
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)

_API = f"https://api.telegram.org/bot{settings.telegram_bot_token}"


async def send_alert(
    camera_name: str,
    event_type: str,
    description: str,
    confidence: float,
    snapshot_path: Optional[Path],
    alert_id: int,
) -> bool:
    """Send an anomaly alert to the configured Telegram chat.

    Returns True on success.
    """
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.warning("Telegram not configured — skipping alert")
        return False

    caption = (
        f"🚨 *이상 감지* — {camera_name}\n"
        f"유형: `{event_type}`\n"
        f"설명: {description}\n"
        f"신뢰도: {confidence:.0%}"
    )

    # Inline keyboard: false-positive vs confirmed
    reply_markup = {
        "inline_keyboard": [[
            {"text": "오탐이에요 ✓", "callback_data": f"fp:{alert_id}"},
            {"text": "실제 이상 ⚠", "callback_data": f"real:{alert_id}"},
        ]]
    }

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            if snapshot_path and snapshot_path.exists():
                with open(snapshot_path, "rb") as f:
                    resp = await client.post(
                        f"{_API}/sendPhoto",
                        data={
                            "chat_id": settings.telegram_chat_id,
                            "caption": caption,
                            "parse_mode": "Markdown",
                            "reply_markup": str(reply_markup).replace("'", '"'),
                        },
                        files={"photo": f},
                    )
            else:
                resp = await client.post(
                    f"{_API}/sendMessage",
                    json={
                        "chat_id": settings.telegram_chat_id,
                        "text": caption,
                        "parse_mode": "Markdown",
                        "reply_markup": reply_markup,
                    },
                )
            resp.raise_for_status()
            logger.info("Telegram alert sent for alert_id=%d", alert_id)
            return True
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False


async def send_morning_report(
    report_date: str,
    total_alerts: int,
    anomaly_count: int,
    camera_summaries: list[dict],
) -> bool:
    """Send the daily morning summary report."""
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        return False

    lines = [
        f"📋 *WatchLog 야간 리포트* — {report_date}",
        f"",
        f"총 이상 감지: *{anomaly_count}건* / 전체 이벤트 {total_alerts}건",
        "",
    ]

    for cam in camera_summaries:
        lines.append(f"📷 *{cam['name']}*: 이상 {cam['anomalies']}건, 사람 감지 {cam['persons']}건")

    if anomaly_count == 0:
        lines.append("\n✅ 야간 이상 없음. 안전한 밤이었습니다.")
    else:
        lines.append("\n⚠️ 이상 감지 건들을 확인해 주세요.")

    text = "\n".join(lines)
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.post(
                f"{_API}/sendMessage",
                json={
                    "chat_id": settings.telegram_chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                },
            )
            resp.raise_for_status()
            logger.info("Morning report sent for %s", report_date)
            return True
        except Exception as exc:
            logger.error("Telegram morning report failed: %s", exc)
            return False
