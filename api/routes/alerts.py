"""Alert listing and feedback routes."""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.models import AlertFeedback, AlertResponse
from storage.database import get_db, list_alerts, mark_alert_feedback

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=List[AlertResponse])
async def get_alerts(
    camera_id: Optional[int] = Query(default=None),
    limit: int = Query(default=50, le=200),
    db: AsyncSession = Depends(get_db),
):
    alerts = await list_alerts(db, camera_id=camera_id, limit=limit)
    return alerts


@router.post("/{alert_id}/feedback", response_model=AlertResponse)
async def submit_feedback(
    alert_id: int,
    body: AlertFeedback,
    db: AsyncSession = Depends(get_db),
):
    """Mark an alert as false positive or confirmed."""
    alert = await mark_alert_feedback(db, alert_id, body.false_positive)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert
