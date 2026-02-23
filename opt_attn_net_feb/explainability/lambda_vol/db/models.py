from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for Lambda-Vol store."""


class PressureRunORM(Base):
    __tablename__ = "lv_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    config_json: Mapped[str] = mapped_column(Text, nullable=False)


class PressureEpochORM(Base):
    __tablename__ = "lv_pressure_epoch"
    __table_args__ = (
        UniqueConstraint("run_id", "epoch", "task_id", "concept_id", name="uq_lv_pressure_epoch"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False)
    concept_id: Mapped[str] = mapped_column(String(128), nullable=False)

    tcav: Mapped[float] = mapped_column(Float, nullable=False)
    tcav_smoothed: Mapped[float] = mapped_column(Float, nullable=False)
    delta_tcav: Mapped[float] = mapped_column(Float, nullable=False)
    attention_support: Mapped[float] = mapped_column(Float, nullable=False)
    prevalence: Mapped[float] = mapped_column(Float, nullable=False)
    rho: Mapped[float] = mapped_column(Float, nullable=False)
    drift: Mapped[float] = mapped_column(Float, nullable=False)

    regime_label: Mapped[str] = mapped_column(String(64), nullable=False)

    regime_core: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    feedback: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    trend_loop: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    revert_loop: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    dissipation: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    context_term: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class TaskEpochORM(Base):
    __tablename__ = "lv_task_epoch"
    __table_args__ = (
        UniqueConstraint("run_id", "epoch", "task_id", name="uq_lv_task_epoch"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False)

    attention_entropy: Mapped[float] = mapped_column(Float, nullable=False)
    witness_rate: Mapped[float] = mapped_column(Float, nullable=False)
    train_metric: Mapped[float] = mapped_column(Float, nullable=False)
    val_metric: Mapped[float] = mapped_column(Float, nullable=False)
    loss: Mapped[float] = mapped_column(Float, nullable=False)
    calibration_error: Mapped[float] = mapped_column(Float, nullable=False)
    concentration_entropy: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    concentration_gini: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    topk_mass: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    regime_label: Mapped[str] = mapped_column(String(64), nullable=False)
    context_json: Mapped[str] = mapped_column(Text, nullable=False)


class PressureAlertORM(Base):
    __tablename__ = "lv_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    code: Mapped[str] = mapped_column(String(128), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    task_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    concept_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details_json: Mapped[str] = mapped_column(Text, nullable=False)


class RecommendationORM(Base):
    __tablename__ = "lv_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    recommendation_type: Mapped[str] = mapped_column(String(128), nullable=False)
    action_level: Mapped[str] = mapped_column(String(64), nullable=False)
    task_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    concept_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    params_json: Mapped[str] = mapped_column(Text, nullable=False)


__all__ = [
    "Base",
    "PressureAlertORM",
    "PressureEpochORM",
    "PressureRunORM",
    "RecommendationORM",
    "TaskEpochORM",
]
