"""phase 12 evaluation and ablation runs

Revision ID: 0012_phase12_evaluation
Revises: 0011_llm_experiment_runs
"""

from alembic import op
import sqlalchemy as sa


revision = "0012_phase12_evaluation"
down_revision = "0011_llm_experiment_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.String(length=120), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("suite", sa.String(length=40), nullable=False),
        sa.Column("dataset_version", sa.String(length=40), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("total_cases", sa.Integer(), nullable=False),
        sa.Column("passed", sa.Integer(), nullable=False),
        sa.Column("failed", sa.Integer(), nullable=False),
        sa.Column("pass_rate", sa.Float(), nullable=False),
        sa.Column("overall_score", sa.Float(), nullable=False),
        sa.Column("safety_score", sa.Float(), nullable=False),
        sa.Column("evidence_score", sa.Float(), nullable=False),
        sa.Column("agent_trace_score", sa.Float(), nullable=False),
        sa.Column("llm_safety_score", sa.Float(), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("report_json", sa.JSON(), nullable=False),
        sa.Column("markdown_report", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_evaluation_runs_run_id", "evaluation_runs", ["run_id"], unique=True)
    op.create_index("ix_evaluation_runs_actor_user_id", "evaluation_runs", ["actor_user_id"])
    op.create_index("ix_evaluation_runs_status", "evaluation_runs", ["status"])
    op.create_table(
        "evaluation_results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("evaluation_run_id", sa.Integer(), sa.ForeignKey("evaluation_runs.id"), nullable=False),
        sa.Column("case_id", sa.String(length=120), nullable=False),
        sa.Column("dataset", sa.String(length=120), nullable=False),
        sa.Column("category", sa.String(length=80), nullable=False),
        sa.Column("metric_name", sa.String(length=120), nullable=False),
        sa.Column("severity", sa.String(length=40), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("actual_json", sa.JSON(), nullable=False),
        sa.Column("expected_json", sa.JSON(), nullable=False),
        sa.Column("trace_id", sa.String(length=80), nullable=True),
        sa.Column("evidence_packet_id", sa.String(length=80), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    for name in ("evaluation_run_id", "case_id", "dataset", "category", "metric_name", "trace_id", "evidence_packet_id"):
        op.create_index(f"ix_evaluation_results_{name}", "evaluation_results", [name])
    op.create_table(
        "ablation_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ablation_id", sa.String(length=120), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("baseline_run_id", sa.String(length=120), nullable=True),
        sa.Column("experiment_count", sa.Integer(), nullable=False),
        sa.Column("report_json", sa.JSON(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_ablation_runs_ablation_id", "ablation_runs", ["ablation_id"], unique=True)
    op.create_index("ix_ablation_runs_actor_user_id", "ablation_runs", ["actor_user_id"])
    op.create_index("ix_ablation_runs_status", "ablation_runs", ["status"])
    op.create_index("ix_ablation_runs_baseline_run_id", "ablation_runs", ["baseline_run_id"])


def downgrade() -> None:
    op.drop_table("ablation_runs")
    op.drop_table("evaluation_results")
    op.drop_table("evaluation_runs")
