"""Simulation result aggregation and export."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class SimulationResult:
    """Telemetry collected from one simulation run."""

    scenario_name: str = ""
    total_ticks: int = 0
    wall_time_s: float = 0.0

    # Tick timing (milliseconds)
    tick_times_ms: list[float] = field(default_factory=list)

    # Decision trace
    rule_counts: dict[str, int] = field(default_factory=dict)
    routine_counts: dict[str, int] = field(default_factory=dict)
    transitions: int = 0

    # GOAP
    goap_plans_generated: int = 0
    goap_plans_completed: int = 0
    goap_plans_invalidated: int = 0
    goap_avg_cost_error: float = 0.0

    # Learning snapshots (populated at end of run)
    fight_stats: dict[str, dict] = field(default_factory=dict)
    weight_drift: dict[str, float] = field(default_factory=dict)
    scorecard: dict[str, int | float | str] = field(default_factory=dict)

    # Optional per-tick trace
    tick_trace: list[dict] | None = None

    def record_tick(
        self,
        tick_ms: float,
        rule: str,
        routine: str,
        phase: str,
        actions: list[str],
        goap_plan: str | None = None,
    ) -> None:
        """Record one tick of telemetry."""
        self.total_ticks += 1
        self.tick_times_ms.append(tick_ms)
        if rule:
            self.rule_counts[rule] = self.rule_counts.get(rule, 0) + 1
        if routine:
            self.routine_counts[routine] = self.routine_counts.get(routine, 0) + 1
        if self.tick_trace is not None:
            self.tick_trace.append(
                {
                    "tick": self.total_ticks,
                    "ms": round(tick_ms, 3),
                    "rule": rule,
                    "routine": routine,
                    "phase": phase,
                    "actions": actions,
                    "goap_plan": goap_plan,
                }
            )

    # -- Computed properties --

    def _sorted_times(self) -> list[float]:
        if not self.tick_times_ms:
            return [0.0]
        return sorted(self.tick_times_ms)

    @property
    def tick_ms_p50(self) -> float:
        t = self._sorted_times()
        return t[len(t) // 2]

    @property
    def tick_ms_p95(self) -> float:
        t = self._sorted_times()
        return t[int(len(t) * 0.95)]

    @property
    def tick_ms_p99(self) -> float:
        t = self._sorted_times()
        return t[int(len(t) * 0.99)]

    @property
    def tick_ms_max(self) -> float:
        return max(self.tick_times_ms) if self.tick_times_ms else 0.0

    @property
    def ticks_over_100ms(self) -> int:
        return sum(1 for t in self.tick_times_ms if t > 100.0)

    @property
    def goap_completion_rate(self) -> float:
        if self.goap_plans_generated == 0:
            return 0.0
        return self.goap_plans_completed / self.goap_plans_generated * 100

    # -- Export --

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        d: dict = {
            "scenario": self.scenario_name,
            "total_ticks": self.total_ticks,
            "wall_time_s": round(self.wall_time_s, 2),
            "timing": {
                "p50_ms": round(self.tick_ms_p50, 3),
                "p95_ms": round(self.tick_ms_p95, 3),
                "p99_ms": round(self.tick_ms_p99, 3),
                "max_ms": round(self.tick_ms_max, 3),
                "over_100ms": self.ticks_over_100ms,
            },
            "decisions": {
                "rule_counts": self.rule_counts,
                "routine_counts": self.routine_counts,
                "transitions": self.transitions,
            },
            "goap": {
                "plans_generated": self.goap_plans_generated,
                "plans_completed": self.goap_plans_completed,
                "plans_invalidated": self.goap_plans_invalidated,
                "completion_rate_pct": round(self.goap_completion_rate, 1),
                "avg_cost_error_s": round(self.goap_avg_cost_error, 1),
            },
            "learning": {
                "fight_stats": self.fight_stats,
                "weight_drift": {k: round(v, 4) for k, v in self.weight_drift.items()},
                "scorecard": self.scorecard,
            },
        }
        if self.tick_trace is not None:
            d["tick_trace"] = self.tick_trace
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Human-readable terminal summary."""
        lines = [
            "",
            f"Scenario: {self.scenario_name} ({self.total_ticks} ticks, {self.wall_time_s:.1f}s wall time)",
            "",
            "Tick Timing",
            f"  p50:  {self.tick_ms_p50:.1f}ms",
            f"  p95:  {self.tick_ms_p95:.1f}ms",
            f"  p99:  {self.tick_ms_p99:.1f}ms",
            f"  max:  {self.tick_ms_max:.1f}ms",
            f"  over 100ms: {self.ticks_over_100ms} "
            f"({self.ticks_over_100ms / max(self.total_ticks, 1) * 100:.1f}%)",
            "",
            "Decision Stack",
        ]
        top_rules = sorted(self.rule_counts.items(), key=lambda x: -x[1])
        parts = [f"{name}: {count}" for name, count in top_rules[:8]]
        lines.append(f"  {' | '.join(parts)}")
        lines.append(f"  Transitions: {self.transitions}")

        if self.goap_plans_generated > 0:
            lines.extend(
                [
                    "",
                    "GOAP Planner",
                    f"  Plans: {self.goap_plans_generated} generated, "
                    f"{self.goap_plans_completed} completed, "
                    f"{self.goap_plans_invalidated} invalidated "
                    f"({self.goap_completion_rate:.0f}% completion)",
                    f"  Avg cost error: {self.goap_avg_cost_error:+.1f}s",
                ]
            )

        if self.scorecard:
            grade = self.scorecard.get("grade", "?")
            overall = self.scorecard.get("overall", 0)
            lines.extend(["", f"Scorecard: {grade} ({overall}/100)"])

        budget_ok = self.ticks_over_100ms == 0
        lines.extend(
            [
                "",
                f"{'PASS' if budget_ok else 'FAIL'}: "
                f"{'All' if budget_ok else 'Not all'} ticks under 100ms budget",
            ]
        )
        return "\n".join(lines)
