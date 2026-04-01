"""CLI entry point: python3 -m simulator [mode] [options]."""

from __future__ import annotations

import argparse
import logging

from simulator.runner import SimulationRunner
from simulator.scenarios import Scenario


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="simulator",
        description="Compass headless simulator: verify the decision stack without a game client.",
    )
    p.add_argument(
        "mode",
        nargs="?",
        default="replay",
        choices=["benchmark", "replay", "converge"],
        help="benchmark: realtime 10 Hz timing | replay: fast scenario run | converge: multi-session learning",
    )
    p.add_argument(
        "--scenario",
        default="camp_session",
        help="Built-in name (camp_session, survival_stress, exploration) or path to JSON file",
    )
    p.add_argument("--sessions", type=int, default=5, help="Sessions for converge mode (default: 5)")
    p.add_argument("--trace", action="store_true", help="Include per-tick trace in output")
    p.add_argument("--output", metavar="PATH", help="Write results as JSON to file")
    p.add_argument("--utility-phase", type=int, default=2, help="Utility scoring phase 0-4")
    p.add_argument("--no-goap", action="store_true", help="Disable GOAP planner")
    p.add_argument("--quiet", action="store_true", help="Suppress terminal output")
    return p.parse_args()


def _load_scenario(name: str) -> Scenario:
    if name == "camp_session":
        return Scenario.camp_session()
    if name == "survival_stress":
        return Scenario.survival_stress()
    if name == "exploration":
        return Scenario.exploration()
    return Scenario.from_json(name)


def _run_converge(runner: SimulationRunner, scenario: Scenario, sessions: int, quiet: bool) -> str:
    """Run convergence mode and format output."""
    results = runner.run_convergence(scenario, sessions=sessions)

    lines = [
        "",
        f"Convergence: {scenario.name} x {sessions} sessions",
        "",
        f"{'Session':>8}  {'Grade':>5}  {'Ticks':>6}  {'p99 ms':>7}  "
        f"{'Fights':>6}  {'Avg Dur':>8}  {'Drift':>6}  {'GOAP %':>6}",
        "-" * 70,
    ]

    for i, r in enumerate(results, 1):
        grade = r.scorecard.get("grade", "?")
        fights = sum(s.get("fights", 0) for s in r.fight_stats.values())
        avg_dur_vals = [s["avg_duration"] for s in r.fight_stats.values() if s.get("fights", 0) > 0]
        avg_dur = sum(avg_dur_vals) / len(avg_dur_vals) if avg_dur_vals else 0.0
        max_drift = max((abs(v) for v in r.weight_drift.values()), default=0.0)
        goap_pct = r.goap_completion_rate

        lines.append(
            f"{i:>8}  {grade:>5}  {r.total_ticks:>6}  {r.tick_ms_p99:>6.1f}  "
            f"{fights:>6}  {avg_dur:>7.1f}s  {max_drift * 100:>5.1f}%  {goap_pct:>5.0f}%"
        )

    # Summary
    first, last = results[0], results[-1]
    first_dur = next((s["avg_duration"] for s in first.fight_stats.values()), 0)
    last_dur = next((s["avg_duration"] for s in last.fight_stats.values()), 0)
    improvement = ((first_dur - last_dur) / first_dur * 100) if first_dur > 0 else 0

    lines.extend(
        [
            "",
            f"Fight duration: {first_dur:.1f}s -> {last_dur:.1f}s ({improvement:.0f}% improvement)",
            f"Grade: {first.scorecard.get('grade', '?')} -> {last.scorecard.get('grade', '?')}",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    args = _parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
    else:
        logging.basicConfig(level=logging.CRITICAL)

    scenario = _load_scenario(args.scenario)
    runner = SimulationRunner(
        utility_phase=args.utility_phase,
        enable_goap=not args.no_goap,
    )

    header = f"Compass Headless Simulator - {args.mode} mode"

    if args.mode == "benchmark":
        if not args.quiet:
            print(header)
            print(f"Scenario: {scenario.name} ({scenario.tick_count} ticks)")
            print("Running with realtime tick pacing (10 Hz)...\n")
        result = runner.run(scenario, realtime=True, trace=args.trace)
        if not args.quiet:
            print(result.summary())

    elif args.mode == "converge":
        if not args.quiet:
            print(header)
        output = _run_converge(runner, scenario, args.sessions, args.quiet)
        if not args.quiet:
            print(output)
        # For JSON output, serialize the last session's result
        result = runner.run(scenario)  # final snapshot

    else:  # replay
        if not args.quiet:
            print(header)
            print(f"Scenario: {scenario.name} ({scenario.tick_count} ticks)\n")
        result = runner.run(scenario, realtime=False, trace=args.trace)
        if not args.quiet:
            print(result.summary())

    # JSON export
    if args.output:
        if args.mode == "converge":
            # Re-run to get clean result for export
            import json

            results = runner.run_convergence(scenario, sessions=args.sessions)
            data = [r.to_dict() for r in results]
            with open(args.output, "w") as f:
                json.dump(data, f, indent=2)
        else:
            with open(args.output, "w") as f:
                f.write(result.to_json())
        if not args.quiet:
            print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
