"""Analyze a session's structured event log (.jsonl).

Usage: py -3 -m util.analyze_session [path_to.jsonl]
       py -3 -m util.analyze_session --compare old.jsonl new.jsonl
       If no path given, reads the most recent .jsonl in logs/sessions/.

Analysis follows CLAUDE.md guidelines:
  1. Timeline reconstruction
  2. Moving window analysis (5-min defeat rate trends)
  3. Defeat drought detection with context
  4. Failure chain analysis
  5. Phase timing (defeat cycle breakdown)
  6. Efficiency metrics
  7. Spatial analysis (camp distance, drift)
  8. Mana utilization
  9. Wander effectiveness
  10. Actionable recommendations
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_events(path: str) -> list[dict]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _p(text: str) -> None:
    """Print with ASCII fallback for Windows cp1252 terminals."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def _print_fight_analysis(fights: list[dict]) -> None:
    """Print fight statistics and per-npc breakdown."""
    _p(f"\n-- Fight Analysis ({len(fights)} fights) --")
    durations = [f["duration"] for f in fights]
    mana_costs = [f["mana_spent"] for f in fights]
    cast_counts = [f["casts"] for f in fights]
    hp_deltas = [f["hp_end"] - f["hp_start"] for f in fights]

    _p(
        f"  Duration:   avg={sum(durations) / len(durations):.1f}s  "
        f"min={min(durations):.1f}s  max={max(durations):.1f}s"
    )
    _p(
        f"  Mana/fight: avg={sum(mana_costs) / len(mana_costs):.0f}  "
        f"min={min(mana_costs)}  max={max(mana_costs)}"
    )
    _p(
        f"  Casts/fight: avg={sum(cast_counts) / len(cast_counts):.1f}  "
        f"min={min(cast_counts)}  max={max(cast_counts)}"
    )

    hp_loss_fights = [d for d in hp_deltas if d < -0.01]
    if hp_loss_fights:
        _p(
            f"  HP loss fights: {len(hp_loss_fights)}/{len(fights)}  "
            f"avg loss={sum(hp_loss_fights) / len(hp_loss_fights) * 100:.0f}%"
        )
    else:
        _p(f"  HP loss fights: 0/{len(fights)} (pet tanked everything)")

    backstep_fights = sum(1 for f in fights if f.get("backsteps", 0) > 0)
    retarget_fights = sum(1 for f in fights if f.get("retargets", 0) > 0)
    pet_heal_fights = sum(1 for f in fights if f.get("pet_heals", 0) > 0)
    if backstep_fights or retarget_fights or pet_heal_fights:
        _p(f"  Backsteps: {backstep_fights}  Retargets: {retarget_fights}  Pet heals: {pet_heal_fights}")

    # Per-npc breakdown (group by base name, strip trailing digits)
    mob_stats: dict[str, list] = defaultdict(list)
    for f in fights:
        base = f["npc"].rstrip("0123456789")
        mob_stats[base].append(f)

    if mob_stats:
        _p("\n  Per-npc type:")
        for name, fs in sorted(mob_stats.items(), key=lambda x: -len(x[1])):
            avg_dur = sum(f["duration"] for f in fs) / len(fs)
            avg_mana = sum(f["mana_spent"] for f in fs) / len(fs)
            avg_casts = sum(f["casts"] for f in fs) / len(fs)
            _p(
                f"    {name:30s}  x{len(fs):2d}  "
                f"avg={avg_dur:.1f}s  mana={avg_mana:.0f}  casts={avg_casts:.1f}"
            )


def _print_kill_cycles(cycles: list[dict]) -> None:
    """Print defeat cycle phase breakdown."""
    _p(f"\n-- Defeat Cycle Breakdown ({len(cycles)} cycles) --")
    phase_names = ["ACQUIRE", "PULL", "IN_COMBAT", "LOOT"]
    for phase in phase_names:
        times = [c.get(phase, 0) for c in cycles if phase in c]
        if times:
            _p(
                f"  {phase:15s} avg={sum(times) / len(times):5.1f}s  "
                f"min={min(times):5.1f}s  max={max(times):5.1f}s"
            )
    total_cycle = [sum(c.get(p, 0) for p in phase_names) for c in cycles]
    if total_cycle:
        _p(
            f"  {'TOTAL':15s} avg={sum(total_cycle) / len(total_cycle):5.1f}s  "
            f"min={min(total_cycle):5.1f}s  max={max(total_cycle):5.1f}s"
        )
    # Idle time between cycles
    idle_times = [c.get("_idle", 0) for c in cycles if "_idle" in c]
    if idle_times:
        _p(
            f"  {'IDLE (between)':15s} avg={sum(idle_times) / len(idle_times):5.1f}s  "
            f"min={min(idle_times):5.1f}s  max={max(idle_times):5.1f}s"
        )


def _print_kill_rate_trend(windows: list[dict]) -> None:
    """Print moving window defeat rates with trend direction."""
    _p("\n-- Defeat Rate Trend (5-min windows) --")
    for w in windows:
        bar = "#" * w["defeats"]
        _p(f"  [{w['start_m']:.0f}-{w['end_m']:.0f}m] {w['dph']:5.1f}/hr  {w['defeats']} defeats  {bar}")
    # Trend direction
    if len(windows) >= 2:
        first_kph = windows[0]["dph"]
        last_kph = windows[-1]["dph"]
        if last_kph > first_kph * 1.2:
            _p("  >>> Trend: IMPROVING (ramping up)")
        elif last_kph < first_kph * 0.8:
            _p("  >>> Trend: DECLINING (degrading)")
        else:
            _p("  >>> Trend: STABLE")


def _print_kill_timeline(fights: list[dict]) -> None:
    """Print chronological defeat timeline with gap detection."""
    _p("\n-- Defeat Timeline --")
    for i, f in enumerate(fights):
        elapsed = f["elapsed"]
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        gap_str = ""
        if i > 0:
            gap = f["elapsed"] - fights[i - 1]["elapsed"]
            flag = " <<<" if gap > 60 else ""
            gap_str = f"  (gap={gap:.0f}s{flag})"
        _p(
            f"  [{mins:2d}:{secs:02d}] Defeat {i + 1}: {f['npc']:<30s}  "
            f"{f['duration']:.0f}s  mana={f['mana_spent']:3d}  "
            f"HP={f['hp_end'] * 100:.0f}%{gap_str}"
        )


def _print_loot_analysis(events: list[dict]) -> None:
    """Print loot window and item statistics."""
    loots = [e for e in events if e["event"] == "loot_end"]
    if not loots:
        return
    _p(f"\n-- Loot Analysis ({len(loots)} corpses) --")
    windows_opened = sum(1 for l in loots if l.get("window_opened"))
    windows_failed = sum(1 for l in loots if l.get("window_opened") is False)
    _p(f"  Window opened: {windows_opened}/{len(loots)}  Failed to open: {windows_failed}")

    # Items seen across all loot windows
    all_items_seen = []
    for l in loots:
        all_items_seen.extend(l.get("items_seen", []))
    if all_items_seen:
        item_counts = Counter(all_items_seen)
        _p(f"  Unique item IDs seen: {len(item_counts)}  Total items: {len(all_items_seen)}")
        _p(f"  Item IDs: {dict(item_counts.most_common(10))}")

    # Items actually looted
    all_looted = []
    for l in loots:
        all_looted.extend(l.get("items_looted", []))
    if all_looted:
        loot_counts = Counter(all_looted)
        _p(f"  Items looted: {dict(loot_counts)}")
    else:
        _p("  Items looted: none (0 allow-listed items found)")


def _print_spatial(spatial: dict) -> None:
    """Print spatial analysis (camp distance, roaming area)."""
    _p("\n-- Spatial Analysis --")
    _p(
        f"  Camp dist: avg={spatial['avg_camp']:.0f}  "
        f"min={spatial['min_camp']:.0f}  max={spatial['max_camp']:.0f}"
    )
    _p(
        f"  Position range: X=[{spatial['min_x']:.0f},{spatial['max_x']:.0f}]  "
        f"Y=[{spatial['min_y']:.0f},{spatial['max_y']:.0f}]"
    )
    _p(f"  Roaming area: ~{spatial['area']:.0f} sq units")
    if spatial["max_camp"] > 300:
        _p(f"  >>> WARNING: drifted to {spatial['max_camp']:.0f}u from camp (hunt_max ~350)")


def _print_mana(mana_stats: dict) -> None:
    """Print mana utilization warnings and stats."""
    _p("\n-- Mana Utilization --")
    _p(f"  Time at 100% mana: {mana_stats['pct_full']:.0f}% of snapshots")
    _p(
        f"  Avg mana: {mana_stats['avg_mana']:.0f} / {mana_stats['avg_max']:.0f} "
        f"({mana_stats['avg_pct']:.0f}%)"
    )
    if mana_stats["total_casts"] == 0:
        _p("  >>> WARNING: 0 spells cast during combat (pet-only every fight)")
    elif mana_stats["pct_full"] > 80:
        _p(
            f"  >>> WARNING: mana full {mana_stats['pct_full']:.0f}% of time "
            f"-- casting more would speed defeats"
        )
    _p(
        f"  Total combat casts: {mana_stats['total_casts']}  "
        f"Total mana spent: {mana_stats['total_mana_spent']}"
    )


def _print_wander(wander_stats: dict) -> None:
    """Print wander effectiveness section."""
    _p("\n-- Wander Effectiveness --")
    _p(
        f"  Total wanders: {wander_stats['total']}  "
        f"Led to defeat: {wander_stats['led_to_kill']}  "
        f"({wander_stats['conversion_pct']:.0f}% conversion)"
    )
    _p(f"  Thrash cycles (walked=0, no progress): {wander_stats['thrash_count']}")
    _p(f"  Time in wander: {wander_stats['total_time']:.0f}s ({wander_stats['time_pct']:.0f}% of session)")


def _print_failures_and_tail(
    events: list[dict],
    fights: list[dict],
    snapshots: list[dict],
    routine_counts: Counter,
    failure_chains: list,
    cycles: list,
    droughts: list,
    spatial: dict,
    mana_stats: dict,
    wander_stats: dict,
    duration: float,
) -> list[dict]:
    """Print failure, spatial, mana, wander, loot, proximity, and recommendations."""
    failures = [e for e in events if e["event"] == "routine_end" and e["result"] == "FAILURE"]
    if failures:
        _p(f"\n-- Failures ({len(failures)}) --")
        fail_counts = Counter(f["routine"] for f in failures)
        for name, count in fail_counts.most_common():
            total = routine_counts.get(name, count)
            rate = count / max(total, 1) * 100
            _p(f"  {name}: {count}/{total} ({rate:.0f}% fail rate)")

    if failure_chains:
        _p("\n-- Failure Chains (consecutive fails) --")
        for chain in failure_chains[:5]:
            _p(
                f"  {chain['routine']} x{chain['count']} "
                f"at [{chain['start_m']:.1f}m-{chain['end_m']:.1f}m] "
                f"({chain['duration']:.0f}s stuck)"
            )

    if spatial:
        _print_spatial(spatial)
    if mana_stats:
        _print_mana(mana_stats)
    if wander_stats:
        _print_wander(wander_stats)

    _print_loot_analysis(events)

    close_snapshots = [s for s in snapshots if s.get("nearest_player", 9999) < 200]
    if close_snapshots:
        _p("\n-- Player Proximity --")
        _p(f"  Snapshots with player <200u: {len(close_snapshots)}/{len(snapshots)}")
        closest = min(s["nearest_player"] for s in close_snapshots)
        _p(f"  Closest player seen: {closest}u")

    recs = _recommendations(
        events,
        fights,
        failures,
        failure_chains,
        droughts,
        mana_stats,
        wander_stats,
        spatial,
        cycles,
        duration,
    )
    if recs:
        _p("\n-- Recommendations --")
        for i, rec in enumerate(recs, 1):
            _p(f"  {i}. [{rec['priority']}] {rec['issue']}")
            _p(f"     -> {rec['fix']}")

    return failures


def _print_report(
    events: list[dict],
    minutes: float,
    total_kills: int,
    dph: float,
    fights: list[dict],
    snapshots: list[dict],
    routine_times: dict,
    routine_counts: Counter,
    routine_results: dict,
    cycles: list,
    failure_chains: list,
    windows: list,
    droughts: list,
    spatial: dict,
    mana_stats: dict,
    wander_stats: dict,
    duration: float,
) -> list[dict]:
    """Print the analysis report and return failures list."""
    _p("=" * 72)
    _p(f"SESSION ANALYSIS -- {minutes:.1f} min | {total_kills} defeats | {dph:.1f} defeats/hr")
    _p("=" * 72)

    # -- 1. Routine Time Distribution --
    _p("\n-- Routine Time Distribution --")
    total_routine_time = sum(routine_times.values()) or 1
    for name, t in sorted(routine_times.items(), key=lambda x: -x[1]):
        pct = t / total_routine_time * 100
        results: dict[str, int] = routine_results.get(name, {})
        result_str = ", ".join(f"{k}={v}" for k, v in sorted(results.items()))
        count = routine_counts[name]
        _p(f"  {name:15s} {t:6.1f}s ({pct:4.1f}%)  x{count:3d}  [{result_str}]")

    if fights:
        _print_fight_analysis(fights)
    if cycles:
        _print_kill_cycles(cycles)
    if windows:
        _print_kill_rate_trend(windows)
    if droughts:
        _p("\n-- Defeat Droughts (gaps > 45s) --")
        for d in droughts:
            _p(f"  [{d['start_m']:.1f}-{d['end_m']:.1f}m] {d['gap']:.0f}s gap  |  {d['context']}")
    if fights:
        _print_kill_timeline(fights)

    failures = _print_failures_and_tail(
        events,
        fights,
        snapshots,
        routine_counts,
        failure_chains,
        cycles,
        droughts,
        spatial,
        mana_stats,
        wander_stats,
        duration,
    )

    _p("\n" + "=" * 72)
    return failures


def analyze(events: list[dict]) -> dict:
    """Run full analysis. Returns metrics dict for comparison."""
    if not events:
        _p("No events found.")
        return {}

    duration = events[-1].get("elapsed", 0)
    minutes = duration / 60

    # -- Collect typed events --
    fights = [e for e in events if e["event"] == "fight_end"]
    snapshots = [e for e in events if e["event"] == "snapshot"]

    total_kills = fights[-1]["defeats"] if fights else 0
    dph = total_kills / max(minutes / 60, 0.01) if minutes > 0 else 0

    # -- Sub-analyses --
    routine_times, routine_counts, routine_results = _routine_distribution(events)
    cycles = _kill_cycles(events)
    failure_chains = _failure_chains(events)
    windows = _moving_windows(fights, duration)
    droughts = _kill_droughts(events, fights, duration)
    spatial = _spatial_analysis(snapshots)
    mana_stats = _mana_utilization(snapshots, fights)
    wander_stats = _wander_effectiveness(events)

    # -- Output --
    failures = _print_report(
        events,
        minutes,
        total_kills,
        dph,
        fights,
        snapshots,
        routine_times,
        routine_counts,
        routine_results,
        cycles,
        failure_chains,
        windows,
        droughts,
        spatial,
        mana_stats,
        wander_stats,
        duration,
    )

    return {
        "duration_min": round(minutes, 1),
        "defeats": total_kills,
        "dph": round(dph, 1),
        "fights": len(fights),
        "avg_fight_s": round(sum(f["duration"] for f in fights) / max(len(fights), 1), 1),
        "total_casts": sum(f["casts"] for f in fights),
        "acquire_fails": sum(1 for e in failures if e["routine"] == "ACQUIRE"),
        "acquire_total": routine_counts.get("ACQUIRE", 0),
        "wander_total": routine_counts.get("WANDER", 0),
        "avg_camp_dist": spatial.get("avg_camp", 0) if spatial else 0,
        "mana_pct_full": mana_stats.get("pct_full", 0) if mana_stats else 0,
        "droughts": len(droughts),
        "failure_chains": len(failure_chains),
    }


# ---------------------------------------------------------------------------
# Sub-analyses
# ---------------------------------------------------------------------------


def _routine_distribution(events: list[dict]) -> tuple[dict[str, float], Counter, dict[str, Counter]]:
    """Compute time spent in each routine."""
    routine_times: dict[str, float] = defaultdict(float)
    routine_starts: dict[str, float] = {}
    routine_counts: Counter = Counter()
    routine_results: dict[str, Counter] = defaultdict(Counter)

    for e in events:
        ev = e["event"]
        if ev == "routine_start":
            name = e["routine"]
            routine_starts[name] = e["t"]
            routine_counts[name] += 1
        elif ev == "routine_end":
            name = e["routine"]
            if name in routine_starts:
                routine_times[name] += e["t"] - routine_starts[name]
                del routine_starts[name]
            routine_results[name][e["result"]] += 1

    return routine_times, routine_counts, routine_results


def _kill_cycles(events: list[dict]) -> list[dict[str, float]]:
    """Break session into defeat cycles: ACQUIRE -> PULL -> COMBAT -> LOOT.

    Measures time spent in each phase per cycle, plus idle time between cycles.
    """
    cycles = []
    current_cycle: dict[str, float] = {}
    phase_start: dict[str, float] = {}
    last_cycle_end = None

    for e in events:
        if e["event"] == "routine_start":
            name = e["routine"]
            if name in ("ACQUIRE", "PULL", "IN_COMBAT", "LOOT"):
                phase_start[name] = e["t"]
                if name == "ACQUIRE" and not current_cycle:
                    # Start of new cycle
                    if last_cycle_end is not None:
                        current_cycle["_idle"] = e["t"] - last_cycle_end
        elif e["event"] == "routine_end":
            name = e["routine"]
            if name in phase_start:
                elapsed = e["t"] - phase_start[name]
                current_cycle[name] = current_cycle.get(name, 0) + elapsed
                del phase_start[name]
        elif e["event"] == "fight_end":
            # Cycle complete
            if current_cycle:
                cycles.append(current_cycle)
            current_cycle = {}
            last_cycle_end = e["t"]

    return cycles


def _failure_chains(events: list[dict]) -> list[dict]:
    """Detect consecutive failure runs (e.g., ACQUIRE fail x5)."""
    chains = []
    current_routine = None
    current_count = 0
    chain_start = 0

    for e in events:
        if e["event"] == "routine_end":
            if e["result"] == "FAILURE":
                if e["routine"] == current_routine:
                    current_count += 1
                else:
                    if current_count >= 3:
                        chains.append(
                            {
                                "routine": current_routine,
                                "count": current_count,
                                "start_m": chain_start / 60,
                                "end_m": e["elapsed"] / 60,
                                "duration": e["elapsed"] - chain_start,
                            }
                        )
                    current_routine = e["routine"]
                    current_count = 1
                    chain_start = e["elapsed"]
            else:
                if current_count >= 3:
                    chains.append(
                        {
                            "routine": current_routine,
                            "count": current_count,
                            "start_m": chain_start / 60,
                            "end_m": e["elapsed"] / 60,
                            "duration": e["elapsed"] - chain_start,
                        }
                    )
                current_routine = None
                current_count = 0

    # Flush trailing chain
    if current_count >= 3 and events:
        chains.append(
            {
                "routine": current_routine,
                "count": current_count,
                "start_m": chain_start / 60,
                "end_m": events[-1]["elapsed"] / 60,
                "duration": events[-1]["elapsed"] - chain_start,
            }
        )

    return chains


def _moving_windows(fights: list[dict], duration: float, window_s: float = 300) -> list[dict]:
    """Compute defeat rate in sliding windows."""
    if not fights or duration < 60:
        return []

    windows = []
    step = min(window_s, max(duration / 4, 60))
    t: float = 0
    while t < duration:
        end = min(t + window_s, duration)
        defeats_in_window = sum(1 for f in fights if t <= f["elapsed"] < end)
        window_hours = (end - t) / 3600
        dph = defeats_in_window / max(window_hours, 0.01)
        windows.append(
            {
                "start_m": t / 60,
                "end_m": end / 60,
                "defeats": defeats_in_window,
                "dph": dph,
            }
        )
        t += step
        if t >= duration:
            break

    return windows


def _kill_droughts(
    events: list[dict], fights: list[dict], duration: float, threshold_s: float = 45
) -> list[dict]:
    """Detect gaps > threshold between defeats.

    Provides context: what routines were active during the drought.
    """
    droughts = []
    defeat_times = [0] + [f["elapsed"] for f in fights] + [duration]

    for i in range(len(defeat_times) - 1):
        gap = defeat_times[i + 1] - defeat_times[i]
        if gap > threshold_s:
            start = defeat_times[i]
            end = defeat_times[i + 1]

            # What was happening during this gap?
            routines_active: Counter[str] = Counter()
            for e in events:
                if e["event"] == "routine_start" and start <= e["elapsed"] <= end:
                    routines_active[e["routine"]] += 1

            context_parts = []
            for r, c in routines_active.most_common(3):
                context_parts.append(f"{r}x{c}")

            # Check for acquire fails during drought
            acq_fails = sum(
                1
                for e in events
                if e["event"] == "routine_end"
                and e["routine"] == "ACQUIRE"
                and e["result"] == "FAILURE"
                and start <= e["elapsed"] <= end
            )
            if acq_fails:
                context_parts.append(f"acq_fail={acq_fails}")

            context = ", ".join(context_parts) if context_parts else "unknown"
            droughts.append(
                {
                    "start_m": start / 60,
                    "end_m": end / 60,
                    "gap": gap,
                    "context": context,
                }
            )

    return droughts


def _spatial_analysis(snapshots: list[dict]) -> dict:
    """Analyze position drift and camp distance from snapshots."""
    if not snapshots:
        return {}

    camp_dists = [s["camp_dist"] for s in snapshots if "camp_dist" in s]
    xs = [s["x"] for s in snapshots if "x" in s]
    ys = [s["y"] for s in snapshots if "y" in s]

    if not camp_dists or not xs:
        return {}

    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)

    return {
        "avg_camp": sum(camp_dists) / len(camp_dists),
        "min_camp": min(camp_dists),
        "max_camp": max(camp_dists),
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
        "area": x_range * y_range,
    }


def _mana_utilization(snapshots: list[dict], fights: list[dict]) -> dict:
    """Analyze mana usage patterns."""
    if not snapshots:
        return {}

    mana_vals = []
    mana_maxes = []
    full_count = 0

    for s in snapshots:
        mana = s.get("mana", 0)
        mana_max = s.get("mana_max", 1)
        mana_vals.append(mana)
        mana_maxes.append(mana_max)
        if mana_max > 0 and mana >= mana_max * 0.98:
            full_count += 1

    avg_max = sum(mana_maxes) / len(mana_maxes) if mana_maxes else 1
    avg_mana = sum(mana_vals) / len(mana_vals) if mana_vals else 0
    total_casts = sum(f.get("casts", 0) for f in fights)
    total_mana_spent = sum(max(f.get("mana_spent", 0), 0) for f in fights)

    return {
        "pct_full": full_count / max(len(snapshots), 1) * 100,
        "avg_mana": avg_mana,
        "avg_max": avg_max,
        "avg_pct": avg_mana / max(avg_max, 1) * 100,
        "total_casts": total_casts,
        "total_mana_spent": total_mana_spent,
    }


def _find_matching_end(start_t: float, ends: list[dict]) -> dict | None:
    """Return the first end event whose timestamp is after start_t."""
    for we in ends:
        if we["t"] > start_t:
            return we
    return None


def _count_wander_conversions(wander_ends: list[dict], events: list[dict]) -> int:
    """Count wanders followed by a successful ACQUIRE within 15s."""
    led_to_kill = 0
    for we in wander_ends:
        for e in events:
            if e["t"] > we["t"] + 15:
                break
            if (
                e["event"] == "routine_end"
                and e["routine"] == "ACQUIRE"
                and e["result"] == "SUCCESS"
                and e["t"] > we["t"]
            ):
                led_to_kill += 1
                break
    return led_to_kill


def _wander_effectiveness(events: list[dict]) -> dict:
    """Analyze wander: how often does it lead to a defeat vs thrash?"""
    wander_starts = []
    wander_ends = []
    total_time = 0.0

    for e in events:
        if e["event"] == "routine_start" and e["routine"] == "WANDER":
            wander_starts.append(e)
        elif e["event"] == "routine_end" and e["routine"] == "WANDER":
            wander_ends.append(e)

    if not wander_starts:
        return {}

    duration = events[-1].get("elapsed", 0)
    for ws in wander_starts:
        matching_end = _find_matching_end(ws["t"], wander_ends)
        if matching_end:
            total_time += matching_end["t"] - ws["t"]

    led_to_kill = _count_wander_conversions(wander_ends, events)

    thrash_count = 0
    for ws in wander_starts:
        matching_end = _find_matching_end(ws["t"], wander_ends)
        if matching_end and (matching_end["t"] - ws["t"]) < 1.5:
            thrash_count += 1

    return {
        "total": len(wander_starts),
        "led_to_kill": led_to_kill,
        "conversion_pct": led_to_kill / max(len(wander_starts), 1) * 100,
        "thrash_count": thrash_count,
        "total_time": total_time,
        "time_pct": total_time / max(duration, 1) * 100,
    }


# ---------------------------------------------------------------------------
# Recommendations engine
# ---------------------------------------------------------------------------


def _recommendations(
    events: list[dict],
    fights: list[dict],
    failures: list[dict],
    failure_chains: list[dict],
    droughts: list[dict],
    mana_stats: dict,
    wander_stats: dict,
    spatial: dict,
    cycles: list[dict],
    duration: float,
) -> list[dict]:
    """Generate actionable recommendations from analysis."""
    recs = []

    # -- Zero casts --
    if mana_stats and mana_stats.get("total_casts", 0) == 0 and fights:
        recs.append(
            {
                "priority": "HIGH",
                "issue": "Zero spells cast in combat -- pet-only every fight",
                "fix": (
                    "Opportunistic casting not firing. At 100% mana, "
                    "DC or Lifespike should be cast to speed defeats. "
                    "Check combat.py opportunistic casting conditions."
                ),
            }
        )

    # -- Mana wasted --
    if mana_stats and mana_stats.get("pct_full", 0) > 80 and fights:
        recs.append(
            {
                "priority": "MEDIUM",
                "issue": (f"Mana full {mana_stats['pct_full']:.0f}% of session -- wasting regen"),
                "fix": (
                    "Lower mana thresholds for casting. Even LIGHT_BLUE "
                    "cons could get DC when mana is above 90%."
                ),
            }
        )

    # -- High ACQUIRE fail rate --
    acq_fails = sum(1 for f in failures if f["routine"] == "ACQUIRE")
    acq_total = sum(1 for e in events if e["event"] == "routine_start" and e["routine"] == "ACQUIRE")
    if acq_total > 0 and acq_fails / acq_total > 0.4:
        recs.append(
            {
                "priority": "HIGH",
                "issue": (
                    f"ACQUIRE fails {acq_fails}/{acq_total} "
                    f"({acq_fails / acq_total * 100:.0f}%) -- "
                    f"Tab not reaching npcs"
                ),
                "fix": (
                    "Tab effective range is ~60u, not 80u. Walk closer "
                    "BEFORE first tab, or lower approach threshold from 4 "
                    "empty tabs to 2."
                ),
            }
        )

    # -- Wander thrashing --
    if wander_stats and wander_stats.get("thrash_count", 0) > 3:
        recs.append(
            {
                "priority": "HIGH",
                "issue": (f"Wander thrash {wander_stats['thrash_count']}x (stopped immediately, 0 distance)"),
                "fix": (
                    "Wander check_fn threshold ({current}u) doesn't match "
                    "actual Tab range (~60u). Lower check_fn to 55u or "
                    "ignore npcs that Tab has already failed to reach."
                ),
            }
        )

    # -- Failure chains --
    for chain in failure_chains:
        if chain["duration"] > 15:
            recs.append(
                {
                    "priority": "MEDIUM",
                    "issue": (
                        f"{chain['routine']} failed {chain['count']}x "
                        f"in a row ({chain['duration']:.0f}s stuck)"
                    ),
                    "fix": (
                        "After 3+ consecutive fails, try a longer wander "
                        "to a new area instead of retrying the same spot."
                    ),
                }
            )
            break  # only report worst chain

    # -- Defeat droughts --
    for drought in droughts:
        if drought["gap"] > 60:
            recs.append(
                {
                    "priority": "MEDIUM",
                    "issue": (f"{drought['gap']:.0f}s defeat drought at {drought['start_m']:.1f}m"),
                    "fix": (
                        f"Context: {drought['context']}. "
                        f"If ACQUIRE-heavy, area may be npc-sparse -- "
                        f"wander should walk further. If WANDER-heavy, "
                        f"check_fn may be stopping walks prematurely."
                    ),
                }
            )
            break  # only report worst drought

    # -- Camp drift --
    if spatial and spatial.get("max_camp", 0) > 300:
        recs.append(
            {
                "priority": "LOW",
                "issue": (
                    f"Drifted to {spatial['max_camp']:.0f}u from camp (avg={spatial['avg_camp']:.0f}u)"
                ),
                "fix": (
                    "Tighten wander clamping or reduce hunt_max. "
                    "Far-from-camp defeats have longer return walks."
                ),
            }
        )

    # -- Slow fights --
    if fights:
        slow = [f for f in fights if f["duration"] > 40]
        if slow:
            names = ", ".join(f["npc"] for f in slow[:3])
            recs.append(
                {
                    "priority": "LOW",
                    "issue": (f"{len(slow)} fights over 40s ({names})"),
                    "fix": (
                        "Long fights waste time. Cast DC during pull for "
                        "all cons, not just BLUE. Consider Poison Bolt on "
                        "fights exceeding 25s."
                    ),
                }
            )

    # Sort by priority
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recs.sort(key=lambda r: priority_order.get(r["priority"], 9))

    return recs


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------


def compare(old_metrics: dict, new_metrics: dict, old_path: str, new_path: str) -> None:
    """Compare two sessions side by side."""
    _p("\n" + "=" * 72)
    _p("SESSION COMPARISON")
    _p("=" * 72)
    _p(f"  Old: {Path(old_path).name}")
    _p(f"  New: {Path(new_path).name}")
    _p("")

    fields = [
        ("Duration (min)", "duration_min", ""),
        ("Defeats", "defeats", ""),
        ("Defeats/hr", "dph", ""),
        ("Avg fight (s)", "avg_fight_s", "lower=better"),
        ("Total casts", "total_casts", ""),
        ("Acquire fails", "acquire_fails", "lower=better"),
        ("Acquire total", "acquire_total", ""),
        ("Wander count", "wander_total", "lower=better"),
        ("Avg camp dist", "avg_camp_dist", ""),
        ("Mana % full", "mana_pct_full", "lower=better"),
        ("Droughts", "droughts", "lower=better"),
        ("Fail chains", "failure_chains", "lower=better"),
    ]

    _p(f"  {'Metric':25s} {'Old':>10s} {'New':>10s} {'Delta':>10s}")
    _p(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10}")

    for label, key, note in fields:
        old_val = old_metrics.get(key, 0)
        new_val = new_metrics.get(key, 0)
        if isinstance(old_val, float):
            delta = new_val - old_val
            sign = "+" if delta >= 0 else ""
            _p(f"  {label:25s} {old_val:10.1f} {new_val:10.1f} {sign}{delta:9.1f}")
        else:
            delta = new_val - old_val
            sign = "+" if delta >= 0 else ""
            _p(f"  {label:25s} {old_val:10d} {new_val:10d} {sign}{delta:9d}")

    _p("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def find_latest_jsonl() -> str | None:
    sessions = Path("logs/sessions")
    if not sessions.exists():
        return None
    files = sorted(sessions.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return str(files[-1]) if files else None


def find_previous_jsonl(current: str) -> str | None:
    """Find the session JSONL before the given one."""
    sessions = Path("logs/sessions")
    files = sorted(sessions.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    current_path = Path(current).resolve()
    for i, f in enumerate(files):
        if f.resolve() == current_path and i > 0:
            return str(files[i - 1])
    return None


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--compare":
        # Compare mode: --compare old.jsonl new.jsonl
        old_path = sys.argv[2]
        new_path = sys.argv[3]
        old_events = load_events(old_path)
        new_events = load_events(new_path)
        _p(f"=== OLD SESSION: {Path(old_path).name} ===")
        old_metrics = analyze(old_events)
        _p(f"\n=== NEW SESSION: {Path(new_path).name} ===")
        new_metrics = analyze(new_events)
        compare(old_metrics, new_metrics, old_path, new_path)
    elif len(sys.argv) > 1:
        path = sys.argv[1]
        events = load_events(path)
        metrics = analyze(events)
        # Auto-compare with previous session if available
        prev = find_previous_jsonl(path)
        if prev:
            prev_events = load_events(prev)
            if len(prev_events) > 5:  # skip trivial sessions
                _p(f"\n(Auto-comparing with previous: {Path(prev).name})")
                prev_metrics = analyze(prev_events)
                compare(prev_metrics, metrics, prev, path)
    else:
        latest = find_latest_jsonl()
        if not latest:
            _p("No .jsonl files found in logs/sessions/")
            sys.exit(1)
        _p(f"Analyzing: {latest}\n")
        events = load_events(latest)
        analyze(events)


if __name__ == "__main__":
    main()
