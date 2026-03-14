"""Fight outcome learning: per-npc-type stats that improve over time.

Records mana spent, duration, HP lost, pet deaths per npc base name.
After enough fights, replaces hardcoded con-based heuristics with
learned values for target scoring and mana conservation.

Persists to data/memory/<zone>_fights.json across sessions.
"""

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

from core.types import normalize_entity_name as normalize_mob_name

log = logging.getLogger(__name__)

# Minimum fights before learned data overrides heuristics
MIN_FIGHTS_FOR_LEARNED = 5
# Maximum samples kept per npc type (sliding window)
MAX_SAMPLES = 30


@dataclass(slots=True)
class FightRecord:
    """Single fight outcome -- persistent training data for utility functions."""

    duration: float  # seconds
    mana_spent: int  # mana used
    hp_delta: float  # HP change (-0.05 = lost 5%)
    casts: int  # number of spell casts
    pet_heals: int  # pet heal count
    pet_died: bool  # pet died during fight
    defeated: bool  # npc actually died (vs flee/abort)
    adds: int = 0  # number of social extra_npcs that joined during fight
    mob_level: int = 0  # npc level at fight time
    player_level: int = 0  # player level at fight time
    con: str = ""  # con color (green/light_blue/blue/white/yellow/red)
    strategy: str = ""  # combat strategy used (pet_tank/pet_and_dot/etc)
    mana_start: int = 0  # mana before fight
    mana_end: int = 0  # mana after fight
    pet_hp_start: float = 0.0  # pet HP% before fight
    pet_hp_end: float = 0.0  # pet HP% after fight
    xp_gained: bool = False  # whether this defeat gave XP
    cycle_time: float = 0.0  # total time from acquire start to fight end
    fitness: float = 0.0  # encounter fitness score (0.0-1.0, training signal)
    extra_npc_types: tuple[str, ...] = ()  # base names of entities that added


@dataclass(slots=True)
class MobStats:
    """Aggregated stats for a npc base name."""

    fights: int = 0
    defeats: int = 0
    avg_duration: float = 0.0
    avg_mana: float = 0.0
    avg_hp_lost: float = 0.0  # average HP% lost per fight (0.0-1.0)
    avg_casts: float = 0.0
    pet_death_rate: float = 0.0  # fraction of fights where pet died
    danger_score: float = 0.0  # 0 = trivial, 1 = very dangerous
    avg_extra_npcs: float = 0.0  # average social extra_npcs per fight

    # Posterior parameters for Thompson Sampling (Bayesian online learning).
    # Normal conjugate posteriors for continuous values, Beta for rates.
    # Priors are weakly informative; posteriors tighten as data accumulates.
    dur_post_mean: float = 25.0  # Normal posterior: fight duration (seconds)
    dur_post_var: float = 100.0  # Normal posterior: duration variance
    mana_post_mean: float = 15.0  # Normal posterior: mana cost
    mana_post_var: float = 400.0  # Normal posterior: mana variance
    danger_post_mean: float = 0.3  # Normal posterior: danger score [0, 1]
    danger_post_var: float = 0.1  # Normal posterior: danger variance
    add_alpha: float = 1.0  # Beta posterior: add probability (hits + prior)
    add_beta: float = 1.0  # Beta posterior: add probability (misses + prior)
    pet_death_alpha: float = 1.0  # Beta posterior: pet death rate (deaths + prior)
    pet_death_beta: float = 3.0  # Beta posterior: pet death rate (survivals + prior)


def _sample_variance(values: list[float], mean: float) -> float:
    """Unbiased sample variance, floored to prevent zero-variance collapse."""
    n = len(values)
    if n < 2:
        return 100.0  # wide prior-like variance for single observations
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return max(var, 1.0)


def _normal_posterior(
    sample_mean: float,
    sample_var: float,
    n: int,
    prior_mean: float,
    prior_var: float,
) -> tuple[float, float]:
    """Normal-Normal conjugate posterior. Returns (posterior_mean, posterior_var)."""
    prior_prec = 1.0 / prior_var
    data_prec = n / max(sample_var, 1e-6)
    post_prec = prior_prec + data_prec
    post_var = 1.0 / post_prec
    post_mean = (prior_mean * prior_prec + sample_mean * data_prec) / post_prec
    return post_mean, post_var


class FightHistory:
    """Per-npc-type fight outcome tracker with persistence."""

    def __init__(self, zone: str = "", data_dir: str = "") -> None:
        self._zone = zone
        self._data_dir = data_dir or str(Path("data") / "memory")
        # mob_base_name -> list of FightRecord
        self._records: dict[str, list[FightRecord]] = {}
        # Cached aggregates (recomputed on record)
        self._stats: dict[str, MobStats] = {}
        self._dirty = False
        # Append-only log of (fitness, mob_name) for gradient tuner consumption
        self._recent_fitness: list[tuple[float, str]] = []
        # Regret tracking: per-encounter instantaneous and cumulative regret
        self._regret_log: list[float] = []  # instantaneous regret per encounter
        self._cumulative_regret: float = 0.0
        self._load()

    def _path(self) -> str:
        return str(Path(self._data_dir) / f"{self._zone}_fights.json")

    def _load(self) -> None:
        """Load persisted fight data from disk."""
        path = self._path()
        if not Path(path).exists():
            return
        try:
            with open(path) as f:
                raw = json.load(f)
            # v1 format: {"v": 1, "npcs": {name: [records]}}
            # legacy format: {name: [records]} (no version key)
            if isinstance(raw, dict) and "npcs" in raw:
                mob_data = raw["npcs"]
            else:
                log.info("[LIFECYCLE] FightHistory: no schema version in %s (pre-v1)", self._zone)
                mob_data = raw
            count = 0
            for mob_name, records in mob_data.items():
                self._records[mob_name] = []
                for r in records[-MAX_SAMPLES:]:
                    self._records[mob_name].append(
                        FightRecord(
                            duration=r.get("dur", 0),
                            mana_spent=r.get("mana", 0),
                            hp_delta=r.get("hp", 0),
                            casts=r.get("casts", 0),
                            pet_heals=r.get("pet_h", 0),
                            pet_died=r.get("pet_d", False),
                            defeated=r.get("defeat", True),
                            adds=r.get("adds", 0),
                            mob_level=r.get("mlv", 0),
                            player_level=r.get("plv", 0),
                            con=r.get("con", ""),
                            strategy=r.get("strat", ""),
                            mana_start=r.get("m0", 0),
                            mana_end=r.get("m1", 0),
                            pet_hp_start=r.get("ph0", 0.0),
                            pet_hp_end=r.get("ph1", 0.0),
                            xp_gained=r.get("xp", False),
                            cycle_time=r.get("cyc", 0.0),
                            fitness=r.get("fit", 0.0),
                            extra_npc_types=tuple(r.get("at", ())),
                        )
                    )
                    count += 1
                self._recompute(mob_name)
            if count > 0:
                log.info(
                    "[LIFECYCLE] FightHistory: loaded %d records for %d npc types from %s",
                    count,
                    len(self._records),
                    self._zone,
                )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            log.warning("[LIFECYCLE] FightHistory: load failed: %s", e)

    def save(self) -> None:
        """Persist fight data to disk."""
        if not self._dirty:
            return
        path = self._path()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        npcs = {}
        for mob_name, records in self._records.items():
            npcs[mob_name] = [
                {
                    "dur": round(r.duration, 1),
                    "mana": r.mana_spent,
                    "hp": round(r.hp_delta, 3),
                    "casts": r.casts,
                    "pet_h": r.pet_heals,
                    "pet_d": r.pet_died,
                    "defeat": r.defeated,
                    "adds": r.adds,
                    "mlv": r.mob_level,
                    "plv": r.player_level,
                    "con": r.con,
                    "strat": r.strategy,
                    "m0": r.mana_start,
                    "m1": r.mana_end,
                    "ph0": round(r.pet_hp_start, 2),
                    "ph1": round(r.pet_hp_end, 2),
                    "xp": r.xp_gained,
                    "cyc": round(r.cycle_time, 1),
                    "fit": round(r.fitness, 3),
                    "at": list(r.extra_npc_types) if r.extra_npc_types else [],
                }
                for r in records[-MAX_SAMPLES:]
            ]
        data = {"v": 1, "npcs": npcs}
        try:
            with open(path, "w") as f:
                json.dump(data, f, separators=(",", ":"))
            self._dirty = False
            log.info("[LIFECYCLE] FightHistory: saved %d npc types to %s", len(npcs), self._zone)
        except (OSError, TypeError) as e:
            log.warning("[LIFECYCLE] FightHistory: save failed: %s", e)

    def record(
        self,
        mob_name: str,
        duration: float,
        mana_spent: int,
        hp_delta: float,
        casts: int,
        pet_heals: int,
        pet_died: bool,
        defeated: bool,
        adds: int = 0,
        mob_level: int = 0,
        player_level: int = 0,
        con: str = "",
        strategy: str = "",
        mana_start: int = 0,
        mana_end: int = 0,
        pet_hp_start: float = 0.0,
        pet_hp_end: float = 0.0,
        xp_gained: bool = False,
        cycle_time: float = 0.0,
        fitness: float = 0.0,
        extra_npc_types: tuple[str, ...] = (),
    ) -> None:
        """Record a fight outcome with full training data."""
        key = normalize_mob_name(mob_name)
        if key not in self._records:
            self._records[key] = []
        self._records[key].append(
            FightRecord(
                duration=duration,
                mana_spent=mana_spent,
                hp_delta=hp_delta,
                casts=casts,
                pet_heals=pet_heals,
                pet_died=pet_died,
                defeated=defeated,
                adds=adds,
                mob_level=mob_level,
                player_level=player_level,
                con=con,
                strategy=strategy,
                mana_start=mana_start,
                mana_end=mana_end,
                pet_hp_start=pet_hp_start,
                pet_hp_end=pet_hp_end,
                xp_gained=xp_gained,
                cycle_time=cycle_time,
                fitness=fitness,
                extra_npc_types=extra_npc_types,
            )
        )
        # Trim to sliding window
        if len(self._records[key]) > MAX_SAMPLES:
            self._records[key] = self._records[key][-MAX_SAMPLES:]
        self._recompute(key)
        self._dirty = True
        if fitness > 0.0:
            self._recent_fitness.append((fitness, key))

    def _recompute(self, key: str) -> None:
        """Recompute aggregated stats and Bayesian posteriors for a npc type."""
        records = self._records.get(key, [])
        if not records:
            self._stats.pop(key, None)
            return
        n = len(records)
        defeats = sum(1 for r in records if r.defeated)
        avg_dur = sum(r.duration for r in records) / n
        avg_mana = sum(r.mana_spent for r in records) / n
        avg_hp = sum(abs(r.hp_delta) for r in records) / n
        avg_casts = sum(r.casts for r in records) / n
        pet_deaths = sum(1 for r in records if r.pet_died)
        pet_death_rate = pet_deaths / n
        avg_extra_npcs = sum(r.adds for r in records) / n

        # Danger score: weighted combination of HP loss + pet death rate
        danger = min(1.0, avg_hp * 2.0 + pet_death_rate * 0.5)

        # -- Bayesian posteriors for Thompson Sampling --
        # Duration: Normal conjugate (prior: N(25, 100))
        dur_var = _sample_variance([r.duration for r in records], avg_dur)
        dur_post_mean, dur_post_var = _normal_posterior(
            avg_dur,
            dur_var,
            n,
            prior_mean=25.0,
            prior_var=100.0,
        )
        # Mana: Normal conjugate (prior: N(15, 400))
        mana_var = _sample_variance(
            [float(r.mana_spent) for r in records],
            avg_mana,
        )
        mana_post_mean, mana_post_var = _normal_posterior(
            avg_mana,
            mana_var,
            n,
            prior_mean=15.0,
            prior_var=400.0,
        )
        # Danger: Normal conjugate (prior: N(0.3, 0.1))
        per_fight_danger = [min(1.0, abs(r.hp_delta) * 2.0 + (0.5 if r.pet_died else 0.0)) for r in records]
        dg_mean = sum(per_fight_danger) / n
        dg_var = _sample_variance(per_fight_danger, dg_mean)
        danger_post_mean, danger_post_var = _normal_posterior(
            dg_mean,
            max(dg_var, 0.01),
            n,
            prior_mean=0.3,
            prior_var=0.1,
        )
        # Add probability: Beta conjugate (prior: Beta(1, 1) = uniform)
        with_adds = sum(1 for r in records if r.adds > 0)
        # Pet death rate: Beta conjugate (prior: Beta(1, 3) -- biased toward survival)

        self._stats[key] = MobStats(
            fights=n,
            defeats=defeats,
            avg_duration=avg_dur,
            avg_mana=avg_mana,
            avg_hp_lost=avg_hp,
            avg_casts=avg_casts,
            pet_death_rate=pet_death_rate,
            danger_score=danger,
            avg_extra_npcs=avg_extra_npcs,
            dur_post_mean=dur_post_mean,
            dur_post_var=dur_post_var,
            mana_post_mean=mana_post_mean,
            mana_post_var=mana_post_var,
            danger_post_mean=danger_post_mean,
            danger_post_var=danger_post_var,
            add_alpha=1.0 + with_adds,
            add_beta=1.0 + n - with_adds,
            pet_death_alpha=1.0 + pet_deaths,
            pet_death_beta=3.0 + n - pet_deaths,
        )

    def learned_add_probability(self, mob_name: str) -> float | None:
        """Probability of getting at least one add when engaging this entity type.

        Returns None if insufficient data (< MIN_FIGHTS_FOR_LEARNED).
        """
        key = normalize_mob_name(mob_name)
        records = self._records.get(key, [])
        if len(records) < MIN_FIGHTS_FOR_LEARNED:
            return None
        with_adds = sum(1 for r in records if r.adds > 0)
        return with_adds / len(records)

    def learned_add_types(self, mob_name: str) -> dict[str, float]:
        """Frequency of each add entity type when engaging mob_name.

        Returns {add_base_name: fraction_of_encounters}. Empty if no data.
        """
        key = normalize_mob_name(mob_name)
        records = self._records.get(key, [])
        if len(records) < MIN_FIGHTS_FOR_LEARNED:
            return {}
        type_counts: dict[str, int] = {}
        for r in records:
            for at in r.extra_npc_types:
                type_counts[at] = type_counts.get(at, 0) + 1
        n = len(records)
        return {k: v / n for k, v in type_counts.items()}

    # -- Thompson Sampling: draw from posterior distributions --------

    def sample_duration(self, mob_name: str) -> float:
        """Thompson sample from duration posterior. Wide prior if unknown."""
        key = normalize_mob_name(mob_name)
        stats = self._stats.get(key)
        if stats is None:
            return max(1.0, random.gauss(25.0, 10.0))
        sigma = math.sqrt(max(stats.dur_post_var, 0.01))
        return max(1.0, random.gauss(stats.dur_post_mean, sigma))

    def sample_mana(self, mob_name: str) -> int:
        """Thompson sample from mana cost posterior. Wide prior if unknown."""
        key = normalize_mob_name(mob_name)
        stats = self._stats.get(key)
        if stats is None:
            return max(0, int(random.gauss(15.0, 20.0)))
        sigma = math.sqrt(max(stats.mana_post_var, 0.01))
        return max(0, int(random.gauss(stats.mana_post_mean, sigma)))

    def sample_danger(self, mob_name: str) -> float:
        """Thompson sample from danger score posterior, clamped to [0, 1]."""
        key = normalize_mob_name(mob_name)
        stats = self._stats.get(key)
        if stats is None:
            return max(0.0, min(1.0, random.gauss(0.3, 0.316)))
        sigma = math.sqrt(max(stats.danger_post_var, 0.001))
        return max(0.0, min(1.0, random.gauss(stats.danger_post_mean, sigma)))

    def sample_add_probability(self, mob_name: str) -> float:
        """Thompson sample from add probability posterior (Beta distribution)."""
        key = normalize_mob_name(mob_name)
        stats = self._stats.get(key)
        if stats is None:
            return random.betavariate(1.0, 1.0)
        return random.betavariate(stats.add_alpha, max(stats.add_beta, 0.01))

    def sample_pet_death_rate(self, mob_name: str) -> float:
        """Thompson sample from pet death rate posterior (Beta distribution)."""
        key = normalize_mob_name(mob_name)
        stats = self._stats.get(key)
        if stats is None:
            return random.betavariate(1.0, 3.0)
        return random.betavariate(stats.pet_death_alpha, max(stats.pet_death_beta, 0.01))

    def drain_recent_fitness(self) -> list[tuple[float, str]]:
        """Return and clear recent (fitness, mob_name) pairs for gradient tuner."""
        result = self._recent_fitness
        self._recent_fitness = []
        return result

    def get_all_stats(self) -> dict[str, MobStats]:
        """Return a snapshot of all per-npc aggregated stats."""
        return dict(self._stats)

    def get_stats(self, mob_name: str) -> MobStats | None:
        """Get aggregated stats for a npc type. None if no data."""
        key = normalize_mob_name(mob_name)
        return self._stats.get(key)

    def has_learned(self, mob_name: str) -> bool:
        """True if enough fights to use learned data over heuristics."""
        stats = self.get_stats(mob_name)
        return stats is not None and stats.fights >= MIN_FIGHTS_FOR_LEARNED

    def learned_duration(self, mob_name: str) -> float | None:
        """Learned average fight duration. None if insufficient data."""
        stats = self.get_stats(mob_name)
        if stats and stats.fights >= MIN_FIGHTS_FOR_LEARNED:
            return stats.avg_duration
        return None

    def learned_mana(self, mob_name: str) -> int | None:
        """Learned average mana cost. None if insufficient data."""
        stats = self.get_stats(mob_name)
        if stats and stats.fights >= MIN_FIGHTS_FOR_LEARNED:
            return int(stats.avg_mana)
        return None

    def learned_danger(self, mob_name: str) -> float | None:
        """Learned danger score (0-1). None if insufficient data."""
        stats = self.get_stats(mob_name)
        if stats and stats.fights >= MIN_FIGHTS_FOR_LEARNED:
            return stats.danger_score
        return None

    def learned_adds(self, mob_name: str) -> float | None:
        """Learned average social extra_npcs per fight. None if insufficient data."""
        stats = self.get_stats(mob_name)
        if stats and stats.fights >= MIN_FIGHTS_FOR_LEARNED:
            return stats.avg_extra_npcs
        return None

    def learned_pet_death_rate(self, mob_name: str) -> float | None:
        """Learned pet death rate (0.0-1.0). None if insufficient data."""
        stats = self.get_stats(mob_name)
        if stats and stats.fights >= MIN_FIGHTS_FOR_LEARNED:
            return stats.pet_death_rate
        return None

    # -- Regret tracking -------------------------------------------

    def record_regret(self, chosen_fitness: float, best_fitness: float) -> None:
        """Record per-encounter regret: best_available - chosen.

        Regret measures the cost of the agent's target selection relative
        to the best target that was available at decision time. Sublinear
        cumulative regret growth proves the learning is converging.
        """
        instantaneous = max(0.0, best_fitness - chosen_fitness)
        self._regret_log.append(instantaneous)
        self._cumulative_regret += instantaneous
        # Cap log length to prevent unbounded growth
        if len(self._regret_log) > 1000:
            self._regret_log = self._regret_log[-500:]

    @property
    def cumulative_regret(self) -> float:
        return self._cumulative_regret

    @property
    def regret_count(self) -> int:
        return len(self._regret_log)

    def avg_recent_regret(self, window: int = 50) -> float:
        """Average regret over the last N encounters."""
        if not self._regret_log:
            return 0.0
        recent = self._regret_log[-window:]
        return sum(recent) / len(recent)

    def regret_summary(self) -> str:
        """Human-readable regret statistics."""
        n = len(self._regret_log)
        if n == 0:
            return "Regret: no data"
        avg = self._cumulative_regret / n
        recent = self.avg_recent_regret(50)
        early = sum(self._regret_log[: min(50, n)]) / min(50, n) if n > 0 else 0.0
        return (
            f"Regret: cumulative={self._cumulative_regret:.2f} over {n} encounters "
            f"(avg={avg:.3f}, early_50={early:.3f}, recent_50={recent:.3f})"
        )

    def summary(self) -> str:
        """Human-readable summary of learned npc data."""
        if not self._stats:
            return "FightHistory: no data"
        lines = [f"FightHistory: {len(self._stats)} npc types"]
        for key in sorted(self._stats.keys()):
            s = self._stats[key]
            learned = "*" if s.fights >= MIN_FIGHTS_FOR_LEARNED else " "
            adds_info = f", adds={s.avg_extra_npcs:.1f}" if s.avg_extra_npcs > 0 else ""
            lines.append(
                f"  {learned}{key}: {s.fights} fights, "
                f"dur={s.avg_duration:.0f}s, mana={s.avg_mana:.0f}, "
                f"hp_lost={s.avg_hp_lost * 100:.0f}%, "
                f"danger={s.danger_score:.2f}{adds_info}"
                f"{' PET_DEATHS!' if s.pet_death_rate > 0.2 else ''}"
            )
        return "\n".join(lines)
