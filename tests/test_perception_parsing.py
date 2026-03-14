"""Tests for perception/log_parser.py and perception/struct_validator.py."""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import patch

import pytest

from core.types import Disposition
from perception.log_parser import (
    DEATH_RE,
    FACTION_PATTERNS,
    FIZZLE_RE,
    FOOD_DRINK_RE,
    LOS_BLOCK_RE,
    MOB_ATTACKS_PET_RE,
    MUST_STAND_RE,
    SPELL_INTERRUPTED_RE,
    STUNNED_RE,
    TIMESTAMP_RE,
    XP_RE,
    ZONE_RE,
    LogParser,
)
from perception.struct_validator import (
    CheckResult,
    StructValidator,
    ValidationResult,
)

TS = "[Mon Mar 27 22:58:44 2026] "


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level regex pattern tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTimestampRE:
    def test_strips_standard_timestamp(self):
        m = TIMESTAMP_RE.match("[Mon Mar 27 22:58:44 2026] You died.")
        assert m is not None
        assert m.end() == len("[Mon Mar 27 22:58:44 2026] ")

    def test_no_match_without_brackets(self):
        assert TIMESTAMP_RE.match("You died.") is None

    def test_no_match_empty(self):
        assert TIMESTAMP_RE.match("") is None


class TestFactionPatterns:
    """Each of the 10 FACTION_PATTERNS maps a faction message to a Disposition."""

    @pytest.mark.parametrize(
        "phrase, expected_disposition",
        [
            ("a guard regards you as an ally -- what would you like", Disposition.ALLY),
            ("a guard looks upon you warmly -- what would you like", Disposition.WARMLY),
            ("a guard kindly considers you -- what would you like", Disposition.KINDLY),
            ("a guard judges you amiably -- what would you like", Disposition.AMIABLE),
            ("a guard regards you indifferently -- what would you like", Disposition.INDIFFERENT),
            ("a guard looks your way apprehensively -- what would you like", Disposition.APPREHENSIVE),
            ("a guard glowers at you dubiously -- what would you like", Disposition.DUBIOUS),
            ("a guard glares at you threateningly -- what would you like", Disposition.THREATENING),
            ("a guard scowls at you, ready to attack -- what would you like", Disposition.READY_TO_ATTACK),
            (
                "a guard scowls at you, what would you like your tombstone to say",
                Disposition.SCOWLING,
            ),
        ],
    )
    def test_faction_pattern_match(self, phrase, expected_disposition):
        matched = False
        for pattern, disposition in FACTION_PATTERNS:
            if m := pattern.match(phrase):
                assert disposition == expected_disposition
                assert m.group(1).strip()  # npc name captured
                matched = True
                break
        assert matched, f"No pattern matched: {phrase}"

    def test_faction_patterns_count(self):
        assert len(FACTION_PATTERNS) == 10

    def test_npc_name_capture(self):
        pattern, _ = FACTION_PATTERNS[0]  # ally pattern
        m = pattern.match("Guard Alnara regards you as an ally")
        assert m is not None
        assert m.group(1) == "Guard Alnara"


class TestZoneRE:
    def test_standard_zone(self):
        m = ZONE_RE.match("You have entered Nektulos Forest.")
        assert m is not None
        assert m.group(1) == "Nektulos Forest"

    def test_single_word_zone(self):
        m = ZONE_RE.match("You have entered Freeport.")
        assert m is not None
        assert m.group(1) == "Freeport"

    def test_no_trailing_period(self):
        assert ZONE_RE.match("You have entered Freeport") is None

    def test_unrelated_line(self):
        assert ZONE_RE.match("You gain experience!!") is None


class TestXPRE:
    def test_solo_xp(self):
        assert XP_RE.match("You gain experience!!") is not None

    def test_party_xp(self):
        assert XP_RE.match("You gain party experience!!") is not None

    def test_unrelated(self):
        assert XP_RE.match("You gain nothing.") is None


class TestDeathRE:
    def test_slain_by_npc(self):
        assert DEATH_RE.match("You have been slain by a_skeleton!") is not None

    def test_died(self):
        assert DEATH_RE.match("You died.") is not None

    def test_unrelated(self):
        assert DEATH_RE.match("You have been stunned.") is None


class TestStunnedRE:
    def test_stunned(self):
        assert STUNNED_RE.match("You have been stunned.") is not None

    def test_unrelated(self):
        assert STUNNED_RE.match("You have been slain by something!") is None


class TestFizzleRE:
    def test_fizzle(self):
        assert FIZZLE_RE.match("Your spell fizzles!") is not None

    def test_unrelated(self):
        assert FIZZLE_RE.match("Your spell is interrupted.") is None


class TestLOSBlockRE:
    def test_cannot_see(self):
        assert LOS_BLOCK_RE.match("You cannot see your target") is not None

    def test_cant_see(self):
        assert LOS_BLOCK_RE.match("You can't see your target") is not None

    def test_unrelated(self):
        assert LOS_BLOCK_RE.match("You are out of range") is None


class TestSpellInterruptedRE:
    def test_spell_is_interrupted(self):
        assert SPELL_INTERRUPTED_RE.match("Your spell is interrupted.") is not None

    def test_casting_is_interrupted(self):
        assert SPELL_INTERRUPTED_RE.match("Your casting is interrupted") is not None

    def test_unrelated(self):
        assert SPELL_INTERRUPTED_RE.match("Your spell fizzles!") is None


class TestMustStandRE:
    def test_must_be_standing(self):
        assert MUST_STAND_RE.match("You must be standing") is not None

    def test_full_message(self):
        assert MUST_STAND_RE.match("You must be standing to cast a spell.") is not None

    def test_unrelated(self):
        assert MUST_STAND_RE.match("You are already standing.") is None


class TestFoodDrinkRE:
    def test_low_on_food(self):
        m = FOOD_DRINK_RE.match("You are low on food.")
        assert m is not None
        assert m.group(1) == "low on"
        assert m.group(2) == "food"

    def test_out_of_drink(self):
        m = FOOD_DRINK_RE.match("You are out of drink.")
        assert m is not None
        assert m.group(1) == "out of"
        assert m.group(2) == "drink"

    def test_unrelated(self):
        assert FOOD_DRINK_RE.match("You are hungry.") is None


class TestMobAttacksPetRE:
    def test_hits_for_damage(self):
        m = MOB_ATTACKS_PET_RE.match("A spiderling hits Garann000 for 3 points of damage.")
        assert m is not None
        assert m.group(1) == "A spiderling"
        assert m.group(2) == "Garann000"

    def test_tries_to_bite(self):
        m = MOB_ATTACKS_PET_RE.match("A spiderling tries to bite Varn, but misses!")
        assert m is not None
        assert m.group(1) == "A spiderling"
        assert m.group(2) == "Varn"

    def test_crushes(self):
        m = MOB_ATTACKS_PET_RE.match("a gnoll crushes Bobbik for 12 points of damage.")
        assert m is not None
        assert m.group(1) == "a gnoll"
        assert m.group(2) == "Bobbik"

    def test_slashes(self):
        m = MOB_ATTACKS_PET_RE.match("a skeleton slashes Petname for 5 points of damage.")
        assert m is not None
        assert m.group(2) == "Petname"

    def test_unrelated(self):
        assert MOB_ATTACKS_PET_RE.match("You gain experience!!") is None


# ═══════════════════════════════════════════════════════════════════════════════
# LogParser tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestLogParserInit:
    def test_opens_and_seeks_to_end(self, tmp_path: Path):
        log_file = tmp_path / "eqlog.txt"
        log_file.write_text("preexisting line\n")
        parser = LogParser(str(log_file))
        # poll should return nothing because we seeked past the preexisting line
        events = parser.poll()
        assert events.zone_name is None
        assert events.xp_gained == 0
        assert events.dispositions == {}
        parser.close()

    def test_file_not_found(self, tmp_path: Path):
        parser = LogParser(str(tmp_path / "nonexistent.txt"))
        assert parser._file is None
        # poll on a broken parser returns empty events
        events = parser.poll()
        assert events.zone_name is None
        assert events.dispositions == {}


class TestLogParserPoll:
    def _make_parser(self, tmp_path: Path, pet_names: set[str] | None = None) -> tuple[Path, LogParser]:
        log_file = tmp_path / "eqlog.txt"
        log_file.write_text("")  # create empty
        parser = LogParser(str(log_file), pet_names=pet_names)
        return log_file, parser

    def _append(self, log_file: Path, lines: list[str]) -> None:
        with open(log_file, "a") as f:
            for line in lines:
                f.write(f"{TS}{line}\n")

    def test_lines_without_timestamp_skipped(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path)
        # Write lines without timestamp prefix
        with open(log_file, "a") as f:
            f.write("No timestamp here\n")
            f.write("Also no timestamp\n")
        events = parser.poll()
        assert events.zone_name is None
        assert events.xp_gained == 0
        parser.close()

    @patch("nav.zone_graph.normalize_zone_name", return_value="nektulos")
    def test_zone_detection(self, mock_norm, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path)
        self._append(log_file, ["You have entered Nektulos Forest."])
        events = parser.poll()
        assert events.zone_name == "Nektulos Forest"
        assert events.zone_short == "nektulos"
        mock_norm.assert_called_once_with("Nektulos Forest")
        parser.close()

    def test_xp_gain(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path)
        self._append(
            log_file,
            [
                "You gain experience!!",
                "You gain party experience!!",
                "You gain experience!!",
            ],
        )
        events = parser.poll()
        assert events.xp_gained == 3
        parser.close()

    def test_death_detection(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path)
        self._append(log_file, ["You have been slain by a_skeleton!"])
        events = parser.poll()
        assert events.player_died is True
        parser.close()

    def test_death_you_died(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path)
        self._append(log_file, ["You died."])
        events = parser.poll()
        assert events.player_died is True
        parser.close()

    def test_stunned(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path)
        self._append(log_file, ["You have been stunned."])
        events = parser.poll()
        assert events.player_stunned is True
        parser.close()

    def test_spell_fizzle(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path)
        self._append(log_file, ["Your spell fizzles!"])
        events = parser.poll()
        assert events.spell_fizzled is True
        parser.close()

    def test_los_blocked(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path)
        self._append(log_file, ["You cannot see your target"])
        events = parser.poll()
        assert events.los_blocked is True
        parser.close()

    def test_spell_interrupted(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path)
        self._append(log_file, ["Your spell is interrupted."])
        events = parser.poll()
        assert events.spell_interrupted is True
        parser.close()

    def test_casting_interrupted(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path)
        self._append(log_file, ["Your casting is interrupted"])
        events = parser.poll()
        assert events.spell_interrupted is True
        parser.close()

    def test_must_be_standing(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path)
        self._append(log_file, ["You must be standing to cast a spell."])
        events = parser.poll()
        assert events.must_be_standing is True
        parser.close()

    def test_food_warning(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path)
        self._append(log_file, ["You are low on food."])
        events = parser.poll()
        assert events.food_drink_warnings == ["low on food"]
        parser.close()

    def test_drink_warning(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path)
        self._append(log_file, ["You are out of drink."])
        events = parser.poll()
        assert events.food_drink_warnings == ["out of drink"]
        parser.close()

    def test_multiple_food_drink_warnings(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path)
        self._append(
            log_file,
            [
                "You are low on food.",
                "You are out of drink.",
            ],
        )
        events = parser.poll()
        assert len(events.food_drink_warnings) == 2
        assert "low on food" in events.food_drink_warnings
        assert "out of drink" in events.food_drink_warnings
        parser.close()

    def test_pet_attack_detection_with_known_pet(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path, pet_names={"Garann000"})
        self._append(log_file, ["A spiderling hits Garann000 for 3 points of damage."])
        events = parser.poll()
        assert "A spiderling" in events.pet_attackers
        parser.close()

    def test_pet_attack_ignored_when_victim_not_pet(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path, pet_names={"Garann000"})
        self._append(log_file, ["A spiderling hits SomePlayer for 3 points of damage."])
        events = parser.poll()
        assert len(events.pet_attackers) == 0
        parser.close()

    def test_pet_attack_ignored_when_no_pet_names(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path)
        self._append(log_file, ["A spiderling hits Garann000 for 3 points of damage."])
        events = parser.poll()
        assert len(events.pet_attackers) == 0
        parser.close()

    def test_faction_parsing(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path)
        self._append(
            log_file,
            [
                "Guard Alnara regards you as an ally -- blah",
                "a fire beetle glares at you threateningly -- blah",
            ],
        )
        events = parser.poll()
        assert events.dispositions["Guard Alnara"] == Disposition.ALLY
        assert events.dispositions["a fire beetle"] == Disposition.THREATENING
        parser.close()

    def test_multiple_events_single_poll(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path, pet_names={"MyPet"})
        self._append(
            log_file,
            [
                "You gain experience!!",
                "You have been stunned.",
                "Your spell fizzles!",
                "A gnoll hits MyPet for 5 points of damage.",
                "Guard Fippy regards you indifferently -- faction check",
            ],
        )
        events = parser.poll()
        assert events.xp_gained == 1
        assert events.player_stunned is True
        assert events.spell_fizzled is True
        assert "A gnoll" in events.pet_attackers
        assert events.dispositions["Guard Fippy"] == Disposition.INDIFFERENT
        parser.close()

    def test_sequential_polls(self, tmp_path: Path):
        """Each poll only returns events from new lines since last poll."""
        log_file, parser = self._make_parser(tmp_path)
        self._append(log_file, ["You gain experience!!"])
        events1 = parser.poll()
        assert events1.xp_gained == 1

        self._append(log_file, ["You gain experience!!", "You gain experience!!"])
        events2 = parser.poll()
        assert events2.xp_gained == 2
        parser.close()

    def test_empty_poll(self, tmp_path: Path):
        log_file, parser = self._make_parser(tmp_path)
        events = parser.poll()
        assert events.zone_name is None
        assert events.xp_gained == 0
        assert events.dispositions == {}
        assert events.player_died is False
        parser.close()


class TestLogParserClose:
    def test_close(self, tmp_path: Path):
        log_file = tmp_path / "eqlog.txt"
        log_file.write_text("")
        parser = LogParser(str(log_file))
        assert parser._file is not None
        parser.close()
        assert parser._file is None

    def test_close_idempotent(self, tmp_path: Path):
        log_file = tmp_path / "eqlog.txt"
        log_file.write_text("")
        parser = LogParser(str(log_file))
        parser.close()
        parser.close()  # should not raise
        assert parser._file is None


class TestPollDispositions:
    def test_returns_only_dispositions(self, tmp_path: Path):
        log_file = tmp_path / "eqlog.txt"
        log_file.write_text("")
        parser = LogParser(str(log_file))
        with open(log_file, "a") as f:
            f.write(f"{TS}Guard Alnara looks upon you warmly -- hi\n")
            f.write(f"{TS}You gain experience!!\n")
        result = parser.poll_dispositions()
        assert isinstance(result, dict)
        assert result["Guard Alnara"] == Disposition.WARMLY
        parser.close()


# ═══════════════════════════════════════════════════════════════════════════════
# struct_validator tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckResult:
    def test_passed(self):
        r = CheckResult(name="test", passed=True, detail="ok")
        assert r.name == "test"
        assert r.passed is True
        assert r.detail == "ok"

    def test_failed(self):
        r = CheckResult(name="fail_check", passed=False, detail="bad value")
        assert r.passed is False

    def test_default_detail(self):
        r = CheckResult(name="x", passed=True)
        assert r.detail == ""


class TestValidationResult:
    def test_all_passed(self):
        checks = [
            CheckResult("a", True, "ok"),
            CheckResult("b", True, "ok"),
        ]
        vr = ValidationResult(checks=checks)
        assert vr.compatible is True
        assert vr.passed_count == 2
        assert vr.failed_count == 0
        assert "2/2 checks passed" in vr.summary

    def test_one_failed(self):
        checks = [
            CheckResult("a", True, "ok"),
            CheckResult("b", False, "bad"),
        ]
        vr = ValidationResult(checks=checks)
        assert vr.compatible is False
        assert vr.passed_count == 1
        assert vr.failed_count == 1
        assert "1/2 checks FAILED" in vr.summary
        assert "b" in vr.summary

    def test_all_failed(self):
        checks = [
            CheckResult("x", False, "oops"),
            CheckResult("y", False, "nope"),
        ]
        vr = ValidationResult(checks=checks)
        assert vr.compatible is False
        assert vr.failed_count == 2
        assert "x" in vr.summary
        assert "y" in vr.summary

    def test_empty(self):
        vr = ValidationResult(checks=[])
        assert vr.compatible is True  # vacuously true
        assert vr.passed_count == 0
        assert vr.failed_count == 0


# ── Mock reader for StructValidator tests ─────────────────────────────────────


class MockValidatorReader:
    """Mock reader that returns pre-configured memory responses."""

    def __init__(self):
        self._responses: dict[tuple[str, int], object] = {}
        self._profile_base_cache: int | None = None

    def _configure(self, method: str, address: int, value: object) -> None:
        self._responses[(method, address)] = value

    def _read_pointer(self, address: int) -> int:
        key = ("pointer", address)
        if key in self._responses:
            return self._responses[key]  # type: ignore[return-value]
        return 0

    def _read_bytes(self, address: int, size: int) -> bytes:
        key = ("bytes", address)
        if key in self._responses:
            return self._responses[key][:size]  # type: ignore[index]
        raise OSError(f"No mock data for bytes at 0x{address:08X}")

    def _read_int32(self, address: int) -> int:
        key = ("int32", address)
        if key in self._responses:
            return self._responses[key]  # type: ignore[return-value]
        return struct.unpack("<i", self._read_bytes(address, 4))[0]

    def _read_uint32(self, address: int) -> int:
        return self._read_int32(address)

    def _read_string(self, address: int, max_len: int = 64) -> str:
        key = ("string", address)
        if key in self._responses:
            return self._responses[key]  # type: ignore[return-value]
        raise OSError(f"No mock data for string at 0x{address:08X}")

    def _resolve_profile_base(self) -> int | None:
        return self._profile_base_cache


def _build_spawn_buf(
    level: int = 30,
    race: int = 1,
    cls: int = 5,
    spawn_type: int = 0,
    hp_max: int = 1000,
) -> bytes:
    """Build a minimal entity struct buffer with canary fields set."""
    from perception import offsets

    buf = bytearray(0x02F0)
    buf[offsets.LEVEL] = level
    struct.pack_into("<I", buf, offsets.RACE, race)
    buf[offsets.CLASS] = cls
    buf[offsets.TYPE] = spawn_type
    struct.pack_into("<i", buf, offsets.HP_MAX, hp_max)
    return bytes(buf)


# ── StructValidator individual check tests ────────────────────────────────────


@pytest.fixture()
def _distinct_root_ptrs(monkeypatch):
    """Patch REDACTED root pointers to distinct addresses so tests can configure them independently."""
    from perception import offsets as _offsets

    monkeypatch.setattr(_offsets, "PLAYER_SPAWN_PTR", 0x10)
    monkeypatch.setattr(_offsets, "ZONE_PTR", 0x14)
    monkeypatch.setattr(_offsets, "GAME_ENGINE_PTR", 0x18)
    monkeypatch.setattr(_offsets, "ENGINE_STATE_PTR", 0x1C)


@pytest.mark.usefixtures("_distinct_root_ptrs")
class TestCheckStaticPointers:
    def test_valid_pointers(self):
        from perception import offsets

        reader = MockValidatorReader()
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, 0x00400000)
        reader._configure("pointer", offsets.ZONE_PTR, 0x00500000)
        reader._configure("pointer", offsets.GAME_ENGINE_PTR, 0x00600000)
        sv = StructValidator(reader)
        result = sv._check_static_pointers()
        assert result.passed is True

    def test_null_player_pointer(self):
        from perception import offsets

        reader = MockValidatorReader()
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, 0)
        reader._configure("pointer", offsets.ZONE_PTR, 0x00500000)
        reader._configure("pointer", offsets.GAME_ENGINE_PTR, 0x00600000)
        sv = StructValidator(reader)
        result = sv._check_static_pointers()
        assert result.passed is False
        assert "null" in result.detail

    def test_null_engine_pointer(self):
        from perception import offsets

        reader = MockValidatorReader()
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, 0x00400000)
        reader._configure("pointer", offsets.ZONE_PTR, 0x00500000)
        reader._configure("pointer", offsets.GAME_ENGINE_PTR, 0)
        sv = StructValidator(reader)
        result = sv._check_static_pointers()
        assert result.passed is False
        assert "null" in result.detail

    def test_out_of_range_pointer(self):
        from perception import offsets

        reader = MockValidatorReader()
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, 0x00000001)  # < 0x10000
        reader._configure("pointer", offsets.ZONE_PTR, 0x00500000)
        reader._configure("pointer", offsets.GAME_ENGINE_PTR, 0x00600000)
        sv = StructValidator(reader)
        result = sv._check_static_pointers()
        assert result.passed is False
        assert "outside valid range" in result.detail

    def test_high_out_of_range(self):
        from perception import offsets

        reader = MockValidatorReader()
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, 0x80000000)  # > 0x7FFFFFFF
        reader._configure("pointer", offsets.ZONE_PTR, 0x00500000)
        reader._configure("pointer", offsets.GAME_ENGINE_PTR, 0x00600000)
        sv = StructValidator(reader)
        result = sv._check_static_pointers()
        assert result.passed is False
        assert "outside valid range" in result.detail


class TestCheckSpawninfoLayout:
    def test_valid_layout(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("bytes", base, _build_spawn_buf())
        sv = StructValidator(reader)
        result = sv._check_spawninfo_layout()
        assert result.passed is True

    def test_null_player_spawn(self):
        from perception import offsets

        reader = MockValidatorReader()
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, 0)
        sv = StructValidator(reader)
        result = sv._check_spawninfo_layout()
        assert result.passed is False
        assert "null" in result.detail

    def test_bad_level(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("bytes", base, _build_spawn_buf(level=0))
        sv = StructValidator(reader)
        result = sv._check_spawninfo_layout()
        assert result.passed is False
        assert "level=" in result.detail

    def test_bad_race(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("bytes", base, _build_spawn_buf(race=999))
        sv = StructValidator(reader)
        result = sv._check_spawninfo_layout()
        assert result.passed is False
        assert "race=" in result.detail

    def test_bad_class(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("bytes", base, _build_spawn_buf(cls=99))
        sv = StructValidator(reader)
        result = sv._check_spawninfo_layout()
        assert result.passed is False
        assert "class=" in result.detail

    def test_bad_type(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("bytes", base, _build_spawn_buf(spawn_type=3))
        sv = StructValidator(reader)
        result = sv._check_spawninfo_layout()
        assert result.passed is False
        assert "type=" in result.detail

    def test_negative_hp_max(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("bytes", base, _build_spawn_buf(hp_max=-1))
        sv = StructValidator(reader)
        result = sv._check_spawninfo_layout()
        assert result.passed is False
        assert "hp_max=" in result.detail


class TestCheckSpawninfoName:
    def test_valid_name(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("string", base + offsets.NAME, "Windozer")
        sv = StructValidator(reader)
        result = sv._check_spawninfo_name()
        assert result.passed is True
        assert "Windozer" in result.detail

    def test_too_short(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("string", base + offsets.NAME, "Ab")
        sv = StructValidator(reader)
        result = sv._check_spawninfo_name()
        assert result.passed is False
        assert "too short" in result.detail

    def test_non_ascii(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("string", base + offsets.NAME, "W\xe9ndozer")
        sv = StructValidator(reader)
        result = sv._check_spawninfo_name()
        assert result.passed is False
        assert "not ASCII" in result.detail

    def test_non_alpha(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("string", base + offsets.NAME, "Wind123")
        sv = StructValidator(reader)
        result = sv._check_spawninfo_name()
        assert result.passed is False
        assert "not alphabetic" in result.detail

    def test_empty_name(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("string", base + offsets.NAME, "")
        sv = StructValidator(reader)
        result = sv._check_spawninfo_name()
        assert result.passed is False

    def test_null_player(self):
        from perception import offsets

        reader = MockValidatorReader()
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, 0)
        sv = StructValidator(reader)
        result = sv._check_spawninfo_name()
        assert result.passed is False
        assert "null" in result.detail


class TestCheckProfileChain:
    def _setup_valid_chain(self, reader: MockValidatorReader) -> int:
        """Wire up a valid 3-hop pointer chain and return profile_base."""
        from perception import offsets

        base = 0x00400000
        ci = 0x00500000
        intermediate = 0x00600000
        profile_base = 0x00700000

        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("pointer", base + offsets.CHARINFO_PTR, ci)
        reader._configure("pointer", ci + offsets.CHARINFO_PROFILE_INDIR, intermediate)
        reader._configure("pointer", intermediate + offsets.PROFILE_PTR_OFFSET, profile_base)

        # Guardrail values
        reader._configure("int32", profile_base + offsets.PROFILE_LEVEL, 30)
        reader._configure("int32", profile_base + offsets.PROFILE_CLASS, 5)
        reader._configure("int32", profile_base + offsets.PROFILE_RACE, 1)
        reader._configure("int32", profile_base + offsets.PROFILE_MANA, 500)

        return profile_base

    def test_valid_chain(self):
        reader = MockValidatorReader()
        self._setup_valid_chain(reader)
        sv = StructValidator(reader)
        result = sv._check_profile_chain()
        assert result.passed is True

    def test_null_player(self):
        from perception import offsets

        reader = MockValidatorReader()
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, 0)
        sv = StructValidator(reader)
        result = sv._check_profile_chain()
        assert result.passed is False
        assert "null" in result.detail

    def test_null_charinfo(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("pointer", base + offsets.CHARINFO_PTR, 0)
        sv = StructValidator(reader)
        result = sv._check_profile_chain()
        assert result.passed is False
        assert "CHARINFO_PTR" in result.detail

    def test_null_intermediate(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        ci = 0x00500000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("pointer", base + offsets.CHARINFO_PTR, ci)
        reader._configure("pointer", ci + offsets.CHARINFO_PROFILE_INDIR, 0)
        sv = StructValidator(reader)
        result = sv._check_profile_chain()
        assert result.passed is False
        assert "null" in result.detail

    def test_null_profile_base(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        ci = 0x00500000
        intermediate = 0x00600000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("pointer", base + offsets.CHARINFO_PTR, ci)
        reader._configure("pointer", ci + offsets.CHARINFO_PROFILE_INDIR, intermediate)
        reader._configure("pointer", intermediate + offsets.PROFILE_PTR_OFFSET, 0)
        sv = StructValidator(reader)
        result = sv._check_profile_chain()
        assert result.passed is False
        assert "null" in result.detail

    def test_guardrail_bad_level(self):
        from perception import offsets

        reader = MockValidatorReader()
        profile_base = self._setup_valid_chain(reader)
        reader._configure("int32", profile_base + offsets.PROFILE_LEVEL, 999)
        sv = StructValidator(reader)
        result = sv._check_profile_chain()
        assert result.passed is False
        assert "guardrails failed" in result.detail
        assert "level=" in result.detail

    def test_guardrail_bad_mana(self):
        from perception import offsets

        reader = MockValidatorReader()
        profile_base = self._setup_valid_chain(reader)
        reader._configure("int32", profile_base + offsets.PROFILE_MANA, 99999)
        sv = StructValidator(reader)
        result = sv._check_profile_chain()
        assert result.passed is False
        assert "mana=" in result.detail


class TestCheckProfileIdentity:
    def test_matching_values(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        profile_base = 0x00700000

        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("bytes", base, _build_spawn_buf(level=30, race=1, cls=5))
        reader._profile_base_cache = profile_base
        reader._configure("int32", profile_base + offsets.PROFILE_LEVEL, 30)
        reader._configure("int32", profile_base + offsets.PROFILE_RACE, 1)
        reader._configure("int32", profile_base + offsets.PROFILE_CLASS, 5)

        sv = StructValidator(reader)
        result = sv._check_profile_identity()
        assert result.passed is True
        assert "matches" in result.detail

    def test_mismatched_level(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        profile_base = 0x00700000

        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("bytes", base, _build_spawn_buf(level=30, race=1, cls=5))
        reader._profile_base_cache = profile_base
        reader._configure("int32", profile_base + offsets.PROFILE_LEVEL, 25)  # mismatch
        reader._configure("int32", profile_base + offsets.PROFILE_RACE, 1)
        reader._configure("int32", profile_base + offsets.PROFILE_CLASS, 5)

        sv = StructValidator(reader)
        result = sv._check_profile_identity()
        assert result.passed is False
        assert "level" in result.detail
        assert "mismatch" in result.detail.lower()

    def test_mismatched_race_and_class(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        profile_base = 0x00700000

        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("bytes", base, _build_spawn_buf(level=30, race=1, cls=5))
        reader._profile_base_cache = profile_base
        reader._configure("int32", profile_base + offsets.PROFILE_LEVEL, 30)
        reader._configure("int32", profile_base + offsets.PROFILE_RACE, 99)
        reader._configure("int32", profile_base + offsets.PROFILE_CLASS, 12)

        sv = StructValidator(reader)
        result = sv._check_profile_identity()
        assert result.passed is False
        assert "race" in result.detail
        assert "class" in result.detail

    def test_null_profile_chain(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("bytes", base, _build_spawn_buf())
        reader._profile_base_cache = None  # chain not resolved

        sv = StructValidator(reader)
        result = sv._check_profile_identity()
        assert result.passed is False
        assert "not resolved" in result.detail


class TestCheckActorclientReachable:
    def test_valid(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        ac_ptr = 0x00800000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("pointer", base + offsets.ACTORCLIENT_PTR, ac_ptr)
        reader._configure("int32", ac_ptr + offsets.AC_ACTIVITY_STATE, 32)
        reader._configure("int32", ac_ptr + offsets.AC_COMBAT_FLAG, 0)

        sv = StructValidator(reader)
        result = sv._check_actorclient_reachable()
        assert result.passed is True

    def test_null_player(self):
        from perception import offsets

        reader = MockValidatorReader()
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, 0)
        sv = StructValidator(reader)
        result = sv._check_actorclient_reachable()
        assert result.passed is False

    def test_null_ac_pointer(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("pointer", base + offsets.ACTORCLIENT_PTR, 0)
        sv = StructValidator(reader)
        result = sv._check_actorclient_reachable()
        assert result.passed is False
        assert "null" in result.detail

    def test_out_of_range_ac_pointer(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("pointer", base + offsets.ACTORCLIENT_PTR, 0x00000005)  # < 0x10000
        sv = StructValidator(reader)
        result = sv._check_actorclient_reachable()
        assert result.passed is False
        assert "outside valid range" in result.detail

    def test_bad_activity_state(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        ac_ptr = 0x00800000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("pointer", base + offsets.ACTORCLIENT_PTR, ac_ptr)
        reader._configure("int32", ac_ptr + offsets.AC_ACTIVITY_STATE, 999)
        reader._configure("int32", ac_ptr + offsets.AC_COMBAT_FLAG, 0)

        sv = StructValidator(reader)
        result = sv._check_actorclient_reachable()
        assert result.passed is False
        assert "activity_state=" in result.detail

    def test_bad_combat_flag(self):
        from perception import offsets

        reader = MockValidatorReader()
        base = 0x00400000
        ac_ptr = 0x00800000
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("pointer", base + offsets.ACTORCLIENT_PTR, ac_ptr)
        reader._configure("int32", ac_ptr + offsets.AC_ACTIVITY_STATE, 32)
        reader._configure("int32", ac_ptr + offsets.AC_COMBAT_FLAG, 5)  # not 0 or 1

        sv = StructValidator(reader)
        result = sv._check_actorclient_reachable()
        assert result.passed is False
        assert "combat_flag=" in result.detail


class TestCheckContentsVtable:
    def test_matching_vtable(self):
        from perception import offsets

        reader = MockValidatorReader()
        profile_base = 0x00700000
        equip_ptr = 0x00900000
        reader._profile_base_cache = profile_base
        reader._configure("pointer", profile_base + offsets.PROFILE_EQUIP_START, equip_ptr)
        reader._configure("pointer", equip_ptr, offsets.CONTENTS_VTABLE)

        sv = StructValidator(reader)
        result = sv._check_contents_vtable()
        assert result.passed is True
        assert "matches expected" in result.detail

    def test_mismatched_vtable(self):
        from perception import offsets

        reader = MockValidatorReader()
        profile_base = 0x00700000
        equip_ptr = 0x00900000
        reader._profile_base_cache = profile_base
        reader._configure("pointer", profile_base + offsets.PROFILE_EQUIP_START, equip_ptr)
        reader._configure("pointer", equip_ptr, 0xDEADBEEF)

        sv = StructValidator(reader)
        result = sv._check_contents_vtable()
        assert result.passed is False
        assert "mismatch" in result.detail

    def test_empty_slot_skipped(self):
        from perception import offsets

        reader = MockValidatorReader()
        profile_base = 0x00700000
        reader._profile_base_cache = profile_base
        reader._configure("pointer", profile_base + offsets.PROFILE_EQUIP_START, 0)

        sv = StructValidator(reader)
        result = sv._check_contents_vtable()
        assert result.passed is True
        assert "skipped" in result.detail

    def test_null_profile_chain(self):
        reader = MockValidatorReader()
        reader._profile_base_cache = None
        sv = StructValidator(reader)
        result = sv._check_contents_vtable()
        assert result.passed is False
        assert "not resolved" in result.detail


@pytest.mark.usefixtures("_distinct_root_ptrs")
class TestCheckEngineState:
    def test_in_game(self):
        from perception import offsets

        reader = MockValidatorReader()
        engine_ptr = 0x00A00000
        reader._configure("pointer", offsets.ENGINE_STATE_PTR, engine_ptr)
        reader._configure("int32", engine_ptr + offsets.ENGINE_GAME_MODE_OFFSET, 5)

        sv = StructValidator(reader)
        result = sv._check_engine_state()
        assert result.passed is True
        assert "in-game" in result.detail

    def test_char_select(self):
        from perception import offsets

        reader = MockValidatorReader()
        engine_ptr = 0x00A00000
        reader._configure("pointer", offsets.ENGINE_STATE_PTR, engine_ptr)
        reader._configure("int32", engine_ptr + offsets.ENGINE_GAME_MODE_OFFSET, 0)

        sv = StructValidator(reader)
        result = sv._check_engine_state()
        assert result.passed is False
        assert "char select" in result.detail

    def test_unexpected_mode(self):
        from perception import offsets

        reader = MockValidatorReader()
        engine_ptr = 0x00A00000
        reader._configure("pointer", offsets.ENGINE_STATE_PTR, engine_ptr)
        reader._configure("int32", engine_ptr + offsets.ENGINE_GAME_MODE_OFFSET, 99)

        sv = StructValidator(reader)
        result = sv._check_engine_state()
        assert result.passed is False
        assert "unexpected" in result.detail

    def test_null_engine_pointer(self):
        from perception import offsets

        reader = MockValidatorReader()
        reader._configure("pointer", offsets.ENGINE_STATE_PTR, 0)
        sv = StructValidator(reader)
        result = sv._check_engine_state()
        assert result.passed is False
        assert "null" in result.detail


# ── Full validate() integration ───────────────────────────────────────────────


@pytest.mark.usefixtures("_distinct_root_ptrs")
class TestFullValidation:
    def _build_all_pass_reader(self) -> MockValidatorReader:
        """Build a reader where every check will pass."""
        from perception import offsets

        reader = MockValidatorReader()

        base = 0x00400000
        ci = 0x00500000
        intermediate = 0x00600000
        profile_base = 0x00700000
        ac_ptr = 0x00800000
        equip_ptr = 0x00900000
        engine_ptr = 0x00A00000

        # Static pointers
        reader._configure("pointer", offsets.PLAYER_SPAWN_PTR, base)
        reader._configure("pointer", offsets.ZONE_PTR, 0x00500000)
        reader._configure("pointer", offsets.GAME_ENGINE_PTR, engine_ptr)

        # entity struct layout
        reader._configure("bytes", base, _build_spawn_buf(level=30, race=1, cls=5))

        # entity struct name
        reader._configure("string", base + offsets.NAME, "Windozer")

        # Profile chain
        reader._configure("pointer", base + offsets.CHARINFO_PTR, ci)
        reader._configure("pointer", ci + offsets.CHARINFO_PROFILE_INDIR, intermediate)
        reader._configure("pointer", intermediate + offsets.PROFILE_PTR_OFFSET, profile_base)
        reader._configure("int32", profile_base + offsets.PROFILE_LEVEL, 30)
        reader._configure("int32", profile_base + offsets.PROFILE_CLASS, 5)
        reader._configure("int32", profile_base + offsets.PROFILE_RACE, 1)
        reader._configure("int32", profile_base + offsets.PROFILE_MANA, 500)

        # Profile identity (cross-validate needs _resolve_profile_base)
        reader._profile_base_cache = profile_base

        # ActorClient
        reader._configure("pointer", base + offsets.ACTORCLIENT_PTR, ac_ptr)
        reader._configure("int32", ac_ptr + offsets.AC_ACTIVITY_STATE, 32)
        reader._configure("int32", ac_ptr + offsets.AC_COMBAT_FLAG, 0)

        # Contents vtable
        reader._configure("pointer", profile_base + offsets.PROFILE_EQUIP_START, equip_ptr)
        reader._configure("pointer", equip_ptr, offsets.CONTENTS_VTABLE)

        # Engine state
        reader._configure("pointer", offsets.ENGINE_STATE_PTR, engine_ptr)
        reader._configure("int32", engine_ptr + offsets.ENGINE_GAME_MODE_OFFSET, 5)

        return reader

    def test_all_pass(self):
        reader = self._build_all_pass_reader()
        sv = StructValidator(reader)
        result = sv.validate()
        assert result.compatible is True
        assert result.passed_count == 8
        assert result.failed_count == 0
        assert "compatible" in result.summary

    def test_partial_failure(self):
        from perception import offsets

        reader = self._build_all_pass_reader()
        # Break engine state
        engine_ptr = 0x00A00000
        reader._configure("int32", engine_ptr + offsets.ENGINE_GAME_MODE_OFFSET, 0)

        sv = StructValidator(reader)
        result = sv.validate()
        assert result.compatible is False
        assert result.failed_count >= 1
        assert "engine_state" in result.summary
