"""Tests for runtime config-building functions and orchestrator utilities.

Covers runtime/agent.py (ThresholdConfig, find_config, load_zone_config,
_is_admin) and runtime/orchestrator.py (_prune_session_logs, log buffer).
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from runtime.agent import ThresholdConfig, _is_admin, find_config, load_zone_config
from runtime.orchestrator import AgentOrchestrator, _prune_session_logs

# ===========================================================================
# ThresholdConfig
# ===========================================================================


class TestThresholdConfig:
    def test_defaults(self):
        tc = ThresholdConfig()
        assert tc.rest_hp_high == pytest.approx(0.92)
        assert tc.rest_mana_high == pytest.approx(0.70)
        assert tc.rest_hp_low == pytest.approx(0.30)
        assert tc.rest_mana_low == pytest.approx(0.20)

    def test_from_toml_full(self):
        raw = {
            "thresholds": {
                "rest_hp_high": 0.80,
                "rest_mana_high": 0.60,
                "rest_hp_low": 0.25,
                "rest_mana_low": 0.15,
            }
        }
        tc = ThresholdConfig.from_toml(raw)
        assert tc.rest_hp_high == pytest.approx(0.80)
        assert tc.rest_mana_high == pytest.approx(0.60)
        assert tc.rest_hp_low == pytest.approx(0.25)
        assert tc.rest_mana_low == pytest.approx(0.15)

    def test_from_toml_partial(self):
        raw = {
            "thresholds": {
                "rest_hp_high": 0.85,
            }
        }
        tc = ThresholdConfig.from_toml(raw)
        assert tc.rest_hp_high == pytest.approx(0.85)
        # Other fields get defaults
        assert tc.rest_mana_high == pytest.approx(0.70)
        assert tc.rest_hp_low == pytest.approx(0.30)
        assert tc.rest_mana_low == pytest.approx(0.20)

    def test_from_toml_empty_dict(self):
        tc = ThresholdConfig.from_toml({})
        assert tc.rest_hp_high == pytest.approx(0.92)
        assert tc.rest_mana_high == pytest.approx(0.70)
        assert tc.rest_hp_low == pytest.approx(0.30)
        assert tc.rest_mana_low == pytest.approx(0.20)

    def test_from_toml_no_thresholds_key(self):
        raw = {"general": {"some_key": "value"}}
        tc = ThresholdConfig.from_toml(raw)
        assert tc.rest_hp_high == pytest.approx(0.92)

    def test_frozen(self):
        tc = ThresholdConfig()
        with pytest.raises(AttributeError):
            tc.rest_hp_high = 0.50

    def test_field_access(self):
        tc = ThresholdConfig(rest_hp_high=0.99, rest_mana_high=0.88, rest_hp_low=0.11, rest_mana_low=0.05)
        assert tc.rest_hp_high == pytest.approx(0.99)
        assert tc.rest_mana_high == pytest.approx(0.88)
        assert tc.rest_hp_low == pytest.approx(0.11)
        assert tc.rest_mana_low == pytest.approx(0.05)


# ===========================================================================
# find_config
# ===========================================================================


class TestFindConfig:
    def test_finds_first_candidate(self, tmp_path: Path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "settings.toml"
        config_file.write_text("[general]\n")

        # Patch Path.exists so the first candidate resolves to our tmp file
        real_exists = Path.exists

        def mock_exists(self_path):
            if self_path == config_file:
                return True
            if self_path.name == "settings.toml":
                return False
            return real_exists(self_path)

        # Patch the candidate list inside find_config by replacing __file__ resolution
        with patch("runtime.agent.Path.exists", mock_exists):
            # Directly test: create the file at one of the real candidate paths
            pass

        # Simpler: just verify it returns a Path when file exists at a known location
        real_candidates_source = Path(find_config.__code__.co_filename).resolve()
        src_config = real_candidates_source.parents[1] / "config" / "settings.toml"

        def mock_exists_for_candidates(self_path):
            if self_path == src_config:
                return True
            return False

        with patch.object(Path, "exists", mock_exists_for_candidates), patch("builtins.open"):
            result = find_config()
            assert result == src_config

    def test_raises_when_not_found(self):
        """When no candidate exists, find_config raises FileNotFoundError."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                find_config()


# ===========================================================================
# load_zone_config
# ===========================================================================


class TestLoadZoneConfig:
    def test_loads_existing_zone_toml(self, tmp_path: Path):
        zone_dir = tmp_path / "config" / "zones"
        zone_dir.mkdir(parents=True)
        zone_file = zone_dir / "nektulos.toml"
        zone_file.write_text('[zone]\nshort_name = "nektulos"\n\n[[camps]]\nname = "camp_a"\n')

        # Patch so load_zone_config finds our file
        with patch("runtime.agent.Path") as MockPath:
            # Make the first candidate resolve to our file
            instance = MockPath.return_value
            instance.resolve.return_value.parents.__getitem__ = lambda s, i: tmp_path
            # Simpler: patch __file__ parent resolution
            MockPath.__file__ = str(tmp_path / "runtime" / "agent.py")

        # Direct approach: just call with a known path setup
        # load_zone_config checks two paths; let's make one work
        # Patch Path.exists to return True for our tmp file
        real_exists = Path.exists
        real_open = open

        def mock_exists(self_path):
            if self_path.name == "nektulos.toml" and "zones" in str(self_path):
                return True
            return real_exists(self_path)

        with (
            patch.object(Path, "exists", mock_exists),
            patch("builtins.open", lambda p, *a, **kw: real_open(zone_file, *a, **kw)),
        ):
            result = load_zone_config("nektulos")
            assert result["zone"]["short_name"] == "nektulos"

    def test_nonexistent_zone_returns_empty(self):
        with patch.object(Path, "exists", return_value=False):
            result = load_zone_config("nonexistent_zone_xyz_12345")
            assert result == {}

    def test_returns_dict(self, tmp_path: Path):
        zone_dir = tmp_path / "config" / "zones"
        zone_dir.mkdir(parents=True)
        zone_file = zone_dir / "testzone.toml"
        zone_file.write_text('[zone]\nshort_name = "testzone"\n')

        real_exists = Path.exists
        real_open = open

        def mock_exists(self_path):
            if self_path.name == "testzone.toml" and "zones" in str(self_path):
                return True
            return real_exists(self_path)

        with (
            patch.object(Path, "exists", mock_exists),
            patch("builtins.open", lambda p, *a, **kw: real_open(zone_file, *a, **kw)),
        ):
            result = load_zone_config("testzone")
            assert isinstance(result, dict)


# ===========================================================================
# _is_admin
# ===========================================================================


class TestIsAdmin:
    def test_returns_false_on_non_windows(self):
        """On macOS/Linux, ctypes.windll does not exist, so _is_admin returns False."""
        result = _is_admin()
        assert result is False

    def test_returns_false_when_windll_missing(self):
        with patch("ctypes.windll", None, create=True):
            assert _is_admin() is False

    def test_handles_oserror(self):
        with patch("ctypes.windll", create=True) as mock_windll:
            mock_windll.shell32.IsUserAnAdmin.side_effect = OSError("no")
            assert _is_admin() is False

    def test_handles_attribute_error(self):
        with patch("ctypes.windll", create=True) as mock_windll:
            mock_windll.shell32.IsUserAnAdmin.side_effect = AttributeError("no")
            assert _is_admin() is False


# ===========================================================================
# _prune_session_logs
# ===========================================================================


class TestPruneSessionLogs:
    def _create_log_file(self, directory: Path, name: str, age_days: float) -> Path:
        """Create a log file with a specific modification time."""
        path = directory / name
        path.write_text("log data")
        mtime = time.time() - (age_days * 86400)
        import os

        os.utime(path, (mtime, mtime))
        return path

    def test_prunes_old_files(self, tmp_path: Path):
        # Create old files (10 days) beyond keep_min
        for i in range(8):
            self._create_log_file(tmp_path, f"session_{i:02d}_events.jsonl", age_days=10)
            self._create_log_file(tmp_path, f"session_{i:02d}_decisions.jsonl", age_days=10)

        _prune_session_logs(tmp_path, max_age_days=7, keep_min=3)

        events = list(tmp_path.glob("*_events.jsonl"))
        decisions = list(tmp_path.glob("*_decisions.jsonl"))
        # Should keep at most keep_min=3 of each
        assert len(events) <= 3
        assert len(decisions) <= 3

    def test_keeps_recent_files(self, tmp_path: Path):
        # Create recent files (1 day old)
        for i in range(4):
            self._create_log_file(tmp_path, f"session_{i:02d}_events.jsonl", age_days=1)

        _prune_session_logs(tmp_path, max_age_days=7, keep_min=2)

        events = list(tmp_path.glob("*_events.jsonl"))
        # All are recent, none should be pruned
        assert len(events) == 4

    def test_always_keeps_keep_min(self, tmp_path: Path):
        # Create 3 old files
        for i in range(3):
            self._create_log_file(tmp_path, f"session_{i:02d}_events.jsonl", age_days=30)

        _prune_session_logs(tmp_path, max_age_days=7, keep_min=5)

        events = list(tmp_path.glob("*_events.jsonl"))
        # keep_min=5 > 3 files, so none should be pruned
        assert len(events) == 3

    def test_does_not_prune_log_files(self, tmp_path: Path):
        """Only _events.jsonl and _decisions.jsonl are targeted, not .log files."""
        self._create_log_file(tmp_path, "session_old.log", age_days=30)
        self._create_log_file(tmp_path, "session_old_events.jsonl", age_days=30)

        # With keep_min=0 so the JSONL would be pruned
        _prune_session_logs(tmp_path, max_age_days=7, keep_min=0)

        assert (tmp_path / "session_old.log").exists()

    def test_empty_directory(self, tmp_path: Path):
        # Should not raise
        _prune_session_logs(tmp_path, max_age_days=7, keep_min=5)


# ===========================================================================
# AgentOrchestrator log buffer
# ===========================================================================


class TestOrchestratorLogBuffer:
    def _make_orchestrator(self) -> AgentOrchestrator:
        """Create a bare orchestrator without connecting to a process."""
        return AgentOrchestrator()

    def test_add_log_appends(self):
        orch = self._make_orchestrator()
        orch.add_log("first message")
        orch.add_log("second message", "WARNING")

        logs = orch.get_logs_since(-1)
        assert len(logs) == 2
        assert logs[0]["msg"] == "first message"
        assert logs[0]["level"] == "INFO"
        assert logs[1]["msg"] == "second message"
        assert logs[1]["level"] == "WARNING"

    def test_get_logs_since_filters(self):
        orch = self._make_orchestrator()
        orch.add_log("msg_0")
        orch.add_log("msg_1")
        orch.add_log("msg_2")

        # Get logs after index 0 (should return index 1 and 2)
        logs = orch.get_logs_since(0)
        assert len(logs) == 2
        assert logs[0]["msg"] == "msg_1"
        assert logs[1]["msg"] == "msg_2"

    def test_get_logs_since_returns_empty_when_caught_up(self):
        orch = self._make_orchestrator()
        orch.add_log("msg_0")

        logs = orch.get_logs_since(0)
        assert logs == []

    def test_log_index_increments(self):
        orch = self._make_orchestrator()
        orch.add_log("a")
        orch.add_log("b")
        orch.add_log("c")

        logs = orch.get_logs_since(-1)
        indices = [e["idx"] for e in logs]
        assert indices == [0, 1, 2]

    def test_log_has_timestamp(self):
        orch = self._make_orchestrator()
        before = time.time()
        orch.add_log("timed")
        after = time.time()

        logs = orch.get_logs_since(-1)
        assert len(logs) == 1
        assert before <= logs[0]["ts"] <= after

    def test_buffer_caps_at_500(self):
        orch = self._make_orchestrator()
        for i in range(600):
            orch.add_log(f"msg_{i}")

        logs = orch.get_logs_since(-1)
        # Buffer trims to 400 when it exceeds 500
        assert len(logs) <= 500
        assert len(logs) >= 400

    def test_buffer_retains_recent_after_cap(self):
        orch = self._make_orchestrator()
        for i in range(600):
            orch.add_log(f"msg_{i}")

        logs = orch.get_logs_since(-1)
        # Most recent message should be the last one added
        assert logs[-1]["msg"] == "msg_599"

    def test_default_level_is_info(self):
        orch = self._make_orchestrator()
        orch.add_log("test")
        logs = orch.get_logs_since(-1)
        assert logs[0]["level"] == "INFO"
