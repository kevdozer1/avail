"""Paths, the .env loader, the 2 config files and the seed budget guard."""

from __future__ import annotations

import pytest

from avail import config, paths, seeds


def test_parse_dotenv_is_crlf_safe():
    text = "# comment\r\nAVAIL_DATA=data\\somewhere\r\nEMPTY=\r\nexport QUOTED=\"a b\"\r\nBROKEN LINE\r\n"
    values = config.parse_dotenv(text)
    assert values["AVAIL_DATA"] == "data\\somewhere"      # no trailing \r
    assert values["EMPTY"] == ""
    assert values["QUOTED"] == "a b"
    assert "BROKEN LINE" not in values


def test_env_prefers_process_environment_and_treats_empty_as_unset(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_DOTENV", {"SOME_KEY": "from_file", "BLANK": ""})
    monkeypatch.delenv("SOME_KEY", raising=False)
    assert config.env("SOME_KEY") == "from_file"
    monkeypatch.setenv("SOME_KEY", "from_process")
    assert config.env("SOME_KEY") == "from_process"
    monkeypatch.setenv("SOME_KEY", "   ")
    assert config.env("SOME_KEY") == "from_file"
    assert config.env("BLANK") is None
    assert config.env("NOT_SET_ANYWHERE_12345", "fallback") == "fallback"


def test_missing_secret_names_the_variable_and_never_a_value(monkeypatch):
    monkeypatch.setattr(config, "_DOTENV", {})
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert not config.has_secret(["OPENROUTER_API_KEY"])
    with pytest.raises(config.MissingCredentialError) as err:
        config.load_secret(["OPENROUTER_API_KEY"], "OpenRouter")
    assert "OPENROUTER_API_KEY" in str(err.value)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-not-a-real-key")
    assert config.load_secret(["OPENROUTER_API_KEY"], "OpenRouter") == "sk-not-a-real-key"


def test_hardware_gate_is_closed_unless_set_to_1(monkeypatch):
    monkeypatch.setattr(config, "_DOTENV", {})
    monkeypatch.delenv("AVAIL_ALLOW_HARDWARE", raising=False)
    assert config.hardware_allowed() is False
    monkeypatch.setenv("AVAIL_ALLOW_HARDWARE", "true")
    assert config.hardware_allowed() is False
    monkeypatch.setenv("AVAIL_ALLOW_HARDWARE", "1")
    assert config.hardware_allowed() is True


def test_experiment_yaml_parses_and_holds_the_spec_values():
    cfg = config.load_experiment()
    assert cfg.bounds.alpha == 0.02 and cfg.bounds.beta == 0.05
    assert cfg.bounds.true_accept_for_e1 == 0.95
    assert set(cfg.connectors) == {"fric", "latch", "prong"}
    assert cfg.station.render.width == 640 and cfg.station.render.height == 480
    assert cfg.station.cameras == ["top_camera_d405", "left_camera_d405", "right_camera_d405"]
    assert cfg.zoo.classes == ["P0", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9"]
    assert cfg.spend_caps_usd["build_total"] == 15
    # probe caps come from config and nowhere else (CLAUDE.md rule 15)
    assert cfg.push_force_cap_n("fric") == pytest.approx(1.5 * 4)
    assert cfg.push_force_cap_n("latch") == pytest.approx(1.5 * 6)
    assert cfg.tug_force_cap_n("fric") == pytest.approx(0.6 * 8)
    assert cfg.tug_force_cap_n("latch") == pytest.approx(0.6 * 25)
    assert len(config.config_hash()) == 12


def test_seed_ranges_match_docs_07_and_are_disjoint():
    budget = config.load_experiment().seeds
    assert budget.ranges() == {
        "dev": (0, 9999),
        "zoo_fit": (10000, 13999),
        "zoo_calibration": (14000, 15999),
        "zoo_test": (16000, 17999),
        "shift_dev": (20000, 23999),
        "shift_test_e3": (30000, 37999),
        "shift_test_e4": (40000, 44999),
        "shift_test_e7": (50000, 59999),
        "zoo_fit_hard": (60000, 61999),
        "zoo_calibration_hard": (62000, 63999),
        "zoo_test_hard": (64000, 65999),
        "zoo_fit_hard2": (66000, 67999),
        "zoo_calibration_hard2": (68000, 69999),
        "zoo_test_hard2": (70000, 71999),
        "cable_fit": (24000, 25999),
        "cable_calibration": (26000, 27999),
        "fig_test": (72000, 72099),
    }
    assert budget.split_of(0) == "dev" and budget.split_of(9999) == "dev"
    assert budget.split_of(16000) == "zoo_test"
    assert budget.split_of(18000) is None
    assert len(budget.seeds("zoo_calibration")) == 2000


def test_overlapping_seed_ranges_are_rejected():
    raw = config._read_yaml(config.EXPERIMENT_YAML)
    raw["seeds"]["zoo_test"] = [15000, 17999]
    with pytest.raises(ValueError, match="overlap"):
        config.ExperimentConfig.model_validate(raw)


def test_models_yaml_parses():
    data = config.load_models_raw()
    assert data["currency"] == "USD"
    assert "models" in data


def test_seed_guard_refuses_test_seeds_and_records_the_final_read(tmp_path, monkeypatch):
    # E1bE2b is retired now (see the test below). Here it stands for an open final run
    monkeypatch.setattr(seeds, "RETIRED_FINAL_RUNS", {k: v for k, v in seeds.RETIRED_FINAL_RUNS.items() if k == "E1E2"})
    record = tmp_path / "reads.jsonl"
    seeds.guard(range(0, 50), purpose="dev work", record_path=record)                  # dev is free
    seeds.guard(range(14000, 14010), purpose="calibration", record_path=record)
    assert seeds.test_seed_reads(record) == []
    with pytest.raises(seeds.TestSeedError, match="read once"):
        seeds.guard([16000], purpose="peeking", record_path=record)
    with pytest.raises(seeds.TestSeedError):
        seeds.guard([16000], purpose="wrong experiment", final_run="E3", record_path=record)
    with pytest.raises(seeds.TestSeedError, match="outside every range"):
        seeds.guard([19000], purpose="typo", record_path=record)
    assert seeds.test_seed_reads(record) == []                                          # refusals leave no record
    with pytest.raises(seeds.TestSeedError, match="retired"):                          # S4 read 16,000 to 17,999 once
        seeds.guard(range(16000, 16005), purpose="final E1 and E2 again", final_run="E1E2", record_path=record)
    seeds.guard(range(64000, 64005), purpose="final E1b and E2b", final_run="E1bE2b", record_path=record)
    reads = seeds.test_seed_reads(record)
    assert len(reads) == 1
    assert reads[0]["split"] == "zoo_test_hard" and reads[0]["count"] == 5
    assert reads[0]["min"] == 64000 and reads[0]["max"] == 64004 and reads[0]["final_run"] == "E1bE2b"
    assert reads[0]["config_hash"] == config.config_hash()


def test_the_read_final_runs_are_retired(tmp_path):
    for run, start in (("E1E2", 16000), ("E1bE2b", 64000), ("E1cE2c", 70000)):
        with pytest.raises(seeds.TestSeedError, match="retired"):
            seeds.guard([start], purpose="again", final_run=run, record_path=tmp_path / "reads.jsonl")
        with pytest.raises(seeds.TestSeedError, match="retired"):     # a read of stored frames too
            seeds.guard([start], purpose="stored", final_run=run, record_path=tmp_path / "reads.jsonl", stored=True)
    assert not (tmp_path / "reads.jsonl").exists()


def test_a_test_split_read_under_one_config_is_not_read_under_another(tmp_path, monkeypatch):
    monkeypatch.setattr(seeds, "RETIRED_FINAL_RUNS", {})      # E1bE2b is retired now. This tests the guard as it was
    record = tmp_path / "reads.jsonl"
    record.write_text('{"split": "zoo_test_hard", "final_run": "E1bE2b", "count": 1, "min": 64000, "max": 64000, '
                      '"config_hash": "000000000000"}\n', encoding="utf-8")
    with pytest.raises(seeds.TestSeedError, match="already read under config"):
        seeds.guard([64001], purpose="a second read", final_run="E1bE2b", record_path=record)


def test_data_root_comes_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AVAIL_DATA", str(tmp_path / "data"))
    assert paths.data_root() == tmp_path / "data"
    made = paths.ensure_data_dirs()
    assert sorted(p.name for p in made) == ["cache", "runs", "videos", "zoo"]
    assert paths.run_dir("smoke_1").is_dir()
    with pytest.raises(ValueError):
        paths.run_dir("../escape")
    with pytest.raises(ValueError):
        paths.run_dir("")


def test_no_data_drive_path_is_written_in_the_package():
    """No literal data-drive path in committed code. The needles are built so this file holds none either."""
    needles = [drive + ":" + sep for drive in "Dd" for sep in ("\\", "/")]
    offenders = []
    for path in (paths.REPO_ROOT / "avail").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(needle in text for needle in needles):
            offenders.append(str(path.relative_to(paths.REPO_ROOT)))
    assert offenders == []


def test_disk_floor_raises_below_the_floor():
    free = paths.check_disk_floor(min_gb=0.001)
    assert free and all(gb > 0 for gb in free.values())
    with pytest.raises(paths.DiskFloorError, match="Stop and report"):
        paths.check_disk_floor(min_gb=10_000_000)


def test_a_test_seed_cannot_be_simulated_outside_its_final_run(monkeypatch):
    monkeypatch.delenv(seeds.FINAL_RUN_ENV, raising=False)
    seeds.assert_simulable(0)
    seeds.assert_simulable(15999)
    with pytest.raises(seeds.TestSeedError, match="read once"):
        seeds.assert_simulable(16000)
    with pytest.raises(seeds.TestSeedError, match="outside every range"):
        seeds.assert_simulable(19000)
    seeds.unlock_final_run("E3")                            # the wrong experiment opens nothing here
    with pytest.raises(seeds.TestSeedError):
        seeds.assert_simulable(16000)
    with pytest.raises(seeds.TestSeedError, match="retired"):
        seeds.unlock_final_run("E1E2")                      # S4 read 16,000 to 17,999 once
    monkeypatch.setenv(seeds.FINAL_RUN_ENV, "E1E2")         # and a worker that inherits it is refused too
    with pytest.raises(seeds.TestSeedError):
        seeds.assert_simulable(16000)
    with pytest.raises(seeds.TestSeedError, match="retired"):
        seeds.unlock_final_run("E1bE2b")                    # S4b read 64,000 to 65,999 once
    seeds.unlock_final_run("E3")
    seeds.assert_simulable(30000)
    with pytest.raises(seeds.TestSeedError):
        seeds.assert_simulable(16000)                       # E3 opens only the shift test seeds
    with pytest.raises(seeds.TestSeedError):
        seeds.assert_simulable(64000)
    seeds.unlock_final_run(None)
    with pytest.raises(seeds.TestSeedError):
        seeds.assert_simulable(64000)
    with pytest.raises(seeds.TestSeedError, match="unknown final run"):
        seeds.unlock_final_run("E9")
