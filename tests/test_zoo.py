"""The zoo: where it lives, how a run resumes, and the seed guard on its doors."""

from __future__ import annotations

import pandas as pd
import pytest

from avail.paths import aspire_station_xml
from avail.seeds import TestSeedError
from avail.zoo import store

needs_station = pytest.mark.skipif(not aspire_station_xml().is_file(), reason="third_party/ASPIRE is not cloned")


def test_blocks_are_aligned_so_a_resumed_run_meets_the_same_parts():
    blocks = store.blocks(list(range(1030, 1170)))
    assert [b[0] for b in blocks] == [1030, 1050, 1100, 1150]
    assert store.part_name(blocks[0]) == "part_001000_001049"
    assert store.part_name([1049]) == "part_001000_001049"
    assert store.part_name([1050]) == "part_001050_001099"
    assert store.blocks([7, 3, 5]) == [[3, 5, 7]]


def test_split_seeds_match_the_seed_budget():
    assert store.split_seeds("fit") == range(10000, 14000)
    assert store.split_seeds("calibration") == range(14000, 16000)
    assert store.split_seeds("test") == range(16000, 18000)
    with pytest.raises(KeyError):
        store.split_seeds("holdout")


def test_parts_merge_sorted_and_duplicates_are_refused(tmp_path):
    store.write_table_atomic([{"seed": 1051, "x": 2.0}, {"seed": 1050, "x": 1.0}], store.part_path("latch", "dev", "states", [1050], tmp_path))
    store.write_table_atomic([{"seed": 1001, "x": 0.0}], store.part_path("latch", "dev", "states", [1001], tmp_path))
    merged = store.merge("latch", "dev", "states", tmp_path)
    assert list(merged["seed"]) == [1001, 1050, 1051]
    assert store.load("latch", "dev", "states", tmp_path).equals(merged)
    assert not list((tmp_path / "latch" / "dev" / "states").glob("*.tmp"))          # written whole or not at all
    store.write_table_atomic([{"seed": 1001, "x": 9.0}], store.part_path("latch", "dev", "states", [1101], tmp_path))
    with pytest.raises(RuntimeError, match="duplicate"):
        store.merge("latch", "dev", "states", tmp_path)


def test_frames_go_into_one_stored_zip_per_block(tmp_path):
    import numpy as np

    path = store.part_path("fric", "dev", "frames", [1000], tmp_path)
    frames = store.FrameZip(path)
    rgb = (np.arange(48 * 64 * 3) % 255).astype(np.uint8).reshape(48, 64, 3)
    name = frames.add("1000_right_camera_d405.jpg", rgb)
    frames.close()
    assert path.name == "frames_001000_001049.zip" and path.is_file()
    back = store.read_frame("fric", "dev", 1000, name, tmp_path)
    assert back.shape == (48, 64, 3)
    assert abs(back.astype(int) - rgb.astype(int)).mean() < 12            # JPEG quality 92, close but not exact


def test_the_test_split_is_refused_without_the_final_run_flag(tmp_path):
    from avail.zoo.features import run_features
    from avail.zoo.generate import generate

    with pytest.raises(TestSeedError):
        generate("latch", "test", [16000, 16001], workers=1, root=tmp_path)
    with pytest.raises(TestSeedError):
        run_features("latch", "test", [16000, 16001], workers=1, root=tmp_path)
    with pytest.raises(TestSeedError):
        generate("latch", "dev", [16000], workers=1, root=tmp_path)             # a test seed under another name
    assert not any(tmp_path.rglob("*.parquet"))                                  # nothing was simulated


def test_a_connector_without_a_model_is_refused(tmp_path):
    from avail.zoo.generate import generate

    with pytest.raises(ValueError, match="not modeled"):
        generate("prong", "dev", [100], workers=1, root=tmp_path)


@needs_station
@pytest.mark.slow
def test_generate_writes_states_and_frames_and_resumes(tmp_path):
    from avail.zoo.generate import generate

    seeds = list(range(2998, 3003))                         # 2 blocks: 2998 to 2999 and 3000 to 3002
    first = generate("latch", "dev", seeds, workers=2, root=tmp_path)
    assert first["blocks_done"] == 2 and first["blocks_left"] == 0 and first["states"] == 5
    table = store.load("latch", "dev", "states", tmp_path)
    assert list(table["seed"]) == seeds
    assert set(table["cls"]) <= {"P0", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9"}
    assert table["label"].dtype == bool
    for col in ("true_shortfall_mm", "skill_flag", "claim_predicate", "frame_right_camera_d405", "frames_zip"):
        assert col in table.columns
    frame = store.read_frame("latch", "dev", 2998, table.loc[0, "frame_right_camera_d405"], tmp_path)
    assert frame.shape == (480, 640, 3)
    again = generate("latch", "dev", seeds, workers=2, root=tmp_path)
    assert again["blocks_done"] == 0 and again["blocks_skipped"] == 2                # nothing is simulated twice
    assert pd.read_parquet(tmp_path / "latch" / "dev" / "states.parquet").equals(table)
