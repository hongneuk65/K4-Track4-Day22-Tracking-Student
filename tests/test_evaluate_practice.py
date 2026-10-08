"""Kiểm tra cấu hình và cách gọi bộ chấm, không cần dữ liệu thật."""

import json
from evaluate_practice import _load_eval_config, run_trackeval, stage


def test_missing_config_uses_lab_default(tmp_path):
    assert _load_eval_config(tmp_path) == {"benchmark": "LAB_TRACKING", "split": "train"}


def test_supplied_config_is_preserved(tmp_path):
    folder = tmp_path / "video_1"
    folder.mkdir()
    config = {"benchmark": "LAB", "split": "test"}
    (folder / "eval_config.json").write_text(json.dumps(config), encoding="utf-8")
    assert _load_eval_config(tmp_path) == config


def test_stage_respects_split(tmp_path):
    data = tmp_path / "lab" / "video_1"
    (data / "gt").mkdir(parents=True)
    (data / "gt" / "gt.txt").write_text("nhãn", encoding="utf-8")
    (data / "seqinfo.ini").write_text("cấu hình", encoding="utf-8")
    submission = tmp_path / "video_1.txt"
    submission.write_text("kết quả", encoding="utf-8")
    root = tmp_path / "eval"
    stage(root, data.parent, submission, "nhom", "LAB", "test")
    assert (root / "data/gt/mot_challenge/LAB-test/video_1/gt/gt.txt").exists()
    assert (root / "data/trackers/mot_challenge/LAB-test/nhom/data/video_1.txt").exists()


def test_numpy_patch_runs_in_child(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("evaluate_practice.subprocess.run", lambda cmd, check: calls.append(cmd))
    run_trackeval(tmp_path, "nhom", "LAB", "train")
    command = calls[0]
    assert command[1] == "-c"
    assert "np.float = float; np.int = int" in command[2]
    assert command[command.index("--SEQ_INFO") + 1] == "video_1"


def test_child_can_use_removed_numpy_aliases(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    output = tmp_path / "aliases_ok.txt"
    script = (
        "import numpy as np\n"
        "from pathlib import Path\n"
        "assert np.array([1], dtype=np.float).dtype == np.dtype(float)\n"
        "assert np.array([1]).astype(np.int)[0] == 1\n"
        f"Path({str(output)!r}).write_text('ok')\n"
    )
    (scripts / "run_mot_challenge.py").write_text(script, encoding="utf-8")
    run_trackeval(tmp_path, "nhom", "LAB", "train")
    assert output.read_text() == "ok"
