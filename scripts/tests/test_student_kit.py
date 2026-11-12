# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Tests of the student-kit changes: trial labels, green sync LED, motor choice.

Run from the repository root: python -m pytest scripts/tests
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duckiebot_analysis.analysis import case_study as cs  # noqa: E402
from duckiebot_analysis.data.layout import parse_motor_folder  # noqa: E402
from duckiebot_analysis.data.pilogs import parse_trial_id  # noqa: E402
from duckiebot_analysis.tracking.blobs import led_grid  # noqa: E402


@pytest.mark.parametrize(
    ("text", "label", "side", "motors", "channels"),
    [
        (
            "R5-L-m9_B_p050_rep03_20261024_113544",
            "R5-L-m9",
            "L",
            ("m9",),
            ("motor_left",),
        ),
        (
            "R8-R-m16_C_b060p040_rep01_20261024_113544",
            "R8-R-m16",
            "R",
            ("m16",),
            ("motor_right",),
        ),
        (
            "R5-both-m9-m10_E_air_p060_filmedLeft_rep01_20261024_113544",
            "R5-both-m9-m10",
            "both",
            ("m9", "m10"),
            ("motor_left", "motor_right"),
        ),
        (
            "R2-m4_A_sweep_forward_010to036by020_rep01_20261031_101500",
            "R2-m4",
            "R",
            ("m4",),
            ("motor_right",),
        ),
        (
            "R3-L_A_sweep_forward_010to036by020_rep01_20261024_211603",
            "R3-L",
            "L",
            (),
            ("motor_left",),
        ),
        (
            "R3-both_E_air_p060_filmedLeft_rep01_20261025_101500",
            "R3-both",
            "both",
            (),
            ("motor_left", "motor_right"),
        ),
    ],
)
def test_trial_ids_round_trip(text, label, side, motors, channels):
    tid = parse_trial_id(text)
    assert (tid.label, tid.side, tid.motors) == (label, side, motors)
    assert tid.channels == channels
    assert str(tid) == text


@pytest.mark.parametrize(
    "text",
    [
        "R9-L-m1_B_p050_rep01_20261024_113544",
        "R1-L-m17_B_p050_rep01_20261024_113544",
        "R1-X_B_p050_rep01_20261024_113544",
        "",
    ],
)
def test_invalid_trial_ids_rejected(text):
    with pytest.raises(ValueError):
        parse_trial_id(text)


def test_student_folder_name():
    assert parse_motor_folder("db17-r5-left-m9") == ("r5", "left")


def test_led_signal_rises_for_its_own_colour():
    frame = np.zeros((40, 40, 3), np.uint8)
    off = led_grid(frame, 20.5, 20.5, colour="green")
    frame[..., 1] = 200  # a green LED turns on
    assert led_grid(frame, 20.5, 20.5, colour="green").min() > off.max()
    assert led_grid(frame, 20.5, 20.5, colour="red").max() < off.min()
    with pytest.raises(ValueError):
        led_grid(frame, 20.5, 20.5, colour="blue")


def test_configure_points_at_one_motor(tmp_path):
    config = cs.configure("db17-r2-right-m4", "R2-m4")
    with pytest.raises(ValueError):
        cs.configure("db17-r1-both-m1-m2", "R1-both-m1-m2")
    (tmp_path / "R2-m4_C_b060p040-m055_a.parquet").touch()
    (tmp_path / "R2-m4_C_b060p040-m055_b.parquet").touch()
    with pytest.raises(FileNotFoundError):
        cs._one_run(tmp_path, config.e2_run)


# ------------------------------------------------- case study input checks

_RUNS_HEADER = (
    "run,video_file,video_sha256,video_bytes,duration_s,resolution,fps_nominal,"
    "mode,experiment,sync,pi_log,trials,notes\n"
)


def _trace_columns(trial: str, duty: list[float]) -> dict[str, list]:
    n = len(duty)
    return {
        "trial": [trial] * n,
        "t_trial_s": [k / 240 for k in range(n)],
        "omega_rad_s": [0.0] * n,
        "omega_sd_rad_s": [0.1] * n,
        "u_cmd": duty,
    }


def _write_run(path: Path, trials: list[tuple[str, list[float]]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols: dict[str, list] = {}
    for name, duty in trials:
        for k, v in _trace_columns(name, duty).items():
            cols.setdefault(k, []).extend(v)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path)


def _motion_synced_data(root: Path) -> str:
    """Write processed runs of the default motor whose E2 trial lacks LED sync."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    proc = root / "processed" / cs.MOTOR
    e1 = cs.ARTICLE.e1_trial + "20261024_211603"
    neg = cs.ARTICLE.e1_trial_neg + "20261024_211708"
    e2 = cs.ARTICLE.e2_prefix + "01_20261024_212000"
    _write_run(
        proc / f"{cs.LABEL}_A_sweep_dense_x.parquet", [(e1, [0.0]), (neg, [0.0])]
    )
    _write_run(proc / f"{cs.LABEL}_C_b060p040_x.parquet", [(e2, [0.0] * 10)])
    rows = [
        {"trial": e1, "sync": "log+led", "drift": 1e-5, "timing_sd_s": 1e-4},
        {"trial": neg, "sync": "log+led", "drift": 1e-5, "timing_sd_s": 1e-4},
        {"trial": e2, "sync": "log+motion", "drift": None, "timing_sd_s": None},
    ]
    pq.write_table(pa.Table.from_pylist(rows), proc / "trials.parquet")
    return e2


def test_case_study_needs_led_synchronisation(tmp_path):
    trial = _motion_synced_data(tmp_path)
    with pytest.raises(ValueError, match="LED") as exc:
        cs.case_study(tmp_path)
    assert trial in str(exc.value)
    assert "log+led" in str(exc.value) and "script+led" in str(exc.value)


def test_plot_script_reports_motion_sync_without_traceback(tmp_path):
    import subprocess

    trial = _motion_synced_data(tmp_path)
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [
            sys.executable,
            "scripts/plot_case_study_csme2026.py",
            "--data",
            str(tmp_path),
            "--figures",
            str(tmp_path / "figures"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode != 0
    assert "Traceback" not in out.stderr
    assert trial in out.stderr and "LED" in out.stderr


def test_speeds_at_06_skips_trials_without_a_hold_at_06(tmp_path):
    hold = [0.0] * 240 + [0.6] * 480 + [1.0] * 240
    short = [0.0] * 240 + [0.6] * 120 + [1.0] * 240
    good = tmp_path / "db17-a-left-m1" / "R1-L-m1_C_b060p040_x.parquet"
    _write_run(good, [("R1-L-m1_C_b060p040_rep01_20261024_100000", hold)])
    bad = tmp_path / "db17-b-right-m2" / "R2-R-m2_C_b060p040_x.parquet"
    _write_run(bad, [("R2-R-m2_C_b060p040_rep01_20261024_100000", short)])
    assert cs.speeds_at_06(tmp_path) == {"db17-a-left-m1": 0.0}


def test_speeds_at_06_counts_one_motor_on_two_mountings(tmp_path):
    hold = [0.0] * 240 + [0.6] * 480 + [1.0] * 240
    for robot in ("r5", "r6"):
        label = f"{robot.upper()}-L-m9"
        path = tmp_path / f"db17-{robot}-left-m9" / f"{label}_C_b060p040_x.parquet"
        _write_run(path, [(f"{label}_C_b060p040_rep01_20261024_100000", hold)])
    assert cs.speeds_at_06(tmp_path) == {"db17-r5-left-m9": 0.0, "db17-r6-left-m9": 0.0}


def test_speeds_at_06_with_no_usable_trial_is_empty(tmp_path):
    bad = tmp_path / "db17-b-right-m2" / "R2-R-m2_C_b060p040_x.parquet"
    _write_run(bad, [("R2-R-m2_C_b060p040_rep01_20261024_100000", [0.6] * 10)])
    assert cs.speeds_at_06(tmp_path) == {}


def test_quality_tells_to_process_a_tracked_run(tmp_path):
    from duckiebot_analysis.analysis import quality

    raw = tmp_path / "raw" / "db17-r5-left-m9"
    raw.mkdir(parents=True)
    (raw / "setup.toml").write_text(
        'robot = "r5"\nmotor = "left"\nmotor_label = "m9"\n', encoding="utf-8"
    )
    sha = "0" * 64
    rows = [
        f"{run},{run}.mp4,{sha},1,1,1280x720,240,stand,C,log+led,"
        f"pi/{run}_t_events.csv,{run}_t,\n"
        for run in ("R5-L-m9_C_tracked", "R5-L-m9_C_untracked")
    ]
    (raw / "runs.csv").write_text(_RUNS_HEADER + "".join(rows), encoding="utf-8")
    (raw / "frames").mkdir()
    (raw / "frames" / "R5-L-m9_C_tracked.parquet").touch()
    fq = quality.assess_folder(raw, tmp_path / "processed" / raw.name, motion_bias={})
    quality.finish([fq])
    tracked, untracked = fq.runs
    message = " ".join(i.text for i in tracked.issues)
    assert "not processed" in message and "frame table" not in message
    assert "process_raw.py" in tracked.trials[0].recommendation
    assert "track_video.py" not in tracked.trials[0].recommendation
    assert "no frame table" in " ".join(i.text for i in untracked.issues)
    assert "track_video.py" in untracked.trials[0].recommendation
    for rec in (tracked.trials[0].recommendation, untracked.trials[0].recommendation):
        assert "data/README.md" not in rec


def test_student_scripts_cite_no_missing_data_readme():
    scripts = Path(__file__).resolve().parents[1]
    paths = [p for p in scripts.rglob("*.py") if "tests" not in p.parts]
    if not (scripts.parent / "data" / "README.md").exists():
        cited = [p.name for p in paths if "data/README.md" in p.read_text("utf-8")]
        assert cited == []


# ------------------------------------------------- case study on synthetic data


def _run_plot_script(data: Path, figures: Path):
    import subprocess

    root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [
            sys.executable,
            "scripts/plot_case_study_csme2026.py",
            "--data",
            str(data),
            "--figures",
            str(figures),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def synthetic_case_study(tmp_path_factory):
    from synthetic_case_study import write_case_study_data

    data = tmp_path_factory.mktemp("data")
    write_case_study_data(data, cs.MOTOR, cs.LABEL)
    return cs.case_study(data)


def test_plot_script_writes_nothing_when_the_wheel_never_stops(tmp_path):
    from synthetic_case_study import write_case_study_data

    # The sweep ends at 0.2, where the wheel still turns, so no stop duty exists.
    write_case_study_data(
        tmp_path / "data", cs.MOTOR, cs.LABEL, forward=(0.1, 0.2, 0.3, 0.2)
    )
    out = _run_plot_script(tmp_path / "data", tmp_path / "figures")
    assert out.returncode == 1
    assert "Traceback" not in out.stderr
    assert out.stderr.startswith("error:")
    assert "stop" in out.stderr and "sweep" in out.stderr
    figures = tmp_path / "figures"
    assert not figures.exists() or list(figures.iterdir()) == []


@pytest.mark.parametrize(
    ("change", "words"),
    [
        ({"e1_stop": float("nan")}, ("StopDuty", "forward", "stops")),
        ({"e1_neg_stop": float("nan")}, ("NegStopDuty", "backward", "stops")),
        ({"e1_start": float("nan")}, ("StartDuty", "forward", "starts")),
        ({"e1_neg_start": float("nan")}, ("NegStartDuty", "backward", "starts")),
        ({"speeds_06": {}}, ("SpeedSix", "0.6")),
    ],
)
def test_macros_name_the_quantity_that_cannot_be_computed(
    synthetic_case_study, change, words
):
    import dataclasses

    from duckiebot_analysis.analysis.case_study_macros import case_study_macros

    with pytest.raises(ValueError) as exc:
        case_study_macros(dataclasses.replace(synthetic_case_study, **change))
    for word in words:
        assert word in str(exc.value)


def test_plot_script_writes_the_macros_of_synthetic_data(tmp_path):
    from synthetic_case_study import write_case_study_data

    write_case_study_data(tmp_path / "data", cs.MOTOR, cs.LABEL)
    out = _run_plot_script(tmp_path / "data", tmp_path / "figures")
    assert out.returncode == 0, out.stderr
    text = (tmp_path / "figures" / "results.tex").read_text(encoding="utf-8")
    assert "\\newcommand{\\StopDuty}{0.10}" in text


def test_case_study_needs_an_e2_trial_with_the_expected_prefix(tmp_path):
    from synthetic_case_study import write_case_study_data

    write_case_study_data(
        tmp_path, cs.MOTOR, cs.LABEL, e2_prefix=f"{cs.LABEL}_C_b060p040x_rep"
    )
    with pytest.raises(ValueError) as exc:
        cs.case_study(tmp_path)
    message = str(exc.value)
    assert f"{cs.LABEL}_C_b060p040_x" in message
    assert cs.ARTICLE.e2_prefix in message
    assert "runs.csv" in message


def test_case_study_accepts_duties_logged_with_rounding_error(tmp_path):
    from synthetic_case_study import write_case_study_data

    # A script that computes its duties, such as 6 * 0.1, logs 0.6000000000000001.
    write_case_study_data(
        tmp_path, cs.MOTOR, cs.LABEL, duty_text=lambda u: f"{round(u * 10) * 0.1!r}"
    )
    result = cs.case_study(tmp_path)
    assert result.steps["up06"].u_to == 0.6
    assert result.speeds_06 == pytest.approx({cs.MOTOR: 12.0}, abs=0.1)


def test_read_runs_rejects_a_run_listed_twice(tmp_path):
    from duckiebot_analysis.data.layout import read_runs

    run, sha = "R5-L-m9_C_b060p040_20261024_113544", "0" * 64
    row = f"{run},{run}.mp4,{sha},1,1,1280x720,240,stand,C,log+led,,,\n"
    (tmp_path / "runs.csv").write_text(_RUNS_HEADER + row + row, encoding="utf-8")
    with pytest.raises(ValueError, match="delete") as exc:
        read_runs(tmp_path)
    assert run in str(exc.value)
    (tmp_path / "runs.csv").write_text(_RUNS_HEADER + row, encoding="utf-8")
    assert [r.run for r in read_runs(tmp_path)] == [run]


def test_read_trials_reads_the_run_of_each_trial(tmp_path):
    from duckiebot_analysis.data.layout import TRIALS_COLUMNS, read_trials

    trial, run = "R5-L-m9_C_b060p040_rep01_20261024_113544", "R5-L-m9_C_x"
    (tmp_path / "trials.csv").write_text(
        ",".join(TRIALS_COLUMNS) + f"\n{trial},{run},,,,,,,completed,\n",
        encoding="utf-8",
    )
    assert [(t.trial, t.run) for t in read_trials(tmp_path)] == [(trial, run)]


@pytest.mark.parametrize(
    "folder", ["data/raw/db17-r5-left-m9", "db17-r5-left-m9/", "db17-r5-lefty-m9", ""]
)
def test_configure_takes_a_motor_folder_name_only(folder):
    with pytest.raises(ValueError) as exc:
        cs.configure(folder, "R5-L-m9")
    assert "folder name" in str(exc.value) and "db17-r5-left-m9" in str(exc.value)


def test_configure_explains_why_a_both_folder_is_refused():
    with pytest.raises(ValueError, match="one wheel"):
        cs.configure("db17-r1-both-m1-m2", "R1-both-m1-m2")


# ------------------------------------------------- property tests of the name parsers

from duckiebot_analysis.data.layout import (  # noqa: E402
    is_motor_folder_name,
    motor_folder_name,
    motor_ids,
)
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

# No example database, so that a test run writes nothing outside scripts/.
_PROPERTY = settings(database=None, max_examples=300)
_MOTOR_IDS = st.integers(1, 16).map(lambda k: f"m{k}")
_LABELS = st.one_of(
    st.tuples(st.sampled_from("LR"), _MOTOR_IDS).map(lambda x: f"{x[0]}-{x[1]}"),
    st.tuples(_MOTOR_IDS, _MOTOR_IDS)
    .filter(lambda x: x[0] != x[1])
    .map(lambda x: f"both-{x[0]}-{x[1]}"),
    st.sampled_from(["L", "R", "both"]),
    _MOTOR_IDS,
)
_TAGS = st.from_regex(r"[A-Za-z0-9][A-Za-z0-9_-]{0,12}", fullmatch=True)
_STARTS = st.datetimes(
    min_value=datetime(2000, 1, 1), max_value=datetime(2099, 12, 31)
).map(lambda d: d.replace(microsecond=0))


@_PROPERTY
@given(
    st.integers(1, 8),
    _LABELS,
    st.sampled_from("ABCDE"),
    _TAGS,
    st.integers(0, 99),
    _STARTS,
)
def test_trial_id_round_trips(robot, rest, experiment, tag, rep, start):
    label = f"R{robot}-{rest}"
    text = f"{label}_{experiment}_{tag}_rep{rep:02d}_{start:%Y%m%d_%H%M%S}"
    tid = parse_trial_id(text)
    assert str(tid) == text
    assert (tid.label, tid.robot, tid.experiment, tid.tag, tid.rep, tid.start) == (
        label,
        f"R{robot}",
        experiment,
        tag,
        rep,
        start,
    )
    assert tid.side in ("L", "R", "both")


@_PROPERTY
@given(st.text(max_size=60).filter(lambda s: "_rep" not in s))
def test_text_without_a_repeat_is_no_trial_id(text):
    with pytest.raises(ValueError):
        parse_trial_id(text)


# Per part of a trial identifier: values that parse and values that do not.
_TRIAL_PARTS = {
    "label": (
        ["R1-L-m1", "R8-both-m3-m4", "R2-m4", "R3-both"],
        ["R0-L-m1", "R9-L-m1", "R1-L-m0", "R1-L-m17", "R1-X", "r1-L", "R1-both-m1"],
    ),
    "experiment": (["A", "E"], ["F", "a", "AB"]),
    "rep": (["rep01", "rep99"], ["rep1", "rep001", "Rep01"]),
    "stamp": (
        ["20261024_113544", "20000229_000000"],
        ["20261324_113544", "20261024_246000", "2026102_113544", "20010229_000000"],
    ),
}


@_PROPERTY
@given(st.sampled_from(sorted(_TRIAL_PARTS)), st.data())
def test_a_trial_id_with_a_bad_part_is_rejected(bad, data):
    part = {k: data.draw(st.sampled_from(v[k == bad])) for k, v in _TRIAL_PARTS.items()}
    text = f"{part['label']}_{part['experiment']}_tag_{part['rep']}_{part['stamp']}"
    with pytest.raises(ValueError):
        parse_trial_id(text)


_NAMES = st.from_regex(r"[a-z0-9]{1,8}", fullmatch=True)


@st.composite
def _folders(draw):
    robot = draw(_NAMES)
    motor = draw(st.sampled_from(["left", "right", "both"]))
    if draw(st.booleans()):
        ids: tuple[str, ...] = ()  # named before identifiers were required
    elif motor == "both":
        left = draw(_NAMES)
        ids = (left, draw(_NAMES.filter(lambda x: x != left)))
    else:
        ids = (draw(_NAMES),)
    return robot, motor, ids


@_PROPERTY
@given(_folders())
def test_motor_folder_names_round_trip(parts):
    robot, motor, ids = parts
    name = "-".join(("db17", robot, motor, *ids))
    assert motor_folder_name(robot, motor, ids) == name
    assert is_motor_folder_name(name)
    assert parse_motor_folder(name) == (robot, motor)
    assert motor_ids(name) == ids


@_PROPERTY
@given(st.text(max_size=40).filter(lambda s: not s.lower().startswith("db17")))
def test_a_name_not_starting_with_db17_is_no_motor_folder(name):
    assert not is_motor_folder_name(name)
    with pytest.raises(ValueError):
        parse_motor_folder(name)


@_PROPERTY
@given(_folders(), st.sampled_from(["upper", "extra id", "same ids", "suffix"]))
def test_a_misnamed_motor_folder_is_claimed_but_rejected(parts, fault):
    robot, motor, ids = parts
    if fault == "upper":
        robot = robot.upper() + "X"
    elif fault == "extra id":
        ids = (*ids, "m99") if len(ids) == 2 else ("m97", "m98", "m99")
    elif fault == "same ids":
        motor, ids = "both", ("m1", "m1")
    else:
        ids = (*ids[:-1], f"{ids[-1]}_old") if ids else ("left_old",)
    name = "-".join(("db17", robot, motor, *ids))
    assert is_motor_folder_name(name)
    with pytest.raises(ValueError):
        parse_motor_folder(name)


def test_configure_returns_a_frozen_config_and_changes_no_module_state():
    import dataclasses

    config = cs.configure("db17-r2-right-m4", "R2-m4")
    assert (config.folder, config.label, config.channel) == (
        "db17-r2-right-m4",
        "R2-m4",
        "motor_right",
    )
    assert config.e2_run == "R2-m4_C_b060p040*.parquet"
    assert config.e2_prefix == "R2-m4_C_b060p040_rep"
    assert cs.ARTICLE == cs.configure(cs.MOTOR, cs.LABEL)
    assert cs.ARTICLE.channel == "motor_left"
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.label = "R3-L"


def test_case_study_analyses_the_motor_of_its_config(tmp_path):
    from synthetic_case_study import write_case_study_data

    trials = write_case_study_data(tmp_path, "db17-r2-right-m4", "R2-m4")
    result = cs.case_study(tmp_path, cs.configure("db17-r2-right-m4", "R2-m4"))
    assert result.e2_trials == sorted(t for t in trials if "_C_" in t)
    assert result.e1_stop == pytest.approx(0.1)
    with pytest.raises(FileNotFoundError):
        cs.case_study(tmp_path)  # the article's motor has no data here


def test_case_study_reads_each_pi_log_once(tmp_path, monkeypatch):
    from collections import Counter

    from synthetic_case_study import write_case_study_data

    trials = write_case_study_data(tmp_path, cs.MOTOR, cs.LABEL)
    reads: Counter[str] = Counter()
    read = cs.read_pi_log

    def counting(path, *args, **kwargs):
        reads[Path(path).name] += 1
        return read(path, *args, **kwargs)

    monkeypatch.setattr(cs, "read_pi_log", counting)
    cs.case_study(tmp_path)
    assert reads == Counter(f"{t}_events.csv" for t in trials)
