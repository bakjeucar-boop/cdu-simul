"""세션 7.49·7.50 — 네 상태(정상·유휴·막힘·샘)의 운전 결과값 표.

**표는 CSV 에서 뽑기만 한다.** 표의 수는 전부 `results/cdu_dataset.csv` 에서
읽은 것이고 단위 환산이 없다. **판정하지 않는다** — 「신호가 있다·없다」를 적지
않고 값만 놓는다.

**평균으로 뭉개지 않는다.** 가정치 축이 여섯이라 한 상태 안에도 여러 행이
있으므로, 대표 조합 하나를 고정해 **행 하나**를 집어 낸다(`_one_row`).

세션 7.50 이 바꾼 것 둘:
⑴ 표를 **안 ㄴ 하나로 고정**했다(안 ㄱ 은 지웠다 · C2).
⑵ 대표 조합을 **5장 범위의 양 끝 두 벌**(`ENDS`)로 냈다 — 중점은 만들지 않는다.

**한 곳만 물리 모델을 부른다** — C5(퇴화 배치의 총유량 차) 절이다. 그 절은
CSV 를 덮지 않고 이 문서에 수를 적기만 한다.

**모든 수치는 가정값 기반이며 실측이 아니다.**
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from cdu_simul.dataset import _range_axis_values
from cdu_simul.hydraulics import rated_property_temperature_C
from cdu_simul.massloss import (
    SWEEP_FRACTIONS,
    MassLossTopology,
    solve_massloss,
)
from cdu_simul.massloss_thermal import massloss_sizes_Lps, solve_massloss_steady
from cdu_simul.model import CduCase, default_cdu_cases

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "results" / "cdu_dataset.csv"
OUTPUT_PATH = REPO_ROOT / "docs" / "session749-state-table.md"

#: 어느 조합에서도 같은 고정 — 단일 CDU 하나.
#: 다중 CDU 는 이웃 CDU 의 기대 부호가 안 정해졌다(7.45).
FIXED_COMMON: dict[str, object] = {"cdu_config": "single", "cdu_index": 0}

#: 5장 범위의 **양 끝 두 벌**. 값은 전부 CSV 열에 실제로 있는 것이고
#: 이 판이 새로 만든 숫자가 아니다 — 각 열의 고유값이 정확히 이 둘뿐이다.
#: 중점을 만들지 않는다.
ENDS: dict[str, dict[str, float]] = {
    "low": {
        "pump_head_rated_mAq": 20.0,  # Range(20, 30) 의 낮은 끝
        "branch_dp_rated_mAq": 2.0,  # Range(2, 3) 의 낮은 끝
        "valve_dp_rated_mAq": 3.0,  # Range(3, 5) 의 낮은 끝
        "ntu": 2.0,  # Range(2, 3) 의 낮은 끝
        "T_secondary_supply_C": 27.0,  # Range(27, 30) 의 낮은 끝
    },
    "high": {
        "pump_head_rated_mAq": 30.0,  # Range(20, 30) 의 높은 끝
        "branch_dp_rated_mAq": 3.0,  # Range(2, 3) 의 높은 끝
        "valve_dp_rated_mAq": 5.0,  # Range(3, 5) 의 높은 끝
        "ntu": 3.0,  # Range(2, 3) 의 높은 끝
        "T_secondary_supply_C": 30.0,  # Range(27, 30) 의 높은 끝
    },
}

END_LABELS: dict[str, str] = {"low": "낮은 끝", "high": "높은 끝"}

#: 「샘」 구조 자유도 여섯 배치 중 대표 — `g=0` · 펌프=환수.
#: 퇴화 배치와 **한 자리만 다르게** 골랐다(펌프 위치).
REP_TOPOLOGY: tuple[float, bool] = (0.0, False)
#: 퇴화 배치 — `g=0` · 펌프=공급. 수력이 「샘」을 못 본다(5-1 한계 ⑴ · 7.46·7.47).
DEGENERATE_TOPOLOGY: tuple[float, bool] = (0.0, True)

RATED_LOAD = 100.0
IDLE_LOAD = 20.0

#: (표시 이름, CSV 열). 값 표와 차이 표에 모두 나온다.
VALUE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("총유량 L/s", "total_flow_Lps"),
    ("공급−환수 불일치 L/s", "massloss_flow_Lps"),
    ("펌프 양정 mAq", "pump_head_mAq"),
    ("공급온도 ℃", "T_supply_C"),
    ("환수온도 ℃", "T_return_C"),
    ("HX duty kW", "hx_duty_kW"),
    ("랙0 유량 L/s", "rack0_flow_Lps"),
    ("랙0 출구온도 ℃", "rack0_outlet_C"),
)

#: 값 표에만 붙는 랙 요약 두 열 — 랙 여덟을 다 싣지 않기 위한 줄임이다.
RACK_FLOW_COLUMNS = tuple(f"rack{i}_flow_Lps" for i in range(8))
RACK_OUTLET_COLUMNS = tuple(f"rack{i}_outlet_C" for i in range(8))

Row = tuple[str, str, str, "pd.Series[float]"]


def load_dataset() -> pd.DataFrame:
    """CSV 를 그대로 읽는다. 단위 환산을 하지 않는다."""
    return pd.read_csv(CSV_PATH)


def branch_counts(df: pd.DataFrame) -> list[tuple[str, str, int]]:
    """(갈래, 정의, 행 수) — 7.44 D1 의 갈래를 그대로 센다."""
    normal = (df["scenario_kind"] == "정상") & (df["leak_model"] == "K_approx")
    blockage = (df["scenario_kind"] == "이상") & (df["leak_model"] == "K_approx")
    massloss = df["leak_model"] == "massloss"
    return [
        (
            "정상",
            "scenario_kind=정상 · leak_model=K_approx · level=0.0",
            int(normal.sum()),
        ),
        (
            "막힘",
            "scenario_kind=이상 · leak_model=K_approx · level>0",
            int(blockage.sum()),
        ),
        ("샘", "leak_model=massloss (level 빈칸)", int(massloss.sum())),
    ]


def _fixed_mask(df: pd.DataFrame, end: str) -> "pd.Series[bool]":
    mask = pd.Series(True, index=df.index)
    for column, value in (FIXED_COMMON | ENDS[end]).items():
        mask &= df[column] == value
    return mask


def _one_row(
    df: pd.DataFrame, mask: "pd.Series[bool]", what: str
) -> "pd.Series[float]":
    """조건에 맞는 행이 **정확히 하나**인지 확인하고 그 행을 낸다."""
    picked = df[mask]
    if len(picked) != 1:
        raise ValueError(
            f"{what}: 행이 {len(picked)}개다 — 대표 조합이 행 하나를 집지 못했다"
        )
    return picked.iloc[0]


def select_states(
    df: pd.DataFrame, end: str, load_percent: float
) -> list[tuple[str, str, "pd.Series[float]"]]:
    """한 끝·한 부하 수준에서 (상태, 수준, 행) 목록을 낸다."""
    base = _fixed_mask(df, end) & (df["load_percent"] == load_percent)
    where = f"{end} @{load_percent}%"
    rows: list[tuple[str, str, "pd.Series[float]"]] = [
        (
            "정상",
            "—",
            _one_row(df, base & (df["scenario_kind"] == "정상"), f"정상 {where}"),
        )
    ]

    blockage = base & (df["scenario_kind"] == "이상") & (df["leak_model"] == "K_approx")
    for level in sorted(df.loc[blockage, "blockage_level_percent"].unique()):
        rows.append(
            (
                "막힘",
                f"K +{round(level):d} %",
                _one_row(
                    df,
                    blockage & (df["blockage_level_percent"] == level),
                    f"막힘 {level} {where}",
                ),
            )
        )

    massloss = base & (df["leak_model"] == "massloss")
    for label, (share, pump_supply) in (
        ("샘(대표)", REP_TOPOLOGY),
        ("샘(퇴화)", DEGENERATE_TOPOLOGY),
    ):
        topology = (
            massloss
            & (df["residual_return_share"] == share)
            & (df["pump_sees_supply_flow"] == pump_supply)
        )
        for size in sorted(df.loc[topology, "massloss_size_fraction"].unique()):
            rows.append(
                (
                    label,
                    f"크기 {size:g}",
                    _one_row(
                        df,
                        topology & (df["massloss_size_fraction"] == size),
                        f"{label} {size} {where}",
                    ),
                )
            )
    return rows


def _value(row: "pd.Series[float]", column: str) -> float:
    """CSV 값 하나. `massloss_flow_Lps` 의 빈칸은 0 으로 읽는다.

    빈칸은 K_approx 행(질량보존)에서만 나오고, 전 15,104행에서
    `total_flow_Lps - return_flow_Lps == massloss_flow_Lps` 가 빈칸을 0 으로
    두었을 때 성립한다(불일치 0건). 단위 환산이 아니다.
    """
    value = row[column]
    return 0.0 if pd.isna(value) else float(value)


def _spread(row: "pd.Series[float]", columns: tuple[str, ...]) -> float:
    values = [float(row[c]) for c in columns]
    return max(values) - min(values)


def _md_table(header: list[str], body: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def value_table(rows: list[Row]) -> str:
    """값 표 — 상태별 운전 결과값 + 랙 요약 두 열."""
    header = ["상태", "수준", "부하 %"]
    header += [name for name, _ in VALUE_COLUMNS]
    header += ["랙유량 폭 L/s", "랙출구온도 폭 ℃"]
    body = []
    for state, level, load, row in rows:
        cells = [state, level, load]
        cells += [f"{_value(row, col):.4f}" for _, col in VALUE_COLUMNS]
        cells.append(f"{_spread(row, RACK_FLOW_COLUMNS):.4f}")
        cells.append(f"{_spread(row, RACK_OUTLET_COLUMNS):.4f}")
        body.append(cells)
    return _md_table(header, body)


def _delta(row: "pd.Series[float]", base: "pd.Series[float]", column: str) -> float:
    return _value(row, column) - _value(base, column)


def _delta_cell(delta: float, reference: float) -> str:
    if reference == 0.0:
        return f"{delta:+.4f} (—)"
    return f"{delta:+.4f} ({delta / reference * 100.0:+.2f} %)"


def delta_table(rows: list[Row], bases: list["pd.Series[float]"]) -> str:
    """차이 표 — 각 칸은 `절대 (상대 %)`. 부호를 살린다."""
    header = ["상태", "수준", "부하 %"] + [name for name, _ in VALUE_COLUMNS]
    body = []
    for (state, level, load, row), base in zip(rows, bases, strict=True):
        cells = [state, level, load]
        cells += [
            _delta_cell(_delta(row, base, col), _value(base, col))
            for _, col in VALUE_COLUMNS
        ]
        body.append(cells)
    return _md_table(header, body)


def plan_rows(df: pd.DataFrame, end: str) -> list[Row]:
    """안 ㄴ — 부하를 상태가 아니라 축으로 둔다. 네 상태를 부하 두 수준에서 낸다."""
    rows: list[Row] = []
    for load in (RATED_LOAD, IDLE_LOAD):
        rows += [(s, lv, f"{load:g}", r) for s, lv, r in select_states(df, end, load)]
    return rows


def plan_bases(df: pd.DataFrame, end: str, rows: list[Row]) -> list["pd.Series[float]"]:
    """차이의 기준 — **같은 부하의 정상 행**이다."""
    normal = {
        load: select_states(df, end, load)[0][2] for load in (RATED_LOAD, IDLE_LOAD)
    }
    return [normal[float(load)] for _, _, load, _ in rows]


# ─────────────────────────────────────────────────────────────────────────────
# C4 — 양 끝 대조. **판정하지 않는다.** 값이 어디로 움직이는지만 적는다
# ─────────────────────────────────────────────────────────────────────────────
def _paired_deltas(
    df: pd.DataFrame,
) -> list[tuple[str, str, str, dict[str, tuple[float, float]]]]:
    """같은 (상태·수준·부하) 를 양 끝에서 짝지어 Δ 두 개를 낸다."""
    low, high = (plan_rows(df, e) for e in ("low", "high"))
    base_low = plan_bases(df, "low", low)
    base_high = plan_bases(df, "high", high)
    paired = []
    for (state, level, load, r_lo), b_lo, (s2, lv2, ld2, r_hi), b_hi in zip(
        low, base_low, high, base_high, strict=True
    ):
        if (state, level, load) != (s2, lv2, ld2):
            raise ValueError(f"양 끝 행이 짝지어지지 않는다: {state}/{level}/{load}")
        deltas = {
            col: (_delta(r_lo, b_lo, col), _delta(r_hi, b_hi, col))
            for _, col in VALUE_COLUMNS
        }
        paired.append((state, level, load, deltas))
    return paired


def end_compare_table(df: pd.DataFrame) -> str:
    """각 칸은 `Δ낮은끝 → Δ높은끝` (절대값, 정상 대비)."""
    header = ["상태", "수준", "부하 %"] + [name for name, _ in VALUE_COLUMNS]
    body = []
    for state, level, load, deltas in _paired_deltas(df):
        cells = [state, level, load]
        cells += [
            f"{deltas[col][0]:+.4f} → {deltas[col][1]:+.4f}" for _, col in VALUE_COLUMNS
        ]
        body.append(cells)
    return _md_table(header, body)


def sign_flip_lines(df: pd.DataFrame) -> list[str]:
    """부호가 뒤집히는 자리 — 양 끝에서 Δ 의 부호가 갈리는 칸만 센다.

    부호는 `0.0` 을 제 갈래로 둔다(양수·0·음수 셋). 0 은 「뒤집힘」이 아니라
    **부호가 없는 것**이라, 0 이 낀 짝은 「0 을 지난다」로 따로 적는다.
    """
    flips: list[str] = []
    through_zero: list[str] = []
    for state, level, load, deltas in _paired_deltas(df):
        for name, col in VALUE_COLUMNS:
            lo, hi = deltas[col]
            where = f"{state} · {level} · 부하 {load} % · {name}: {lo:+.4e} → {hi:+.4e}"
            if lo * hi < 0.0:
                flips.append(where)
            elif (lo == 0.0) != (hi == 0.0):
                through_zero.append(where)
    lines = [f"- **부호가 뒤집히는 칸: {len(flips)}개**"]
    lines += [f"  - {w}" for w in flips] or ["  - (없다)"]
    lines.append(f"- **한쪽 끝에서만 정확히 0 인 칸: {len(through_zero)}개**")
    lines += [f"  - {w}" for w in through_zero] or ["  - (없다)"]
    return lines


def magnitude_table(df: pd.DataFrame) -> str:
    """신호 크기가 어느 방향으로 움직이는가 — |Δ| 를 상태별·신호별로 센다.

    정상 행(Δ 가 정의상 0)은 세지 않는다.
    """
    states = ("막힘", "샘(대표)", "샘(퇴화)")
    counts: dict[tuple[str, str], list[int]] = {
        (state, col): [0, 0, 0] for state in states for _, col in VALUE_COLUMNS
    }
    for state, _level, _load, deltas in _paired_deltas(df):
        if state not in states:
            continue
        for _, col in VALUE_COLUMNS:
            lo, hi = (abs(v) for v in deltas[col])
            bucket = 0 if hi > lo else (1 if hi < lo else 2)
            counts[(state, col)][bucket] += 1
    header = ["신호"] + [f"{s} (커짐/작아짐/같음)" for s in states]
    body = [
        [name] + ["/".join(str(n) for n in counts[(state, col)]) for state in states]
        for name, col in VALUE_COLUMNS
    ]
    return _md_table(header, body)


# ─────────────────────────────────────────────────────────────────────────────
# C5 — 퇴화 배치(`g=0` · 펌프=공급)의 총유량 차. **물리 모델을 부르는 유일한 절**
# ─────────────────────────────────────────────────────────────────────────────
#: caveat 문언이 나오는 자리 — 이 판은 **고치지 않는다**(확인만 한다).
CAVEAT_SOURCE = "src/cdu_simul/dataset_plan.py:606-618 `MASSLOSS_PROVENANCE`"


def _end_case(end: str, load_percent: float) -> CduCase:
    """`ENDS[end]` 와 값이 같은 `default_cdu_cases()` 조합 하나를 집는다."""
    wanted = ENDS[end]
    picked = [
        case
        for case in default_cdu_cases()
        if _range_axis_values(case) == wanted
    ]
    if len(picked) != 1:
        raise ValueError(f"{end}: 32조합에서 {len(picked)}개가 집혔다")
    return replace(picked[0], load_percent=load_percent)


def degenerate_lines(df: pd.DataFrame, end: str = "low") -> list[str]:
    """세 기준선에서 잰 퇴화 배치의 총유량 차 — 값만 적는다.

    ㄱ. **CSV · K_approx 정상 행** — 7.49 표가 쓴 기준선. 계산하지 않는다.
    ㄴ. **같은 배치의 Q=0 해 · 열까지 물린 것**(`solve_massloss_steady`)
        — 데이터셋의 「샘」 해가 실제로 도는 경로다.
    ㄷ. **같은 배치의 Q=0 해 · 수력 한정 · 물성 온도 고정**(`solve_massloss`)
        — caveat 의 「정확히 0」이 나온 자리(세션 5.6 관측 ④).

    절대 규칙 5: ㄴ·ㄷ 의 solver 표시를 결과에 함께 싣는다.
    """
    topology = MassLossTopology(
        label="g=0.0/펌프=공급",
        residual_return_share=DEGENERATE_TOPOLOGY[0],
        pump_sees_supply_flow=DEGENERATE_TOPOLOGY[1],
    )
    T_prop_C = rated_property_temperature_C()
    header = [
        "부하 %",
        "크기",
        "ㄱ CSV−K_approx 정상 L/s",
        "ㄴ 모델−같은배치 Q=0 (열 포함) L/s",
        "ㄷ 모델−같은배치 Q=0 (수력 한정) L/s",
    ]
    body: list[list[str]] = []
    iers: set[str] = set()
    baseline_gaps: list[float] = []
    for load in (RATED_LOAD, IDLE_LOAD):
        case = _end_case(end, load)
        sizes_Lps = massloss_sizes_Lps(case.hydraulic, T_prop_C)
        thermal_zero = solve_massloss_steady(case, 0.0, topology)
        hydraulic_zero = solve_massloss(case.hydraulic, 0.0, topology, T_prop_C)
        iers.add(f"열 Q=0 ier={thermal_zero.outer_solver_ier}")
        iers.add(f"수력 Q=0 ier={hydraulic_zero.solver_ier}")
        csv_rows = {
            level: row
            for state, level, row in select_states(df, end, load)
            if state == "샘(퇴화)"
        }
        csv_normal = select_states(df, end, load)[0][2]
        baseline_gaps.append(
            _value(csv_normal, "total_flow_Lps") - thermal_zero.supply_flow_Lps
        )
        for fraction in SWEEP_FRACTIONS[1:]:
            size_Lps = sizes_Lps[SWEEP_FRACTIONS.index(fraction)]
            thermal = solve_massloss_steady(case, size_Lps, topology)
            hydraulic = solve_massloss(case.hydraulic, size_Lps, topology, T_prop_C)
            iers.add(f"열 Q>0 ier={thermal.outer_solver_ier}")
            iers.add(f"수력 Q>0 ier={hydraulic.solver_ier}")
            csv_row = csv_rows[f"크기 {fraction:g}"]
            body.append(
                [
                    f"{load:g}",
                    f"{fraction:g}",
                    f"{_delta(csv_row, csv_normal, 'total_flow_Lps'):+.6e}",
                    f"{thermal.supply_flow_Lps - thermal_zero.supply_flow_Lps:+.6e}",
                    f"{hydraulic.supply_flow_Lps - hydraulic_zero.supply_flow_Lps:+.6e}",
                ]
            )
    return [
        f"기준 조합은 표와 같은 **{END_LABELS[end]}**이고 배치는 "
        "`g=0` · 펌프=공급(퇴화)이다. 총유량은 헤더 공급유량 ΣQ_i 다.",
        "",
        _md_table(header, body),
        "",
        f"- 위 ㄴ·ㄷ 은 **물리 모델을 부른 것**이다. solver 표시: "
        f"{' · '.join(sorted(iers))} (전부 `ier=1`). "
        "CSV 를 덮지 않았다 — 이 표에 적기만 했다.",
        f"- **ㄱ 과 ㄴ 의 기준선이 값으로 같다** — CSV 의 K_approx 정상 행 "
        "`total_flow_Lps` 와 모델의 같은 배치 Q=0 해 공급유량의 차가 "
        f"최대 {max(abs(g) for g in baseline_gaps):.3e} L/s 다(부동소수 잡음 자리). "
        "그래서 ㄱ 열과 ㄴ 열이 같은 수로 나온다.",
    ]


def _solver_flags(rows: list[Row]) -> str:
    hydraulic = sorted({str(row["hydraulic_solver_ier"]) for *_, row in rows})
    thermal = sorted({str(row["thermal_solver_converged"]) for *_, row in rows})
    return (
        f"`hydraulic_solver_ier` {hydraulic} · "
        f"`thermal_solver_converged` {thermal}"
    )


def format_report(df: pd.DataFrame) -> str:
    counts = branch_counts(df)
    total = sum(n for *_, n in counts)
    version = sorted(df["dataset_version"].unique())

    out: list[str] = []
    out.append("# 세션 7.49·7.50 — 네 상태의 운전 결과값 표 (양 끝 두 벌)")
    out.append("")
    out.append("> **가정값 기반 — 실측 아님.** 설계데이터가 없는 파일럿 단계의 값이다.")
    out.append(">")
    out.append(f"> 데이터셋 판본 `dataset_version` = {', '.join(version)} · "
               f"원본 `results/cdu_dataset.csv` ({total:,}행)")
    out.append(">")
    out.append("> **단일 CDU · 정상상태(`regime=steady`, 전이 행 없음).** "
               f"고정: {' · '.join(f'`{k}`={v}' for k, v in FIXED_COMMON.items())}")
    out.append(">")
    out.append("> 가정치 축 다섯(정격양정·분기 ΔP·밸브 ΔP·NTU·2차측 온도)을 "
               "**5장 범위의 양 끝 두 벌**로 낸다 — 중점을 만들지 않았다. "
               "부하는 표의 축이다.")
    out.append(">")
    out.append("> **이 문서는 판정하지 않는다.** 값만 놓는다.")
    out.append(">")
    out.append("> 표의 수는 전부 CSV 에서 읽은 것이다 — 단위를 환산하지 않았다. "
               "**예외는 「퇴화 배치의 총유량 차」 절 하나**이고, 그 절은 물리 "
               "모델을 불러 잰 값임을 그 자리에 밝힌다.")
    out.append("")
    out.append("**세션 7.50 이 「안 ㄱ」(정상을 부하로 갈라 네 상태를 세우는 표)을 "
               "지웠다.** 사람이 안 ㄴ 하나로 정했다(7.49 D8-1) — 안 ㄱ 은 "
               "막힘·샘의 부하 20 % 쪽이 빠져 안 ㄴ 이 덮는 것을 못 덮는다. "
               "**되살리려면 7.49 커밋에서 꺼낸다.**")
    out.append("")

    out.append("## 갈래별 행 수")
    out.append("")
    out.append(_md_table(
        ["갈래", "정의 (CSV 열과 값)", "행 수"],
        [[name, definition, f"{n:,}"] for name, definition, n in counts]
        + [["합", "", f"{total:,}"]],
    ))
    out.append("")
    out.append("「유휴」는 별도 갈래가 아니다 — 위 세 갈래 안의 "
               "`load_percent=20.0` 행이다. 5장 부하 「유휴 20 % ~ 정격 100 %」를 "
               "그대로 읽은 것이라 **새 값이 0개**다.")
    out.append("")

    for end in ("low", "high"):
        rows = plan_rows(df, end)
        bases = plan_bases(df, end, rows)
        axis = " · ".join(f"`{k}`={v:g}" for k, v in ENDS[end].items())
        out.append(f"## 대표 조합 — 5장 범위의 **{END_LABELS[end]}**")
        out.append("")
        out.append(f"{axis}")
        out.append("")
        out.append(f"### {END_LABELS[end]} · 값")
        out.append("")
        out.append(value_table(rows))
        out.append("")
        out.append(f"### {END_LABELS[end]} · 같은 부하의 정상 대비 차이 "
                   "— `절대 (상대 %)`")
        out.append("")
        out.append(delta_table(rows, bases))
        out.append("")
        out.append(f"고른 행의 solver 표시: {_solver_flags(rows)}.")
        out.append("")

    out.append("## 양 끝 대조 — 가정치를 바꾸면 신호가 어디로 움직이는가")
    out.append("")
    out.append("각 칸은 **`Δ낮은끝 → Δ높은끝`**(같은 부하의 정상 대비 절대차)다. "
               "**판정하지 않는다** — 값이 어디로 갔는지만 적는다.")
    out.append("")
    out.append(end_compare_table(df))
    out.append("")
    out.append("### 부호")
    out.append("")
    out += sign_flip_lines(df)
    out.append("")
    out.append("### 크기 — |Δ| 가 높은 끝에서 커지는가 작아지는가")
    out.append("")
    out.append(magnitude_table(df))
    out.append("")
    out.append("칸의 수는 **행 수**다(막힘 6 = 3수준 × 2부하 · 샘 각 8 = 4크기 × "
               "2부하). 정상 행은 Δ 가 정의상 0 이라 세지 않았다.")
    out.append("")

    out.append("## 퇴화 배치의 총유량 차 — caveat 「정확히 0」과 맞춰본다")
    out.append("")
    out.append("`massloss` 행의 `signal_sign_caveat` 은 이렇게 적는다 — "
               "「구조 자유도가 g=0 · 펌프=공급 배치인 행에서 수력 다섯 양이 "
               "「정확히 0」이라는 것은 **수력 한정 · 물성 온도 37 ℃ 고정 "
               "기준선**에서의 말이다. 열까지 함께 풀린 이 데이터셋 행에서는 "
               "정확히 0 이 아니라 총유량 차가 1e-04 자리로 남으며, 갈리는 "
               "원인은 물성 평가 온도 하나다」. "
               f"문언은 {CAVEAT_SOURCE} 에서 온다. "
               "**세션 7.52 가 이 조건을 caveat 안으로 옮겨 적었다** — "
               "7.50 까지는 caveat 이 「정확히 0」이라고만 적어 아래 표와 "
               "어긋나 있었다.")
    out.append("")
    out.append("그 「0」이 **무엇을 기준선으로 한 0 인지**는 "
               "`src/cdu_simul/massloss_gate.py:388-396` 의 주석이 적는다 — "
               "「세션 5.6 관측 ④ 는 … **수력 한정 · 물성 37 ℃ 고정**」이고, "
               "「데이터셋의 「샘」 해는 열까지 물린 것이라 그 배치에서도 Δ 가 "
               "0 이 아니다」. 기준선은 **같은 배치의 Q=0 해**이지 K_approx 정상 "
               "행이 아니다(`massloss_thermal.massloss_thermal_deltas` 의 "
               "docstring: 「기준은 같은 배치의 Q_massloss=0 해다」).")
    out.append("")
    out += degenerate_lines(df)
    out.append("")
    out.append("- **ㄷ 이 caveat 이 말하는 자리다.** 수력 한정·물성 고정이면 "
               "`solve_massloss` 식에서 Q 가 사라져 Q=0 해와 항등이다"
               "(5-1 「「샘」 구조 자유도 셋」 한계 ⑴).")
    out.append("- **ㄴ 은 열이 물린 자리**라 물성 평가 온도 하나가 남는다"
               "(세션 7.46 D6).")
    out.append("- **ㄱ 과 ㄴ 은 기준선이 달라 보이지만 값이 같다**(위 마지막 줄). "
               "7.49 D7-1 은 「두 수는 기준선이 달라 서로 다른 것을 잰 것으로 "
               "보인다」고 적었는데, **재어 보니 그 갈래가 아니다** — "
               "ㄱ 과 ㄴ 은 같은 수이고, 갈리는 것은 **ㄷ**(수력 한정·물성 고정) "
               "하나다.")
    out.append("- **이 판은 문언을 고치지 않았다.** 값만 놓았다.")
    out.append("")

    out.append("## 읽는 데 필요한 것")
    out.append("")
    out.append("- **「샘(퇴화)」는 `g=0` · 펌프=공급 배치다.** 이 배치에서 "
               "수력식이 누출유량을 잃는다(5-1 「「샘」 구조 자유도 셋」 한계 ⑴) "
               "— 총유량·펌프 양정이 정상과 거의 같게 나오는 것이 정상이다. "
               "**온도는 그 배치에서도 응답한다.**")
    out.append("- 「샘(대표)」는 `g=0` · 펌프=환수 배치다. 퇴화 배치와 "
               "**펌프 위치 한 자리만** 다르다 — 두 줄의 차이가 곧 "
               "펌프 위치의 몫이다.")
    out.append("- `공급−환수 불일치` 열은 CSV `massloss_flow_Lps` 다. "
               "K_approx 행(정상·막힘)은 이 열이 빈칸이며 **0 으로 읽었다** — "
               "질량이 보존되기 때문이고, 전 행에서 "
               "`total_flow_Lps − return_flow_Lps` 와 일치한다(불일치 0건).")
    out.append("- 랙 여덟 중 `rack0` 만 대표로 실었고, 나머지는 "
               "`최대−최소` 폭 두 열로 줄였다.")
    out.append("")

    out.append("## 이 표의 한계")
    out.append("")
    out.append("1. **막힘과 샘은 총유량 부호가 정반대다**(미해결 #36). "
               "막힘은 총유량이 줄고 샘은 는다 — 「유량 감소 = 이상」 같은 "
               "규칙을 이 표에서 끌어내면 틀린다. "
               "행마다 어느 기구인지 보고 읽어야 한다.")
    out.append("2. **샘은 어느 랙에서 새는지 특정하지 못한다.** CSV "
               "`anomaly_rack_index` 가 massloss 행에서 빈칸이다 — "
               "계통 전체의 질량손실로만 들어간다.")
    out.append("3. **전이(시간축)가 없다.** 데이터셋 전 행이 `regime=steady` 라 "
               "「몇 초 뒤」는 이 표에 없다(미해결 #40·#38).")
    out.append("4. **양 끝 두 벌은 범위의 두 점이지 범위가 아니다.** 다섯 축을 "
               "한꺼번에 옮겼으므로 **어느 축이 얼마나 움직였는지는 이 표가 "
               "가르지 않는다** — 축 하나씩 흔든 것이 아니다.")
    out.append("5. 단일 CDU 다. 다중 CDU(`dual_symmetric`·`dual_asymmetric`) 행은 "
               "이웃 CDU 의 기대 부호가 정해지지 않아 뺐다(7.45).")
    out.append("")
    return "\n".join(out)


def main() -> int:
    df = load_dataset()
    OUTPUT_PATH.write_text(format_report(df), encoding="utf-8")
    print(f"기록: {OUTPUT_PATH.relative_to(REPO_ROOT)} (가정값 기반 · 실측 아님)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
