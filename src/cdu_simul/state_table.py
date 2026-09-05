"""세션 7.49 — 네 상태(정상·유휴·막힘·샘)의 운전 결과값 표 (뽑는 판).

**계산하지 않는다.** 물리 모델을 한 줄도 부르지 않고 `results/cdu_dataset.csv`
를 읽어 고르고 모으기만 한다. **판정하지 않는다** — 「신호가 있다·없다」를
적지 않고 값만 놓는다.

**평균으로 뭉개지 않는다.** 가정치 축이 여섯이라 한 상태 안에도 여러 행이
있으므로, 대표 조합 하나를 고정해 **행 하나**를 집어 낸다(`FIXED`·`REP_*`).
어느 조합인지는 표 머리에 적힌다.

**모든 수치는 가정값 기반이며 실측이 아니다.**
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "results" / "cdu_dataset.csv"
OUTPUT_PATH = REPO_ROOT / "docs" / "session749-state-table.md"

#: 고정한 대표 조합 — **5장 범위의 낮은 쪽 끝**으로 통일한다(중점을 만들지 않는다).
#: 값은 전부 CSV 열에 있는 것이고 이 판이 새로 만든 숫자가 아니다.
FIXED: dict[str, object] = {
    "cdu_config": "single",  # 단일 CDU — 다중은 이웃 CDU 뜻이 안 정해졌다(7.45)
    "cdu_index": 0,
    "pump_head_rated_mAq": 20.0,  # Range(20, 30) 의 낮은 끝
    "branch_dp_rated_mAq": 2.0,  # Range(2, 3) 의 낮은 끝
    "valve_dp_rated_mAq": 3.0,  # Range(3, 5) 의 낮은 끝
    "ntu": 2.0,  # Range(2, 3) 의 낮은 끝
    "T_secondary_supply_C": 27.0,  # Range(27, 30) 의 낮은 끝
}

#: 「샘」 구조 자유도 여섯 배치 중 대표 — `g=0` · 펌프=환수.
#: 퇴화 배치와 **한 자리만 다르게** 골랐다(펌프 위치). 둘을 나란히 놓으면
#: 퇴화의 원인이 펌프 위치라는 것이 표에서 바로 보인다.
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


def _fixed_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for column, value in FIXED.items():
        mask &= df[column] == value
    return mask


def _one_row(df: pd.DataFrame, mask: pd.Series, what: str) -> pd.Series:
    """조건에 맞는 행이 **정확히 하나**인지 확인하고 그 행을 낸다."""
    picked = df[mask]
    if len(picked) != 1:
        raise ValueError(
            f"{what}: 행이 {len(picked)}개다 — 대표 조합이 행 하나를 집지 못했다"
        )
    return picked.iloc[0]


def select_states(
    df: pd.DataFrame, load_percent: float
) -> list[tuple[str, str, pd.Series]]:
    """한 부하 수준에서 (상태, 수준, 행) 목록을 낸다."""
    base = _fixed_mask(df) & (df["load_percent"] == load_percent)
    rows: list[tuple[str, str, pd.Series]] = [
        (
            "정상",
            "—",
            _one_row(
                df,
                base & (df["scenario_kind"] == "정상"),
                f"정상 @{load_percent}%",
            ),
        )
    ]

    blockage = base & (df["scenario_kind"] == "이상") & (df["leak_model"] == "K_approx")
    for level in sorted(df.loc[blockage, "leak_level_percent"].unique()):
        rows.append(
            (
                "막힘",
                f"K +{round(level):d} %",
                _one_row(
                    df,
                    blockage & (df["leak_level_percent"] == level),
                    f"막힘 {level} @{load_percent}%",
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
                        f"{label} {size} @{load_percent}%",
                    ),
                )
            )
    return rows


def _value(row: pd.Series, column: str) -> float:
    """CSV 값 하나. `massloss_flow_Lps` 의 빈칸은 0 으로 읽는다.

    빈칸은 K_approx 행(질량보존)에서만 나오고, 전 15,104행에서
    `total_flow_Lps - return_flow_Lps == massloss_flow_Lps` 가 빈칸을 0 으로
    두었을 때 성립한다(불일치 0건). 단위 환산이 아니다.
    """
    value = row[column]
    return 0.0 if pd.isna(value) else float(value)


def _spread(row: pd.Series, columns: tuple[str, ...]) -> float:
    values = [float(row[c]) for c in columns]
    return max(values) - min(values)


def _md_table(header: list[str], body: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def value_table(rows: list[tuple[str, str, str, pd.Series]]) -> str:
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


def delta_table(
    rows: list[tuple[str, str, str, pd.Series]], bases: list[pd.Series]
) -> str:
    """차이 표 — 각 칸은 `절대 (상대 %)`. 부호를 살린다."""
    header = ["상태", "수준", "부하 %"] + [name for name, _ in VALUE_COLUMNS]
    body = []
    for (state, level, load, row), base in zip(rows, bases, strict=True):
        cells = [state, level, load]
        for _, col in VALUE_COLUMNS:
            delta = _value(row, col) - _value(base, col)
            reference = _value(base, col)
            if reference == 0.0:
                cells.append(f"{delta:+.4f} (—)")
            else:
                cells.append(f"{delta:+.4f} ({delta / reference * 100.0:+.2f} %)")
        body.append(cells)
    return _md_table(header, body)


def _plan_a(df: pd.DataFrame) -> list[tuple[str, str, str, pd.Series]]:
    """안 ㄱ — 정상을 부하로 갈라 「정상(정격)」·「유휴」 둘로 세운다.

    막힘·샘은 정격(100 %)에 놓는다 — 20 % 쪽은 안 ㄴ 에 있다.
    """
    rated = select_states(df, RATED_LOAD)
    idle_normal = select_states(df, IDLE_LOAD)[0]
    rows: list[tuple[str, str, str, pd.Series]] = [
        ("정상(정격)", "—", f"{RATED_LOAD:g}", rated[0][2]),
        ("유휴", "—", f"{IDLE_LOAD:g}", idle_normal[2]),
    ]
    rows += [(s, lv, f"{RATED_LOAD:g}", r) for s, lv, r in rated[1:]]
    return rows


def _plan_b(df: pd.DataFrame) -> list[tuple[str, str, str, pd.Series]]:
    """안 ㄴ — 부하를 상태가 아니라 축으로 둔다. 네 상태를 부하 두 수준에서 낸다."""
    rows: list[tuple[str, str, str, pd.Series]] = []
    for load in (RATED_LOAD, IDLE_LOAD):
        rows += [(s, lv, f"{load:g}", r) for s, lv, r in select_states(df, load)]
    return rows


def _solver_flags(rows: list[tuple[str, str, str, pd.Series]]) -> str:
    hydraulic = sorted({str(row["hydraulic_solver_ier"]) for *_, row in rows})
    thermal = sorted({str(row["thermal_solver_converged"]) for *_, row in rows})
    return (
        f"`hydraulic_solver_ier` {hydraulic} · "
        f"`thermal_solver_converged` {thermal}"
    )


def format_report(df: pd.DataFrame) -> str:
    counts = branch_counts(df)
    total = sum(n for *_, n in counts)
    plan_a = _plan_a(df)
    plan_b = _plan_b(df)
    base_a = [plan_a[0][3]] * len(plan_a)
    #: 안 ㄴ 의 기준은 **같은 부하의 정상 행**이다.
    normal_by_load = {
        load: select_states(df, load)[0][2] for load in (RATED_LOAD, IDLE_LOAD)
    }
    base_b = [normal_by_load[float(load)] for _, _, load, _ in plan_b]

    fixed_line = " · ".join(f"`{k}`={v}" for k, v in FIXED.items())
    version = sorted(df["dataset_version"].unique())

    out: list[str] = []
    out.append("# 세션 7.49 — 네 상태의 운전 결과값 표")
    out.append("")
    out.append("> **가정값 기반 — 실측 아님.** 설계데이터가 없는 파일럿 단계의 값이다.")
    out.append(">")
    out.append(f"> 데이터셋 판본 `dataset_version` = {', '.join(version)} · "
               f"원본 `results/cdu_dataset.csv` ({total:,}행)")
    out.append(">")
    out.append("> **단일 CDU · 정상상태(`regime=steady`, 전이 행 없음).**")
    out.append(">")
    out.append(f"> 고정한 대표 조합: {fixed_line}")
    out.append(">")
    out.append("> 가정치 축 여섯 중 다섯(정격양정·분기 ΔP·밸브 ΔP·NTU·2차측 온도)은 "
               "**5장 범위의 낮은 쪽 끝**으로 통일해 고정했다 — 중점을 만들지 않았다. "
               "부하는 표의 축이다.")
    out.append(">")
    out.append("> **이 문서는 판정하지 않는다.** 값만 놓는다.")
    out.append(">")
    out.append("> 이 문서의 수는 전부 CSV 에서 읽은 것이다 — **새로 만든 숫자가 없다.** "
               "단위는 CSV 열 그대로이며 환산하지 않았다.")
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

    out.append("## 안 ㄱ — 정상을 부하로 갈라 네 상태를 세운다")
    out.append("")
    out.append("막힘·샘은 정격(100 %)에 놓았다. 20 % 쪽은 안 ㄴ 에 있다.")
    out.append("")
    out.append("### ㄱ-1 값")
    out.append("")
    out.append(value_table(plan_a))
    out.append("")
    out.append("### ㄱ-2 정상(정격) 대비 차이 — `절대 (상대 %)`")
    out.append("")
    out.append(delta_table(plan_a, base_a))
    out.append("")

    out.append("## 안 ㄴ — 부하를 축으로 두고 네 상태를 부하 두 수준에서 낸다")
    out.append("")
    out.append("차이의 기준은 **같은 부하의 정상 행**이다.")
    out.append("")
    out.append("### ㄴ-1 값")
    out.append("")
    out.append(value_table(plan_b))
    out.append("")
    out.append("### ㄴ-2 같은 부하의 정상 대비 차이 — `절대 (상대 %)`")
    out.append("")
    out.append(delta_table(plan_b, base_b))
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
    out.append(f"- 고른 행의 solver 표시: {_solver_flags(plan_b)}.")
    out.append("")

    out.append("## 이 표의 한계")
    out.append("")
    out.append("1. **막힘과 샘은 총유량 부호가 정반대다**(미해결 #36). "
               "막힘은 총유량이 줄고 샘은 는다 — 「유량 감소 = 이상」 같은 "
               "규칙을 이 표에서 끌어내면 틀린다. "
               "행마다 어느 기구인지 보고 읽어야 한다.")
    out.append("2. **샘은 어느 랙에서 새는지 특정하지 못한다.** CSV `leak_rack_index` 가 "
               "massloss 행에서 빈칸이다 — 계통 전체의 질량손실로만 들어간다.")
    out.append("3. **전이(시간축)가 없다.** 데이터셋 전 행이 `regime=steady` 라 "
               "「몇 초 뒤」는 이 표에 없다(미해결 #40·#38).")
    out.append("4. 대표 조합 하나를 고정한 값이다. 5장 범위의 **다른 끝**에서는 "
               "수가 다르다 — 이 표는 범위를 보여주지 않는다.")
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
