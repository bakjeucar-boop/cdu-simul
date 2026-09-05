"""세션 7.45 — 「샘」(질량손실) 게이트 판정. **CSV 를 읽어 판정만 한다.**

판정 기준은 `docs/session745-massloss-gate-criteria.md` 에 **계산보다 먼저** 적고
먼저 커밋했다(선행 커밋 `3001588`). 이 파일은 그 문서를 코드로 옮긴 것이고,
결과를 보고 기준을 고치지 않는다.

**물리 모델을 부르지 않는다.** `results/cdu_dataset.csv` 를 읽어 정상 행과의
차이를 내고 통과·실패·해당 없음을 셀 뿐이다 — 재계산하지 않는다.

**「통과」의 뜻은 「모델 안에서 신호가 잡음 위에 있다」까지다.** 「실측으로 감지
가능하다」는 뜻이 아니다 — 계측기 사양이 없고 실측이 없다(세션 4 판정에 붙은
단서와 같다).

**energy balance 는 판정하지 않는다** — 「샘」 행은 누출 엔탈피 항 없이는 닫히지
않고(세션 5.7-D), 그 항을 넣는 것은 물리 모델 변경이다. 미판정으로 남긴다.

**모든 수치는 가정값 기반이며 실측이 아니다.**
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cdu_simul.assumptions import ASSUMPTION_TAG
from cdu_simul.dataset import (
    DEFAULT_OUTPUT_DIR,
    LEAK_CDU_INDEX,
    LEAK_MODEL_K_APPROX,
    LEAK_MODEL_MASSLOSS,
)

# 부호를 0 으로 읽는 절대 임계. **새 임계값을 만들지 않으려고 세션 5.6 이 쓴 것을
# 그대로 가져온다**. 사(私)이름이라 import 가 어색하지만, 숫자를 한 벌 더 적는
# 것보다 낫다(절대 규칙 1·2).
#
# **세션 7.47 이후 「해당 없음」 판정에는 쓰이지 않는다** — 새 규정(기준 문서
# 2-5-B · 커밋 `9ca4f14`)이 수치 임계를 쓰지 않는다. 남은 쓰임은 공급−환수
# 불일치 Δ≠0 을 세는 자리 하나뿐이다(`_mismatch_lines`).
from cdu_simul.massloss import _SIGN_ZERO_TOL as SIGN_ZERO_TOL

#: 세션 4 가 쓴 잡음 임계 셋. **원본은 `tests/test_session4_gates.py:65·70·75`**
#: 이고 그 근거(세션 3-A/3-B 관측 잡음의 1000배 이상)도 거기 적혀 있다. 시험
#: 모듈을 라이브러리에서 import 하지 않으려고 값만 옮겨 적는다 — 새 값이 아니다.
NOISE_THRESHOLD: dict[str, float] = {
    "%": 1.0e-3,  # FLOW_NOISE_THRESHOLD_PERCENT
    "mAq": 1.0e-4,  # HEAD_NOISE_THRESHOLD_mAq
    "K": 1.0e-3,  # TEMPERATURE_NOISE_THRESHOLD_C
}

#: 기준선 행과 이상 행을 짝짓는 열 여덟 (기준 문서 2-2).
PAIR_COLUMNS: tuple[str, ...] = (
    "pump_head_rated_mAq",
    "branch_dp_rated_mAq",
    "valve_dp_rated_mAq",
    "ntu",
    "T_secondary_supply_C",
    "cdu_config",
    "cdu_index",
    "load_percent",
)

#: 「샘」의 구조 자유도 셋 — 기준 B 의 무리를 가를 때 짝짓는 열에 더한다.
TOPOLOGY_COLUMNS: tuple[str, ...] = (
    "residual_return_share",
    "pump_sees_supply_flow",
    "share_uses_return_flow",
)

PASS, FAIL, NA = "통과", "실패", "해당 없음"


@dataclass(frozen=True)
class Signal:
    """세션 4 가 본 신호 하나 (기준 문서 2-3·2-4).

    `massloss_sign` 0 은 「배치가 정한다」는 뜻이다 — 펌프 양정만 그렇고,
    `pump_sees_supply_flow` 가 참이면 하강(−) 거짓이면 상승(+)이다(미해결 #36).
    """

    label: str
    column: str
    unit: str
    massloss_sign: int
    blockage_sign: int


SIGNALS: tuple[Signal, ...] = (
    Signal("⑴ 총유량", "total_flow_Lps", "%", +1, -1),
    Signal("⑵ 펌프 양정", "pump_head_mAq", "mAq", 0, +1),
    Signal("⑶ 주입랙 통과유량", "rack0_flow_Lps", "%", +1, -1),
    Signal("⑷ 타 랙 유량", "rack1_flow_Lps", "%", +1, +1),
    Signal("⑸ 주입랙 출구온도", "rack0_outlet_C", "K", -1, +1),
)

#: 「해당 없음」 규정이 걸리는 신호 — **수력 넷**이다(기준 문서 2-5-B ⑵).
#: ⑸ 출구온도는 퇴화 배치에서도 물성 온도를 통해 응답하므로 덮지 않는다.
HYDRAULIC_SIGNALS: frozenset[str] = frozenset(
    signal.label for signal in SIGNALS if signal.unit != "K"
)

#: 기구별 「누출 수준」 열 — 기준 B 가 이 열을 따라 단조를 본다.
LEVEL_COLUMN: dict[str, str] = {
    LEAK_MODEL_MASSLOSS: "massloss_size_fraction",
    LEAK_MODEL_K_APPROX: "blockage_level_percent",
}

_READ_COLUMNS: list[str] = [
    "scenario_kind",
    "leak_model",
    "blockage_level_percent",
    "anomaly_cdu_index",
    "massloss_size_fraction",
    "return_flow_Lps",
    "massloss_flow_Lps",
    *PAIR_COLUMNS,
    *TOPOLOGY_COLUMNS,
    *(signal.column for signal in SIGNALS),
]


def read_dataset(csv_path: Path) -> pd.DataFrame:
    """필요한 열만 읽는다 — 46 MB 를 통째로 뜨지 않는다."""
    return pd.read_csv(csv_path, usecols=_READ_COLUMNS)


def _baseline(frame: pd.DataFrame) -> pd.DataFrame:
    """정상 행(누출 0)을 짝짓는 열로 세운다. 유일하지 않으면 예외를 던진다."""
    normal = frame[frame["scenario_kind"] == "정상"]
    base = normal[[*PAIR_COLUMNS, *(s.column for s in SIGNALS)]].rename(
        columns={s.column: f"base_{s.column}" for s in SIGNALS}
    )
    if base.duplicated(subset=list(PAIR_COLUMNS)).any():
        raise ValueError("정상 행이 짝짓는 열 여덟으로 유일하지 않다")
    return base


def signal_deltas(frame: pd.DataFrame, leak_model: str) -> pd.DataFrame:
    """(행 · 신호) 짝마다 한 줄인 긴 표. 정상 대비 Δ 와 기대 부호를 담는다.

    순수 함수 — 전역 상태를 읽지 않는다.
    """
    rows = frame[
        (frame["leak_model"] == leak_model) & (frame["scenario_kind"] == "이상")
    ]
    merged = rows.merge(_baseline(frame), on=list(PAIR_COLUMNS), how="left")
    if len(merged) != len(rows):
        raise ValueError("짝짓기가 행 수를 바꿨다 — 정상 행이 유일하지 않다")
    missing = merged[f"base_{SIGNALS[0].column}"].isna().sum()
    if missing:
        raise ValueError(f"기준선을 못 찾은 행 {missing}건")

    level_column = LEVEL_COLUMN[leak_model]
    pieces: list[pd.DataFrame] = []
    for signal in SIGNALS:
        after = merged[signal.column]
        before = merged[f"base_{signal.column}"]
        delta = after - before
        piece = merged[
            [*PAIR_COLUMNS, *TOPOLOGY_COLUMNS, "anomaly_cdu_index", level_column]
        ].copy()
        piece["signal"] = signal.label
        piece["unit"] = signal.unit
        #: 이 행의 CDU 가 이상 기구를 진 쪽인가. 기대 부호는 진 쪽에서 나온 것이라
        #: 이웃 CDU 의 실패를 따로 세야 한다(기준 문서 2-4).
        piece["leak_cdu"] = merged["cdu_index"] == merged["anomaly_cdu_index"]
        piece["level"] = merged[level_column]
        piece["delta_abs"] = delta
        #: 기준 C 가 보는 값 — 유량 셋만 상대 % 다(세션 4 와 같은 정의).
        piece["delta_judged"] = delta / before * 100.0 if signal.unit == "%" else delta
        piece["expected_sign"] = _expected_sign(merged, signal, leak_model)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def _expected_sign(
    merged: pd.DataFrame, signal: Signal, leak_model: str
) -> pd.Series:  # type: ignore[type-arg]
    """기대 부호. 0(배치가 정한다)은 `pump_sees_supply_flow` 로 갈린다."""
    fixed = signal.massloss_sign if leak_model == LEAK_MODEL_MASSLOSS else (
        signal.blockage_sign
    )
    if fixed != 0:
        return pd.Series(fixed, index=merged.index, dtype="int64")
    sees_supply = merged["pump_sees_supply_flow"].astype("boolean")
    return sees_supply.map({True: -1, False: +1}).astype("int64")


def not_applicable(long: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
    """「해당 없음」 — 「샘」의 수력 응답이 **항등적으로 0** 인 배치의 수력 신호.

    기준 문서 2-5-B(세션 7.47 · 커밋 `9ca4f14`). **수치 임계를 쓰지 않는다** —
    퇴화 배치(`residual_return_share == 0.0` 이고 `pump_sees_supply_flow == True`)
    에서는 `massloss_flow_Lps` 가 수력식에서 통째로 떨어져 나간다(g=0 이라
    환수측 잔여저항 항이 0 · 펌프가 공급유량을 본다 · `massloss.py:175-193`).
    그 배치의 Δ 는 물성 평가 온도가 남긴 몫이지 「샘」의 수력 흔적이 아니다
    (세션 7.46 D6).

    「막힘」 행은 두 열이 빈 값이라 결코 걸리지 않는다 — 구조 자유도가 없다.
    """
    degenerate = (long["residual_return_share"] == 0.0) & long[
        "pump_sees_supply_flow"
    ].astype("boolean").fillna(False)
    return (degenerate & long["signal"].isin(HYDRAULIC_SIGNALS)).astype(bool)


def criterion_a(long: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
    """부호 일관성 — 기대 부호와 같으면 통과. 퇴화 배치의 수력 신호는 해당 없음."""
    skip = not_applicable(long)
    correct = (long["delta_abs"] > 0) == (long["expected_sign"] > 0)
    return pd.Series(
        [NA if s else (PASS if c else FAIL) for s, c in zip(skip, correct)],
        index=long.index,
    )


def criterion_c(long: pd.DataFrame, smallest: float) -> pd.Series:  # type: ignore[type-arg]
    """잡음 대비 — 가장 작은 누출 수준의 행만 본다."""
    rows = long[long["level"] == smallest]
    threshold = rows["unit"].map(NOISE_THRESHOLD)
    skip = not_applicable(rows)
    loud = rows["delta_judged"].abs() > threshold
    return pd.Series(
        [NA if s else (PASS if v else FAIL) for s, v in zip(skip, loud)],
        index=rows.index,
    )


def criterion_b(
    long: pd.DataFrame, group_columns: tuple[str, ...]
) -> pd.Series:  # type: ignore[type-arg]
    """수준 간 엄격 단조 — 판정 단위는 (무리 · 신호) 짝이다.

    배치는 무리 안에서 상수이므로(무리 키에 구조 자유도 셋이 들어간다)
    「해당 없음」도 무리 통째로 갈린다.
    """

    def verdict(group: pd.DataFrame) -> str:
        if bool(group["na"].all()):
            return NA
        values = group.sort_values("level")["delta_abs"].abs().to_numpy()
        rising = all(values[i] > values[i - 1] for i in range(1, len(values)))
        return PASS if rising else FAIL

    flagged = long.assign(na=not_applicable(long))
    keys = [*group_columns, "signal"]
    return flagged.groupby(keys, dropna=False)[["level", "delta_abs", "na"]].apply(
        verdict
    )


# ─────────────────────────────────────────────────────────────────────────────
# 표 만들기
# ─────────────────────────────────────────────────────────────────────────────
def _tally(labels: pd.Series, verdicts: pd.Series) -> pd.DataFrame:  # type: ignore[type-arg]
    """신호별 통과·실패·해당 없음 건수 표."""
    table = pd.crosstab(labels, verdicts)
    for column in (PASS, FAIL, NA):
        if column not in table:
            table[column] = 0
    table = table[[PASS, FAIL, NA]]
    table["합"] = table.sum(axis=1)
    return table


def _tally_lines(title: str, table: pd.DataFrame, total: int) -> list[str]:
    lines = [title, "-" * 62]
    lines.append(f"{'신호':<20}{PASS:>10}{FAIL:>10}{NA:>12}{'합':>10}")
    for label, row in table.iterrows():
        lines.append(
            f"{str(label):<20}{row[PASS]:>10,}{row[FAIL]:>10,}"
            f"{row[NA]:>12,}{row['합']:>10,}"
        )
    checked = int(table["합"].sum())
    lines.append("-" * 62)
    lines.append(
        f"  검산: 통과+실패+해당없음 = {checked:,} · 전수(짝 수) = {total:,} · "
        f"{'일치' if checked == total else '불일치'}"
    )
    return lines


def _split_tally_lines(
    title: str, long: pd.DataFrame, verdicts: pd.Series  # type: ignore[type-arg]
) -> list[str]:
    """이상 기구를 진 CDU 와 이웃 CDU 를 **갈라서** 센다 (기준 문서 2-4).

    기대 부호는 기구를 진 CDU 에서 나온 것이라, 이웃 CDU 의 실패는
    「기대가 이웃까지 미치지 않는다」는 뜻이고 모델 결함이 아니다.
    """
    rows = long.loc[verdicts.index]
    lines: list[str] = []
    for is_leak, note in (
        (True, "이상 기구를 진 CDU"),
        (False, "이웃 CDU — 기대 부호를 유도하지 않았다(판정을 결함으로 읽지 않는다)"),
    ):
        mask = rows["leak_cdu"] == is_leak
        if not mask.any():
            continue
        subset = rows[mask]
        lines += _tally_lines(
            f"{title} — {note}", _tally(subset["signal"], verdicts[mask]), len(subset)
        )
        lines.append("")
    return lines


def _fail_axis_lines(
    long: pd.DataFrame,
    verdicts: pd.Series,  # type: ignore[type-arg]
    axes: tuple[str, ...],
) -> list[str]:
    """실패가 어느 축으로 갈리는가. 실패가 없으면 한 줄로 끝낸다."""
    failed = long.loc[verdicts.index][verdicts == FAIL]
    if failed.empty:
        return ["  실패 0건 — 갈리는 축 없음"]
    lines = [f"  실패 {len(failed):,}건이 갈리는 축:"]
    for axis in axes:
        spread = failed.groupby(axis, dropna=False).size()
        if len(spread) == 1:
            lines.append(f"    · {axis}: 값 하나({spread.index[0]})에만 몰림")
            continue
        lines.append(f"    · {axis}: " + " · ".join(
            f"{value}={count:,}" for value, count in spread.items()
        ))
    return lines


def _na_axis_lines(long: pd.DataFrame, verdicts: pd.Series) -> list[str]:  # type: ignore[type-arg]
    """해당 없음이 어느 배치에 몰리는가 — 5.6 관측 ④ 와 맞는지 본다."""
    na_rows = long.loc[verdicts.index][verdicts == NA]
    if na_rows.empty:
        return ["  해당 없음 0건"]
    grouped = na_rows.groupby(
        ["residual_return_share", "pump_sees_supply_flow", "signal"], dropna=False
    ).size()
    lines = [f"  해당 없음 {len(na_rows):,}건의 배치별 분포:"]
    for (share, sees_supply, signal), count in grouped.items():
        lines.append(
            f"    · g={share} · 펌프={'공급' if sees_supply else '환수'} · "
            f"{signal}: {count:,}건"
        )
    return lines


def _leak_only(long: pd.DataFrame, verdicts: pd.Series) -> pd.Series:  # type: ignore[type-arg]
    """이상 기구를 진 CDU 의 판정만 남긴다."""
    return verdicts[long.loc[verdicts.index, "leak_cdu"]]


def _pass_rate_lines(
    long: pd.DataFrame, verdicts: pd.Series, axes: tuple[str, ...]
) -> list[str]:  # type: ignore[type-arg]
    """가정치 범위 양 끝에서 통과율이 갈리는가 (C4)."""
    judged = long.loc[verdicts.index].assign(verdict=verdicts)
    decided = judged[judged["verdict"] != NA]
    lines = []
    for axis in axes:
        parts = []
        for value, group in decided.groupby(axis, dropna=False):
            rate = (group["verdict"] == PASS).mean() * 100.0
            parts.append(f"{value}: {rate:.4f}% ({len(group):,}짝)")
        lines.append(f"    · {axis} — " + " · ".join(parts))
    return lines


def _span_lines(rows: pd.DataFrame) -> list[str]:
    """신호별 |Δ| 최소~최대 — **판정단위**(`delta_judged`)로 찍는다.

    미해결 #65(세션 7.47 에서 닫음): 이 표는 `delta_abs`(원단위 — 유량은 L/s)를
    찍으면서 `unit`(판정단위 — 유량은 %)을 라벨로 붙였다. 유량 셋만 어긋났다
    (양정 mAq · 온도 K 는 원단위가 곧 판정단위다). 판정이 보는 값과 같은
    `delta_judged` 를 찍는 쪽으로 맞춘다 — 라벨과 값이 이제 같은 단위다.
    """
    lines = []
    for label, group in rows.groupby("signal"):
        magnitude = group["delta_judged"].abs()
        lines.append(
            f"    {label:<18} {len(group):>6,}건 · "
            f"{magnitude.min():.3e} ~ {magnitude.max():.3e} [{group['unit'].iloc[0]}]"
        )
    return lines


def _degenerate_lines(
    massloss_long: pd.DataFrame, blockage_long: pd.DataFrame
) -> list[str]:
    """퇴화 배치의 Δ 가 얼마나 되는가 — 「해당 없음」으로 세는 몫의 크기.

    세션 5.6 관측 ④ 는 `g=0 · 펌프=공급` 배치에서 수력 다섯 양이 **정확히 0** 이라고
    했다(수력 한정 · 물성 37 ℃ 고정). 데이터셋의 「샘」 해는 열까지 물린 것이라
    그 배치에서도 Δ 가 0 이 아니다. **세션 7.46 이 그 갈래를 갈랐다** — 기준선
    경로 차의 몫은 0 이고 남는 것은 물성 평가 온도 하나다(D6). 세션 7.47 의
    「해당 없음」 규정이 이 절의 ㄱ 을 통째로 덮는다.
    """
    degenerate = massloss_long[
        (massloss_long["residual_return_share"] == 0.0)
        & massloss_long["pump_sees_supply_flow"].astype("boolean").fillna(False)
        & massloss_long["leak_cdu"]
    ]
    neighbour = blockage_long[~blockage_long["leak_cdu"]]
    return [
        "퇴화 배치의 Δ 크기 — 「해당 없음」으로 세는 몫 (판정단위 · 재기만 한다)",
        "-" * 62,
        "  ㄱ. 「샘」 · g=0 · 펌프=공급 · 이상 기구를 진 CDU:",
        *_span_lines(degenerate),
        "  ㄴ. 대조 — 「막힘」의 **이웃 CDU**(같은 solver 경로 · 2차측으로만 물린다):",
        *_span_lines(neighbour),
        "  · ㄱ 은 0 이 아니지만 **「샘」의 수력 응답이 아니다.** 세션 7.46 이 갈랐다:",
        "    기준선 경로 차의 몫은 배치 6 전수에서 0(g=0.5 의 1.147813e-14 % 는",
        "    부동소수 잡음)이고, 남는 채널은 **물성 평가 온도 하나**다(D6 — 온도만",
        "    옮긴 예측이 실제를 차 0.000e+00 로 덮는다). 단일 CDU 행에는 2차측",
        "    결합이 없다 — 1차측 물성 온도 고정점이 그 채널이다.",
        "  · 그래서 세션 7.47 이 ㄱ 을 「해당 없음」으로 규정했다(기준 문서 2-5-B).",
        "    **수치 임계로 가른 것이 아니라 배치로 갈랐다.**",
        "  · ㄴ 의 최대가 ㄱ 의 최대와 **한 자리 안**이다 — ㄴ 은 기준선과 같은 경로로",
        "    풀린 행이고, 두 크기가 같은 자리라는 관측은 그대로 둔다(ㄴ 은 판정 대상이",
        "    아니라 대조다).",
    ]


def _mismatch_lines(frame: pd.DataFrame) -> list[str]:
    """C6 — 공급 − 환수 유량 불일치. 구성상 항등식이지 물리 검증이 아니다."""
    delta = frame["total_flow_Lps"] - frame["return_flow_Lps"]
    lines = ["공급 − 환수 유량 불일치 Δ = total_flow_Lps − return_flow_Lps", "-" * 62]
    for model in (LEAK_MODEL_MASSLOSS, LEAK_MODEL_K_APPROX):
        mask = frame["leak_model"] == model
        values = delta[mask]
        nonzero = values.abs() > SIGN_ZERO_TOL
        lines.append(
            f"  {model:<10} {len(values):>7,}행 · Δ≠0 {int(nonzero.sum()):>7,}행 · "
            f"범위 {values.min():+.6e} ~ {values.max():+.6e} L/s"
        )
    lines.append(
        "  · K_approx 행 수는 「막힘」 960 + 정상 320 이다 — 둘 다 밀폐루프라 "
        "공급 = 환수다."
    )
    massloss = frame[frame["leak_model"] == LEAK_MODEL_MASSLOSS]
    identity = (
        massloss["total_flow_Lps"]
        - massloss["return_flow_Lps"]
        - massloss["massloss_flow_Lps"]
    ).abs()
    lines.append(
        f"  「샘」 Δ 와 massloss_flow_Lps 의 차 최대 {identity.max():.3e} L/s "
        "— **구성상 항등식이지 질량보존의 물리 검증이 아니다**(세션 5.6-B)"
    )
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# 리포트
# ─────────────────────────────────────────────────────────────────────────────
def format_report(frame: pd.DataFrame) -> str:
    """판정 결과 전문. 순수 함수 — 읽은 표 하나만 받는다."""
    lines: list[str] = [
        "=" * 78,
        "세션 7.45 — 「샘」(질량손실) 게이트 판정 (판정 기준 선기재 · 커밋 3001588)",
        "  · 「해당 없음」 규정만 세션 7.47 이 물리로 다시 세웠다(커밋 9ca4f14) —",
        "    수치 임계가 아니라 「샘의 수력 응답이 항등적으로 0 인 배치」로 가른다.",
        "    세션 7.45 가 낸 건수는 옛 규정(|Δ|≤1e-12)의 것이라 아래와 다르다.",
        "=" * 78,
        "※ " + ASSUMPTION_TAG,
        "※ 「통과」 = 모델 안에서 신호가 잡음 위에 있다. 실측 감지 가능성이 아니다.",
        "※ energy balance 는 미판정이다 — 「샘」 행은 누출 엔탈피 항 없이 닫히지",
        "   않는다(세션 5.7-D). 그 항을 넣는 것은 물리 모델 변경이라 범위 밖이다.",
        "",
    ]

    massloss_long = signal_deltas(frame, LEAK_MODEL_MASSLOSS)
    blockage_long = signal_deltas(frame, LEAK_MODEL_K_APPROX)
    massloss_rows = len(massloss_long) // len(SIGNALS)
    blockage_rows = len(blockage_long) // len(SIGNALS)
    lines += [
        f"판정 대상: 「샘」 {massloss_rows:,}행 × 신호 {len(SIGNALS)} = "
        f"{len(massloss_long):,}짝 · 「막힘」 {blockage_rows:,}행 = "
        f"{len(blockage_long):,}짝",
        "",
    ]

    axes = (
        "residual_return_share",
        "pump_sees_supply_flow",
        "share_uses_return_flow",
        "load_percent",
        "cdu_config",
        "cdu_index",
        "massloss_size_fraction",
    )

    verdict_a = criterion_a(massloss_long)
    lines += _split_tally_lines("기준 A — 부호 일관성 (「샘」)", massloss_long, verdict_a)
    lines += _fail_axis_lines(massloss_long, verdict_a, axes)
    lines += _na_axis_lines(massloss_long, verdict_a)
    lines.append("")

    group_columns = (*PAIR_COLUMNS, *TOPOLOGY_COLUMNS)
    verdict_b = criterion_b(massloss_long, group_columns)
    b_labels = verdict_b.index.get_level_values("signal").to_series(
        index=verdict_b.index
    )
    b_total = len(verdict_b)
    lines += _tally_lines(
        "기준 B — 수준 간 엄격 단조 (「샘」 · 크기 4수준 · 무리당 4행)",
        _tally(b_labels, verdict_b),
        b_total,
    )
    lines.append("")

    smallest = float(massloss_long["level"].min())
    verdict_c = criterion_c(massloss_long, smallest)
    lines += _split_tally_lines(
        f"기준 C — 잡음 대비 (「샘」 · 가장 작은 크기 {smallest:g})",
        massloss_long,
        verdict_c,
    )
    lines += _fail_axis_lines(massloss_long, verdict_c, axes)
    lines.append("")
    lines += _degenerate_lines(massloss_long, blockage_long)
    lines.append("")

    #: 5장 범위 축 여섯 — 데이터셋에 있는 것이 이 여섯뿐이고 전부 양 끝 2수준이다.
    range_axes = (
        "load_percent",
        "pump_head_rated_mAq",
        "branch_dp_rated_mAq",
        "valve_dp_rated_mAq",
        "ntu",
        "T_secondary_supply_C",
    )
    lines += [
        "C4 — 가정치 범위 양 끝에서 통과율이 갈리는가",
        "     (이상 기구를 진 CDU 만 · 해당 없음은 분모에서 뺀다)",
        "-" * 62,
        "  기준 A:",
    ]
    leak_a = _leak_only(massloss_long, verdict_a)
    leak_c = _leak_only(massloss_long, verdict_c)
    lines += _pass_rate_lines(massloss_long, leak_a, range_axes)
    lines += ["  기준 C:"]
    lines += _pass_rate_lines(massloss_long, leak_c, range_axes)
    lines.append("")

    verdict_ka = criterion_a(blockage_long)
    verdict_kb = criterion_b(blockage_long, PAIR_COLUMNS)
    smallest_k = float(blockage_long["level"].min())
    verdict_kc = criterion_c(blockage_long, smallest_k)
    lines += _split_tally_lines(
        "C5 — 같은 기준으로 돌린 「막힘」 (대조 · 판정이 아니다) · 기준 A",
        blockage_long,
        verdict_ka,
    )
    lines += _fail_axis_lines(
        blockage_long, verdict_ka, ("load_percent", "cdu_config", "cdu_index")
    )
    lines.append("")
    lines += _tally_lines(
        "C5 — 「막힘」 기준 B (수준 3)",
        _tally(
            verdict_kb.index.get_level_values("signal").to_series(
                index=verdict_kb.index
            ),
            verdict_kb,
        ),
        len(verdict_kb),
    )
    lines.append("")
    lines += _split_tally_lines(
        f"C5 — 「막힘」 기준 C (가장 작은 수준 {smallest_k:g}%)",
        blockage_long,
        verdict_kc,
    )
    lines += [
        "  세션 4 는 단일 CDU 32조합 × 3수준 = 96건을 직접 풀어 판정했다"
        "(PROCEED.md:20).",
        "  여기는 부하 2 · 구성 3 · CDU 대수까지 곱해진 CSV 행이라 표본이 다르다 —",
        "  어긋남 자체를 결함으로 읽지 않는다.",
        "",
    ]

    lines += _summary_lines(
        massloss_long, blockage_long, verdict_a, verdict_b, verdict_c
    )
    lines.append("")
    lines += _mismatch_lines(frame)
    return "\n".join(lines)


def _leak_side(
    verdicts: pd.Series, long: pd.DataFrame  # type: ignore[type-arg]
) -> pd.Series:  # type: ignore[type-arg]
    """이상 기구를 진 CDU 의 판정만 남긴다 — 기준 B 는 무리 키로 가른다."""
    if isinstance(verdicts.index, pd.MultiIndex):
        level = verdicts.index.get_level_values("cdu_index")
        return verdicts[level == LEAK_CDU_INDEX]
    return _leak_only(long, verdicts)


def _summary_lines(
    massloss_long: pd.DataFrame,
    blockage_long: pd.DataFrame,
    verdict_a: pd.Series,  # type: ignore[type-arg]
    verdict_b: pd.Series,  # type: ignore[type-arg]
    verdict_c: pd.Series,  # type: ignore[type-arg]
) -> list[str]:
    """「샘」·「막힘」 통과율·해당 없음 비율을 나란히 놓는다 (C5-나).

    **이상 기구를 진 CDU 만** 센다 — 이웃 CDU 는 기대 부호를 유도하지 않았으므로
    통과율에 섞으면 뜻이 없다. 이웃 쪽 건수는 위 표들에 그대로 있다.
    """
    lines = [
        "「샘」 대 「막힘」 — 통과율·해당 없음 비율 (이상 기구를 진 CDU 만)",
        "-" * 62,
    ]
    lines.append(f"{'기준':<8}{'기구':<8}{'통과':>10}{'실패':>10}{'해당없음':>12}{'통과율*':>12}")
    pairs = (
        ("A", verdict_a, criterion_a(blockage_long)),
        ("B", verdict_b, criterion_b(blockage_long, PAIR_COLUMNS)),
        (
            "C",
            verdict_c,
            criterion_c(blockage_long, float(blockage_long["level"].min())),
        ),
    )
    for name, massloss_verdict, blockage_verdict in pairs:
        sides = (
            ("샘", _leak_side(massloss_verdict, massloss_long)),
            ("막힘", _leak_side(blockage_verdict, blockage_long)),
        )
        for mechanism, verdicts in sides:
            counts = verdicts.value_counts()
            passed, failed = int(counts.get(PASS, 0)), int(counts.get(FAIL, 0))
            na = int(counts.get(NA, 0))
            decided = passed + failed
            rate = passed / decided * 100.0 if decided else float("nan")
            lines.append(
                f"{name:<8}{mechanism:<8}{passed:>10,}{failed:>10,}{na:>12,}"
                f"{rate:>11.4f}%"
            )
    lines.append("  * 통과율 = 통과 / (통과+실패) — 해당 없음은 분모에서 뺀다")
    return lines


def main() -> int:
    import sys

    if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔 기본 인코딩 대비
        sys.stdout.reconfigure(encoding="utf-8")
    frame = read_dataset(DEFAULT_OUTPUT_DIR / "cdu_dataset.csv")
    print(format_report(frame))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
