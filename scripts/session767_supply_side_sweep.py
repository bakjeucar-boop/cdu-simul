"""세션 7.67 — 공급측 「샘」의 부호를 **전수로** 센다 (계측 스크립트).

**이 스크립트는 모델에 아무것도 넣지 않는다.** `src/` 를 고치지 않고, 자유도를
만들지 않으며, 미해결 #36 도 파일럿 종료 판정 5-b 도 닫지 않는다. 부호를 세는
것이 전부다.

세션 7.65 가 **한 케이스**(전수의 1/768)에서 얻은 「공급측 「샘」은 「막힘」과
같은 쪽」이 전수에서도 성립하는지, 어디서 갈리는지를 본다.

**방법은 7.65 것을 그대로 이어 쓴다.** 공급측 주입의 수력은 재매개화(−Q ·
g→1−g · 펌프 다리 반전)로, 열은 7.65 스크립트가 적어 둔 국소 변형
(`solve_supply_side_massloss_steady`)으로 푼다 — 이 판은 그 파일을 import 해서
쓰고 **열 규약 둘을 새로 정하지 않는다**:

  ⓐ 유출은 T_supply 에서 나간다 → 공급 기준 초과 엔탈피 0
  ⓑ 「펌프=공급」은 펌프를 **누출점 상류**에 둔다

**재매개화가 전 배치에서 항등인지 이 판이 직접 확인한다**(C1). 7.65 는 한
배치(g=0.5·펌프=공급)에서만 썼다. `_supply_side_equation_residual_mAq` 가
공급측 수력식을 `solve_massloss` 와 **독립하게** 다시 적고, 재매개화가 낸 해를
그 식에 넣어 잔차를 잰다. 잔차가 압력평형 수렴 잔차 수준이면 항등이다.

**새 숫자를 만들지 않는다.** 「샘」 크기는 5-1 「「샘」(질량손실) 크기 수준」의
역산 규칙(`massloss_flow_bound_Lps` × `SWEEP_FRACTIONS`)을 그대로 쓴다.
부호를 0 으로 읽는 임계는 7.65 가 쓴 1e-9 를 그대로 이어 쓰고, `_SIGN_ZERO_TOL`
(1e-12)과의 사이에 든 건수를 따로 세어 그 선택이 무엇을 가리는지 드러낸다.

**모든 수치는 가정값 기반이며 실측이 아니다.**
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

# 행은 실수·문자열·불리언을 섞어 담는 CSV 한 줄이라 값 타입을 좁히지 않는다.
from typing import Any

import numpy as np
from session765_supply_side_massloss import (
    _mirrored_topology,
    solve_supply_side_massloss_steady,
)

from cdu_simul.assumptions import ASSUMPTION_TAG, LEAK, SCENARIO
from cdu_simul.hydraulics import (
    apply_leak_to_rack,
    branch_dp_mAq,
    bulk_mean_temperature_C,
    pump_head_mAq,
    residual_dp_mAq,
    residual_resistance_coeff_mAq_per_Lps2,
    valve_dp_mAq,
)
from cdu_simul.massloss import (
    SWEEP_FRACTIONS,
    MassLossTopology,
    _other_rack_index,
    k_approx_results,
    massloss_flow_bound_Lps,
    massloss_topologies,
    solve_massloss,
)
from cdu_simul.massloss_thermal import MassLossThermal, solve_massloss_steady
from cdu_simul.model import (
    CduCase,
    default_cdu_cases,
    energy_balance_residual_percent,
    solve_cdu_steady_state,
)

#: 부호를 0 으로 읽는 임계 — 세션 7.65 가 쓴 값을 그대로 이어 쓴다(새 값이 아니다).
_SIGN_TOL: float = 1.0e-9
#: 세션 5.6 이 정하고 7.47 이 인용한 절대 임계. 위 임계가 무엇을 가리는지 보려고
#: 둘 사이에 든 건수를 따로 센다.
_ABS_ZERO_TOL: float = 1.0e-12
#: 6장 ① 게이트 임계 [%].
_BALANCE_LIMIT_PERCENT: float = 0.1

#: 「샘」의 수력 응답이 항등적으로 0 이 되는 배치 (7.47 2-5-B).
#: 환수측 주입은 (g=0 · 펌프=공급), 공급측 주입은 그 거울인 (g=1 · 펌프=환수)다.
_RETURN_DEGENERATE: tuple[float, bool] = (0.0, True)
_SUPPLY_DEGENERATE: tuple[float, bool] = (1.0, False)

_OUTPUT_DIR: Path = Path(__file__).resolve().parents[1] / "results"


def _is_degenerate(topology: MassLossTopology, supply_side: bool) -> bool:
    key = (topology.residual_return_share, topology.pump_sees_supply_flow)
    return key == (_SUPPLY_DEGENERATE if supply_side else _RETURN_DEGENERATE)


# ─────────────────────────────────────────────────────────────────────────────
# C1 — 재매개화가 항등인지 독립하게 확인한다
# ─────────────────────────────────────────────────────────────────────────────
def _supply_side_equation_residual_mAq(
    case: CduCase,
    topology: MassLossTopology,
    massloss_flow_Lps: float,
    T_property_C: float,
    rack_flows_Lps: tuple[float, ...],
) -> float:
    """공급측 주입 수력식의 최대 절대잔차 [mAq] — `solve_massloss` 를 쓰지 않는다.

    누출이 랙 **상류**에 있으면 다리 둘의 유량이 이렇게 갈린다::

        상류 다리(펌프→누출점) : Q_big   = ΣQ_i + Q
        하류 다리(누출점→랙→환수→HX) : ΣQ_i

    `g` 는 원래 정의(잔여저항 중 **누출점 하류** 몫) 그대로다 — 하류가 ΣQ_i
    다리이므로 g·C 가 그쪽에, (1−g)·C 가 상류 다리에 붙는다. 펌프는 7.65 규약
    ⓑ 대로 「펌프=공급」이면 누출점 **상류**(=Q_big)에 둔다.
    """
    hydraulic = case.hydraulic
    residual_coeff = residual_resistance_coeff_mAq_per_Lps2(hydraulic)
    g = topology.residual_return_share
    Q_racks_Lps = float(sum(rack_flows_Lps))
    Q_big_Lps = Q_racks_Lps + massloss_flow_Lps
    Q_pump_Lps = Q_big_Lps if topology.pump_sees_supply_flow else Q_racks_Lps
    available_mAq = (
        pump_head_mAq(Q_pump_Lps, hydraulic.pump)
        - residual_dp_mAq(Q_big_Lps, (1.0 - g) * residual_coeff, T_property_C)
        - residual_dp_mAq(Q_racks_Lps, g * residual_coeff, T_property_C)
    )
    residuals = [
        available_mAq
        - branch_dp_mAq(Q_i, K_i, T_property_C)
        - valve_dp_mAq(
            Q_i, hydraulic.valve_Kv_max_m3h, hydraulic.opening_fraction, T_property_C
        )
        for Q_i, K_i in zip(rack_flows_Lps, hydraulic.rack_branch_K, strict=True)
    ]
    return float(np.max(np.abs(residuals)))


# ─────────────────────────────────────────────────────────────────────────────
# 관측값
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Observation:
    """한 판이 내는 양들. 수력 다섯(세션 5.6 관측 ④) + 온도 둘 + 잔차."""

    total_flow_Lps: float
    return_flow_Lps: float
    pump_flow_Lps: float
    pump_head_mAq: float
    injection_rack_flow_Lps: float
    other_rack_flow_Lps: float
    injection_rack_outlet_C: float
    T_return_C: float
    balance_residual_percent: float
    solver_ok: bool


#: 부호를 세는 항목 — 프롬프트 C2 가 낼 것으로 정한 넷.
_SIGN_FIELDS: tuple[str, ...] = (
    "total_flow_Lps",
    "injection_rack_flow_Lps",
    "injection_rack_outlet_C",
    "T_return_C",
)
#: C4 가 보는 수력 다섯 + 온도 둘.
_HYDRAULIC_FIELDS: tuple[str, ...] = (
    "total_flow_Lps",
    "return_flow_Lps",
    "pump_flow_Lps",
    "pump_head_mAq",
    "injection_rack_flow_Lps",
    "other_rack_flow_Lps",
)
_THERMAL_FIELDS: tuple[str, ...] = ("injection_rack_outlet_C", "T_return_C")


def _observe_massloss(
    result: MassLossThermal, supply_side: bool, massloss_flow_Lps: float
) -> Observation:
    other = _other_rack_index(LEAK.injection_rack_index)
    total_Lps = result.supply_flow_Lps
    if supply_side:
        # 상류 다리가 큰 유량이다. 펌프는 7.65 규약 ⓑ 대로 「공급」이면 상류에 있다.
        pump_flow_Lps = (
            total_Lps + massloss_flow_Lps
            if result.topology.pump_sees_supply_flow
            else total_Lps
        )
    else:
        pump_flow_Lps = (
            total_Lps
            if result.topology.pump_sees_supply_flow
            else total_Lps - massloss_flow_Lps
        )
    return Observation(
        total_flow_Lps=total_Lps,
        return_flow_Lps=result.return_flow_Lps,
        pump_flow_Lps=pump_flow_Lps,
        pump_head_mAq=result.pump_head_mAq,
        injection_rack_flow_Lps=result.rack_flows_Lps[LEAK.injection_rack_index],
        other_rack_flow_Lps=result.rack_flows_Lps[other],
        injection_rack_outlet_C=result.injection_rack_outlet_temp_C,
        T_return_C=result.T_return_C,
        balance_residual_percent=result.energy_balance_residual_percent,
        solver_ok=result.solver_converged,
    )


def _observe_blockage(case: CduCase, k_multiplier: float) -> Observation:
    """「막힘」 한 판 — 기존 모듈을 그대로 쓴다(본문 무수정)."""
    blocked = CduCase(
        hydraulic=apply_leak_to_rack(case.hydraulic, k_multiplier),
        T_secondary_supply_C=case.T_secondary_supply_C,
        ntu=case.ntu,
        load_percent=case.load_percent,
    )
    solved = solve_cdu_steady_state(blocked)
    other = _other_rack_index(LEAK.injection_rack_index)
    total_Lps = float(sum(solved.flow.rack_flows_Lps))
    return Observation(
        total_flow_Lps=total_Lps,
        return_flow_Lps=total_Lps,  # 「막힘」은 질량이 보존된다
        pump_flow_Lps=total_Lps,
        pump_head_mAq=solved.flow.pump_head_mAq,
        injection_rack_flow_Lps=solved.flow.rack_flows_Lps[LEAK.injection_rack_index],
        other_rack_flow_Lps=solved.flow.rack_flows_Lps[other],
        injection_rack_outlet_C=solved.thermal.rack_return_temps_C[
            LEAK.injection_rack_index
        ],
        T_return_C=solved.thermal.T_return_C,
        balance_residual_percent=energy_balance_residual_percent(solved.thermal),
        solver_ok=solved.solver_converged,
    )


def _sign(value: float, tol: float = _SIGN_TOL) -> int:
    if value > tol:
        return 1
    if value < -tol:
        return -1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 전수
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SweepRow:
    case_label: str
    residual_return_share: float
    pump_sees_supply_flow: bool
    size_fraction: float
    massloss_flow_Lps: float
    reparam_residual_mAq: float
    solver_failures: int


def run_sweep() -> tuple[list[dict[str, Any]], list[str]]:
    """768 축 전수를 돌고 한 축당 한 행을 낸다. 실패는 세되 값을 쓰지 않는다."""
    cases = default_cdu_cases()
    topologies = massloss_topologies()
    fractions = tuple(f for f in SWEEP_FRACTIONS if f != 0.0)
    T_rated_C = bulk_mean_temperature_C(
        SCENARIO.T_primary_supply_C, SCENARIO.T_primary_return_C
    )

    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for case in cases:
        bound_Lps = massloss_flow_bound_Lps(k_approx_results(case.hydraulic, T_rated_C))
        blockage_base = _observe_blockage(case, 1.0)
        blockage_levels = {
            label: _observe_blockage(case, mult)
            for label, mult in LEAK.k_multiplier_levels
            if mult != 1.0
        }
        for topology in topologies:
            normal = _observe_massloss(
                solve_massloss_steady(case, 0.0, topology), False, 0.0
            )
            for fraction in fractions:
                massloss_Lps = fraction * bound_Lps
                ret_result = solve_massloss_steady(case, massloss_Lps, topology)
                sup_result = solve_supply_side_massloss_steady(
                    case, massloss_Lps, topology
                )
                ret = _observe_massloss(ret_result, False, massloss_Lps)
                sup = _observe_massloss(sup_result, True, massloss_Lps)
                reparam_mAq = _supply_side_equation_residual_mAq(
                    case,
                    topology,
                    massloss_Lps,
                    sup_result.property_eval_T_C,
                    sup_result.rack_flows_Lps,
                )
                # C1 의 잣대 — 재매개화가 쓴 `solve_massloss` 자신의 수렴 잔차.
                # 재매개화 잔차가 이것과 같은 자리면 「식이 어긋난 것」이 아니라
                # 「fsolve 가 거기까지만 수렴한 것」이다.
                code_residual_mAq = solve_massloss(
                    case.hydraulic,
                    -massloss_Lps,
                    _mirrored_topology(topology),
                    sup_result.property_eval_T_C,
                ).max_abs_equation_residual_mAq
                row: dict[str, Any] = {
                    "case_label": case.label,
                    "residual_return_share": topology.residual_return_share,
                    "pump_sees_supply_flow": topology.pump_sees_supply_flow,
                    "topology_label": topology.label,
                    "size_fraction": fraction,
                    "massloss_flow_Lps": massloss_Lps,
                    "reparam_residual_mAq": reparam_mAq,
                    "code_solver_residual_mAq": code_residual_mAq,
                    "assumption_tag": ASSUMPTION_TAG,
                }
                for prefix, obs in (
                    ("normal", normal),
                    ("ret", ret),
                    ("sup", sup),
                    ("blk_base", blockage_base),
                ):
                    for key, value in asdict(obs).items():
                        row[f"{prefix}_{key}"] = value
                for label, obs in blockage_levels.items():
                    for key, value in asdict(obs).items():
                        row[f"blk[{label}]_{key}"] = value
                # Δ — 「샘」 둘은 정상(Q=0) 대비, 「막힘」은 K×1.0 대비
                for field in _HYDRAULIC_FIELDS + _THERMAL_FIELDS + (
                    "balance_residual_percent",
                ):
                    row[f"d_ret_{field}"] = getattr(ret, field) - getattr(normal, field)
                    row[f"d_sup_{field}"] = getattr(sup, field) - getattr(normal, field)
                    for label, obs in blockage_levels.items():
                        row[f"d_blk[{label}]_{field}"] = getattr(obs, field) - getattr(
                            blockage_base, field
                        )
                observations = (
                    normal,
                    ret,
                    sup,
                    blockage_base,
                    *blockage_levels.values(),
                )
                row["solver_failures"] = sum(
                    0 if obs.solver_ok else 1 for obs in observations
                )
                rows.append(row)
    if not rows:
        notes.append("행이 하나도 없다 — 축 구성이 잘못됐다.")
    return rows, notes


# ─────────────────────────────────────────────────────────────────────────────
# 세기
# ─────────────────────────────────────────────────────────────────────────────
def _blockage_reference_label() -> str:
    """부호 비교의 「막힘」 기준 — 가장 큰 수준(K+50%)을 쓴다(7.65 와 같다)."""
    return LEAK.k_multiplier_levels[-1][0]


def _row_topology(row: dict[str, Any]) -> MassLossTopology:
    return MassLossTopology(
        label=str(row["topology_label"]),
        residual_return_share=float(row["residual_return_share"]),
        pump_sees_supply_flow=bool(row["pump_sees_supply_flow"]),
    )


def _bucket(row: dict[str, Any], side: str, field: str, ref: str) -> str:
    """(행 · 신호) 짝 하나를 가른다.

    「해당 없음」은 **크기가 아니라 배치**로 가른다 [7.47 2-5-B] — 그 주입점의
    퇴화 배치이고 신호가 수력 넷/다섯 중 하나면 「해당 없음」이다. 그러고 남은
    것만 부호로 가른다.
    """
    if field in _HYDRAULIC_FIELDS and _is_degenerate(
        _row_topology(row), supply_side=(side == "sup")
    ):
        return "해당없음"
    s = _sign(float(row[f"d_{side}_{field}"]))
    s_blk = _sign(float(row[f"d_blk[{ref}]_{field}"]))
    if s_blk == 0:
        return "막힘0"
    if s == 0:
        return "0"
    return "같음" if s == s_blk else "반대"


def count_signs(rows: list[dict[str, Any]]) -> str:
    ref = _blockage_reference_label()
    lines: list[str] = []
    lines.append("── C3. 부호 건수 " + "─" * 60)
    lines.append(
        f"기준: 「막힘」 {ref} 의 Δ 부호. 임계 |Δ| ≤ {_SIGN_TOL:g} 를 0 으로 읽는다."
    )
    lines.append(
        "「해당 없음」은 크기가 아니라 배치로 가른다 [7.47 2-5-B] — "
        "그 주입점의 퇴화 배치 × 수력 신호."
    )
    lines.append("")
    for side, side_name in (("sup", "공급측 「샘」"), ("ret", "환수측 「샘」")):
        lines.append(f"[{side_name}]")
        lines.append(
            f"{'항목':<24}{'같은 쪽':>9}{'반대 쪽':>9}{'0':>7}"
            f"{'해당없음':>10}{'막힘0':>8}"
        )
        for field in _SIGN_FIELDS:
            tally: Counter[str] = Counter(
                _bucket(row, side, field, ref) for row in rows
            )
            lines.append(
                f"{field:<24}{tally['같음']:>9}{tally['반대']:>9}{tally['0']:>7}"
                f"{tally['해당없음']:>10}{tally['막힘0']:>8}"
            )
        lines.append("")
    return "\n".join(lines)


def count_signs_by_topology(rows: list[dict[str, Any]], side: str) -> str:
    """배치별로 갈라 센다 — 갈리는 자리가 있으면 여기서 보인다."""
    ref = _blockage_reference_label()
    name = "공급측" if side == "sup" else "환수측"
    lines = [f"── C3-나. 배치별 ({name} 「샘」) " + "─" * 40]
    lines.append(
        f"{'배치':<20}{'항목':<24}{'같은 쪽':>9}{'반대':>7}{'0':>7}"
        f"{'해당없음':>10}{'   Δ 범위':>26}"
    )
    for label in sorted({str(row["topology_label"]) for row in rows}):
        subset = [row for row in rows if row["topology_label"] == label]
        for field in _SIGN_FIELDS:
            tally: Counter[str] = Counter(
                _bucket(row, side, field, ref) for row in subset
            )
            deltas = [float(row[f"d_{side}_{field}"]) for row in subset]
            lines.append(
                f"{label:<20}{field:<24}{tally['같음']:>9}"
                f"{tally['반대']:>7}{tally['0']:>7}{tally['해당없음']:>10}"
                f"  [{min(deltas):+.3e}, {max(deltas):+.3e}]"
            )
    return "\n".join(lines)


def _degenerate_subset(
    rows: list[dict[str, Any]], key: tuple[float, bool]
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if (
            float(row["residual_return_share"]) == key[0]
            and bool(row["pump_sees_supply_flow"]) == key[1]
        )
    ]


def _verdict(value: float) -> str:
    if value == 0.0:
        return "정확히 0"
    return "≤1e-12" if value <= _ABS_ZERO_TOL else "≠0"


def degenerate_report(rows: list[dict[str, Any]]) -> str:
    """C4 — 퇴화 배치에서 수력 다섯·온도 둘이 얼마인가. 주입점 둘을 나란히 본다."""
    sup_rows = _degenerate_subset(rows, _SUPPLY_DEGENERATE)
    ret_rows = _degenerate_subset(rows, _RETURN_DEGENERATE)
    lines = ["── C4. 퇴화 배치에서의 max |Δ| " + "─" * 46]
    lines.append(
        f"공급측 퇴화 = g=1.0/펌프=환수 ({len(sup_rows)} 행) · "
        f"환수측 퇴화 = g=0.0/펌프=공급 ({len(ret_rows)} 행) · 각 32 조합 × 4 크기"
    )
    lines.append(
        f"{'항목':<26}{'공급측 max |Δ|':>18}{'':>10}"
        f"{'환수측 max |Δ|':>18}{'':>10}"
    )
    for field in _HYDRAULIC_FIELDS + _THERMAL_FIELDS:
        d_sup = max(abs(float(row[f"d_sup_{field}"])) for row in sup_rows)
        d_ret = max(abs(float(row[f"d_ret_{field}"])) for row in ret_rows)
        lines.append(
            f"{field:<26}{d_sup:>18.6e}{_verdict(d_sup):>10}"
            f"{d_ret:>18.6e}{_verdict(d_ret):>10}"
        )
    return "\n".join(lines)


def convention_report(rows: list[dict[str, Any]]) -> str:
    """C5 — 열 규약 ⓐ 가 전수에서 6장 ① 임계 안에 남는지."""
    lines = ["── C5. 열 규약 ⓐ (유출이 T_supply 에서 나간다) " + "─" * 30]
    for prefix, name in (("sup", "공급측"), ("ret", "환수측"), ("normal", "정상")):
        values = [
            abs(float(row[f"{prefix}_balance_residual_percent"]))
            for row in rows
        ]
        over = sum(1 for v in values if v > _BALANCE_LIMIT_PERCENT)
        lines.append(
            f"{name:<8} |잔차| 최대 {max(values):.6f} %  ·  "
            f"임계 {_BALANCE_LIMIT_PERCENT} % 초과 {over} 건 / {len(values)} 건"
        )
    return "\n".join(lines)


def gray_band_report(rows: list[dict[str, Any]]) -> str:
    """임계 선택(1e-9)이 무엇을 가리는가 — 1e-12 와의 사이에 든 건수."""
    lines = ["── 임계 사이대 (1e-12 < |Δ| ≤ 1e-9) 건수 " + "─" * 34]
    for side, name in (("sup", "공급측"), ("ret", "환수측")):
        counts = []
        for field in _SIGN_FIELDS:
            n = sum(
                1
                for row in rows
                if _ABS_ZERO_TOL
                < abs(float(row[f"d_{side}_{field}"]))
                <= _SIGN_TOL
            )
            counts.append(f"{field}={n}")
        lines.append(f"{name}: " + " · ".join(counts))
    return "\n".join(lines)


def reparam_report(rows: list[dict[str, Any]]) -> str:
    lines = ["── C1. 재매개화 항등 확인 (독립 재기술 식의 잔차) " + "─" * 25]
    lines.append(
        "잣대: 같은 해에 대한 `solve_massloss` 자신의 수렴 잔차. 두 값이 같은 자리면"
    )
    lines.append("      식이 어긋난 것이 아니라 fsolve 가 거기까지 수렴한 것이다.")
    lines.append(f"{'배치':<20}{'재기술 max':>16}{'solver max':>16}  [mAq]")
    by_topology: dict[str, tuple[float, float]] = {}
    for row in rows:
        label = str(row["topology_label"])
        prev = by_topology.get(label, (0.0, 0.0))
        by_topology[label] = (
            max(prev[0], float(row["reparam_residual_mAq"])),
            max(prev[1], float(row["code_solver_residual_mAq"])),
        )
    for label in sorted(by_topology):
        own, code = by_topology[label]
        lines.append(f"{label:<20}{own:>16.6e}{code:>16.6e}")
    lines.append(f"전 {len(rows)} 행 · 재기술 max = "
                 f"{max(v[0] for v in by_topology.values()):.6e} mAq")
    return "\n".join(lines)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows, notes = run_sweep()
    csv_path = _OUTPUT_DIR / "session767_supply_side_sweep.csv"
    write_csv(rows, csv_path)

    failures = sum(int(row["solver_failures"]) for row in rows)
    report = "\n\n".join(
        [
            "세션 7.67 — 공급측 「샘」 부호 전수 계측",
            "※ " + ASSUMPTION_TAG,
            "※ 부호를 세는 판이다. 모델에 넣지 않았고 #36 · 판정 5-b 를 닫지 않는다.",
            f"축: 조합 {len({str(r['case_label']) for r in rows})}"
            f" × 배치 {len({str(r['topology_label']) for r in rows})}"
            f" × 크기 {len({float(r['size_fraction']) for r in rows})}"
            f" = {len(rows)} 행",
            f"solver 실패: {failures} 건 (절대 규칙 5 — 실패가 있으면 값을 쓰지 않는다)",
            reparam_report(rows),
            count_signs(rows),
            count_signs_by_topology(rows, "sup"),
            count_signs_by_topology(rows, "ret"),
            degenerate_report(rows),
            convention_report(rows),
            gray_band_report(rows),
            f"결과 파일: {csv_path}",
        ]
        + notes
    )
    (_OUTPUT_DIR / "session767_summary.txt").write_text(report, encoding="utf-8")
    print(report)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
