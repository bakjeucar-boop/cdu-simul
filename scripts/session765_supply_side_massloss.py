"""세션 7.65 — 공급측 「샘」의 부호를 **한 케이스에서** 본다 (계측 스크립트).

**이 스크립트는 답을 내지 않는다.** 한 케이스 · 한 배치 · 한 크기의 부호만 낸다.
전수 판정이 아니고, 미해결 #36 이나 파일럿 종료 판정 5-b 를 닫지 않는다.

**src/ 를 고치지 않는다.** 공급측 주입은 두 가지로 만든다.

⑴ 수력 — **재매개화로 기존 함수를 그대로 부른다.** `massloss.solve_massloss` 의
   식은 「누출점 상류 다리가 큰 유량, 하류 다리가 작은 유량」이라는 구조뿐이라
   주입점을 옮기는 것이 인자 치환으로 정확히 표현된다::

       환수측 주입:  Q_상류 = ΣQ_i        , Q_하류 = ΣQ_i − Q
       공급측 주입:  Q_상류 = ΣQ_i + Q    , Q_하류 = ΣQ_i

   `massloss_flow_Lps = −Q` 를 넣으면 코드의 `Q_return = ΣQ_i − (−Q) = ΣQ_i + Q`
   가 되어 **큰 유량이 반대 다리로 옮겨간다**. 저항 배분도 같이 뒤집어야 하므로
   `residual_return_share` 를 `1 − g` 로, 펌프가 붙는 다리도 반대로 준다
   (`_mirrored_topology`). 근사가 아니라 **같은 식의 항등 치환**이다.

⑵ 열 — **재매개화가 안 된다. 스크립트 안에 국소 변형을 둔다**(`_supply_side_*`).
   `massloss_thermal._steady_at_property_temperature` 는 「환수유량 = 공급유량 − Q」
   와 「유출 엔탈피는 T_return 에서 나간다」를 식에 박고 있는데, 공급측 주입이면
   둘 다 달라진다: 환수유량 = 랙 통과유량이고, 유출은 T_supply 에서 나가 공급
   기준 초과 엔탈피가 0 이다. 인자로는 그 둘을 갈라 줄 수 없어 닫힌 형태를
   이 파일에 다시 적었다 — 원본과 다른 것은 그 두 자리뿐이다.

**새 숫자를 만들지 않는다.** 「샘」 크기는 5-1 「「샘」(질량손실) 크기 수준」의
역산 규칙(`massloss_flow_bound_Lps` × `SWEEP_FRACTIONS` 상한)을 그대로 쓴다.
「공급측 주입에서 펌프는 누출점 상류에 둔다」는 **위치 선택**이지 값이 아니다 —
환수측 배치와 거울로 맞춘 것이고, 그렇게 두어야 세 판이 같은 배치 위에서 비교된다.

**모든 수치는 가정값 기반이며 실측이 아니다.**
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve

from cdu_simul.assumptions import ASSUMPTION_TAG, HEAT_EXCHANGER, LEAK, SCENARIO
from cdu_simul.fluid import coolant_cp_Jkg_K, coolant_density_kgm3
from cdu_simul.hydraulics import (
    PumpHydraulicPower,
    apply_leak_to_rack,
    bulk_mean_temperature_C,
    pump_hydraulic_power_W,
)
from cdu_simul.massloss import (
    SWEEP_FRACTIONS,
    MassLossResult,
    MassLossTopology,
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
    hx_capacity_terms,
    property_temperature_from_state,
    solve_cdu_steady_state,
)

_M3_PER_LITRE: float = 1.0e-3
_W_PER_KW: float = 1.0e3

#: 이 판이 고른 배치 — `massloss_topologies()` 의 「g=0.5/펌프=공급」.
#: 까닭: 환수측·공급측 **양쪽에서 퇴화하지 않는** 유일한 g 다. 퇴화 배치는
#: 환수측이 (g=0·펌프=공급), 공급측이 그 거울인 (g=1·펌프=환수)라 g 양 끝을
#: 고르면 세 판 중 하나가 수력을 잃어 부호가 안 보인다.
_TOPOLOGY_LABEL: str = "g=0.5/펌프=공급"


def _chosen_topology() -> MassLossTopology:
    for topology in massloss_topologies():
        if topology.label == _TOPOLOGY_LABEL:
            return topology
    raise RuntimeError(f"배치를 찾지 못했다: {_TOPOLOGY_LABEL}")


def _mirrored_topology(topology: MassLossTopology) -> MassLossTopology:
    """공급측 주입의 재매개화 — 저항 배분과 펌프 다리를 함께 뒤집는다."""
    return MassLossTopology(
        label=f"공급측[{topology.label}]",
        residual_return_share=1.0 - topology.residual_return_share,
        pump_sees_supply_flow=not topology.pump_sees_supply_flow,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 공급측 「샘」 — 국소 변형 (src/ 무수정)
# ─────────────────────────────────────────────────────────────────────────────
def _supply_side_at_property_temperature(
    T_property_C: float,
    case: CduCase,
    massloss_flow_Lps: float,
    topology: MassLossTopology,
    secondary_flow_Lps: float,
) -> tuple[
    float, float, tuple[float, ...], MassLossResult, float, float, PumpHydraulicPower
]:
    """공급측 주입일 때의 (T_sup, T_ret, 랙 출구온도들, 수력해, ε, C_min, 펌프열).

    원본(`_steady_at_property_temperature`)과 다른 것은 두 자리뿐이다:
    환수유량 = 랙 통과유량(누출이 랙 상류에서 빠져 뒤에 남지 않는다)이고,
    유출 엔탈피는 T_supply 에서 나가 공급 기준 초과분이 0 이다.
    """
    flow = solve_massloss(
        case.hydraulic,
        -massloss_flow_Lps,
        _mirrored_topology(topology),
        T_property_C,
    )
    rho_kgm3 = coolant_density_kgm3(T_property_C)
    cp_Jkg_K = coolant_cp_Jkg_K(T_property_C)

    Q_racks_Lps = flow.supply_flow_Lps  # = 환수유량. 여기가 원본과 갈리는 자리다
    C_W_K = Q_racks_Lps * _M3_PER_LITRE * rho_kgm3 * cp_Jkg_K
    effectiveness, C_min_W_K, _ = hx_capacity_terms(
        C_W_K, case.ntu, case.T_secondary_supply_C, secondary_flow_Lps
    )
    Q_rack_W = case.rack_load_kW * case.hydraulic.n_racks * _W_PER_KW
    pump_heat = pump_hydraulic_power_W(
        flow.pump_head_mAq,
        flow.pump_flow_Lps,
        flow.rack_flows_Lps,
        case.hydraulic,
        T_property_C,
    )

    dT_C = (Q_rack_W + pump_heat.return_node_W) / C_W_K
    T_return_C = case.T_secondary_supply_C + (
        C_W_K * dT_C + pump_heat.supply_node_W
    ) / (effectiveness * C_min_W_K)
    T_supply_C = T_return_C - dT_C
    rack_outlet_temps_C = tuple(
        T_supply_C
        + case.rack_load_kW * _W_PER_KW / (Q_i * _M3_PER_LITRE * rho_kgm3 * cp_Jkg_K)
        + pump_heat.return_node_W / C_W_K
        for Q_i in flow.rack_flows_Lps
    )
    return (
        T_supply_C,
        T_return_C,
        rack_outlet_temps_C,
        flow,
        effectiveness,
        C_min_W_K,
        pump_heat,
    )


def solve_supply_side_massloss_steady(
    case: CduCase,
    massloss_flow_Lps: float,
    topology: MassLossTopology,
    secondary_flow_Lps: float = HEAT_EXCHANGER.secondary_flow_Lps,
) -> MassLossThermal:
    """공급측 주입 정상상태 — 물성 온도 고정점. `MassLossThermal` 에 담아 낸다.

    담는 그릇을 그대로 쓰므로 `energy_balance_residual_percent` 도 그대로 쓴다:
    `return_flow_Lps` 가 랙 통과유량이고 `massloss_enthalpy_kW = 0` 이면 그
    property 가 공급측 주입의 6장 ① 잔차를 정확히 낸다.

    절대 규칙 5: 바깥 `fsolve` 의 `ier` 를 결과에 싣는다.
    """

    def residual(x: np.ndarray) -> np.ndarray:
        T_prop_C = float(x[0])
        T_sup_C, T_ret_C, *_rest = _supply_side_at_property_temperature(
            T_prop_C, case, massloss_flow_Lps, topology, secondary_flow_Lps
        )
        rule_T_C = property_temperature_from_state(T_sup_C, T_ret_C, case.cp_rule)
        return np.array([rule_T_C - T_prop_C])

    guess_C = bulk_mean_temperature_C(
        SCENARIO.T_primary_supply_C, SCENARIO.T_primary_return_C
    )
    solution, _info, ier, message = fsolve(
        residual, np.array([guess_C]), full_output=True
    )

    T_prop_C = float(solution[0])
    (
        T_supply_C,
        T_return_C,
        rack_outlet_temps_C,
        flow,
        effectiveness,
        C_min_W_K,
        pump_heat,
    ) = _supply_side_at_property_temperature(
        T_prop_C, case, massloss_flow_Lps, topology, secondary_flow_Lps
    )
    Q_hx_W = effectiveness * C_min_W_K * (T_return_C - case.T_secondary_supply_C)
    return MassLossThermal(
        case=case,
        topology=topology,
        massloss_flow_Lps=massloss_flow_Lps,
        T_supply_C=T_supply_C,
        T_return_C=T_return_C,
        rack_outlet_temps_C=rack_outlet_temps_C,
        rack_flows_Lps=flow.rack_flows_Lps,
        supply_flow_Lps=flow.supply_flow_Lps,
        # 공급측 주입이라 환수유량 = 랙 통과유량이다 (원본은 ΣQ_i − Q)
        return_flow_Lps=flow.supply_flow_Lps,
        pump_head_mAq=flow.pump_head_mAq,
        property_eval_T_C=T_prop_C,
        hx_duty_kW=Q_hx_W / _W_PER_KW,
        hx_effectiveness=effectiveness,
        rack_load_kW=case.rack_load_kW * case.hydraulic.n_racks,
        # 유출이 T_supply 에서 나가므로 공급 기준 초과 엔탈피가 0 이다
        massloss_enthalpy_kW=0.0,
        pump_heat_return_node_kW=pump_heat.return_node_W / _W_PER_KW,
        pump_heat_supply_node_kW=pump_heat.supply_node_W / _W_PER_KW,
        outer_solver_ier=int(ier),
        outer_solver_message=str(message).strip(),
        hydraulic_solver_converged=flow.solver_converged,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 계측
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Row:
    """한 판의 관측값. 부호를 읽을 다섯 양만 담는다."""

    label: str
    total_flow_Lps: float
    injection_rack_flow_Lps: float
    injection_rack_outlet_C: float
    T_return_C: float
    balance_residual_percent: float
    pump_flow_Lps: float
    solver_ok: bool


def _row_from_massloss(label: str, result: MassLossThermal, pump_flow_Lps: float) -> Row:
    i = LEAK.injection_rack_index
    return Row(
        label=label,
        total_flow_Lps=result.supply_flow_Lps,
        injection_rack_flow_Lps=result.rack_flows_Lps[i],
        injection_rack_outlet_C=result.injection_rack_outlet_temp_C,
        T_return_C=result.T_return_C,
        balance_residual_percent=result.energy_balance_residual_percent,
        pump_flow_Lps=pump_flow_Lps,
        solver_ok=result.solver_converged,
    )


def _blockage_row(case: CduCase, k_multiplier: float, label: str) -> Row:
    """「막힘」 한 판 — 기존 모듈을 그대로 쓴다(본문 무수정)."""
    blocked = CduCase(
        hydraulic=apply_leak_to_rack(case.hydraulic, k_multiplier),
        T_secondary_supply_C=case.T_secondary_supply_C,
        ntu=case.ntu,
        load_percent=case.load_percent,
    )
    solved = solve_cdu_steady_state(blocked)
    i = LEAK.injection_rack_index
    total_Lps = float(sum(solved.flow.rack_flows_Lps))
    return Row(
        label=label,
        total_flow_Lps=total_Lps,
        injection_rack_flow_Lps=solved.flow.rack_flows_Lps[i],
        injection_rack_outlet_C=solved.thermal.rack_return_temps_C[i],
        T_return_C=solved.thermal.T_return_C,
        balance_residual_percent=energy_balance_residual_percent(solved.thermal),
        pump_flow_Lps=total_Lps,
        solver_ok=solved.solver_converged,
    )


def _sign(value: float, tol: float) -> str:
    if value > tol:
        return "+"
    if value < -tol:
        return "-"
    return "0"


def build_rows() -> tuple[CduCase, MassLossTopology, float, list[Row]]:
    # 케이스: 단일 CDU · 부하 100 % (`default_cdu_cases` 기본값) · 32조합의 첫째.
    # 32조합 중 하나를 골라야 하고 이 판은 부호만 읽으므로 첫째를 쓴다 —
    # 다른 31조합은 이 판이 돌리지 않는다.
    case = default_cdu_cases()[0]
    topology = _chosen_topology()

    T_rated_C = bulk_mean_temperature_C(
        SCENARIO.T_primary_supply_C, SCENARIO.T_primary_return_C
    )
    bound_Lps = massloss_flow_bound_Lps(k_approx_results(case.hydraulic, T_rated_C))
    massloss_Lps = SWEEP_FRACTIONS[-1] * bound_Lps  # 스윕 상한 = 「샘」 최대

    normal = solve_massloss_steady(case, 0.0, topology)
    return_side = solve_massloss_steady(case, massloss_Lps, topology)
    supply_side = solve_supply_side_massloss_steady(case, massloss_Lps, topology)

    rows = [
        _row_from_massloss("⑴ 정상 (Q=0)", normal, normal.supply_flow_Lps),
        _row_from_massloss(
            "⑵ 환수측 「샘」", return_side, return_side.supply_flow_Lps
        ),
        _row_from_massloss(
            "⑶ 공급측 「샘」", supply_side, supply_side.supply_flow_Lps + massloss_Lps
        ),
        _blockage_row(case, 1.0, "⑷ 「막힘」 기준 (K×1.0)"),
        _blockage_row(case, LEAK.k_multiplier_levels[-1][1], "⑸ 「막힘」 (K+50%)"),
    ]
    return case, topology, massloss_Lps, rows


def format_report(
    case: CduCase, topology: MassLossTopology, massloss_Lps: float, rows: list[Row]
) -> str:
    normal, return_side, supply_side, blockage_base, blockage = rows
    lines = [
        "세션 7.65 — 공급측 「샘」 부호 계측 (한 케이스 · 한 배치 · 한 크기)",
        "※ " + ASSUMPTION_TAG,
        "※ 전수 판정이 아니다. 미해결 #36 · 파일럿 종료 판정 5-b 를 닫지 않는다.",
        "",
        f"케이스   : {case.label}",
        f"배치     : {topology.label}"
        f" (공급측은 그 거울 {_mirrored_topology(topology).label})",
        f"「샘」 크기: {massloss_Lps:.6f} L/s "
        f"(5-1 역산 상한 × SWEEP_FRACTIONS[-1]={SWEEP_FRACTIONS[-1]:g})",
        f"주입랙   : index {LEAK.injection_rack_index}",
        "",
        "── 절대값 " + "─" * 66,
        f"{'판':<22}{'총유량':>11}{'주입랙유량':>12}{'주입랙출구':>12}"
        f"{'T_return':>11}{'잔차':>10}{'펌프유량':>11}{'solver':>8}",
        f"{'':<22}{'[L/s]':>11}{'[L/s]':>12}{'[C]':>12}{'[C]':>11}"
        f"{'[%]':>10}{'[L/s]':>11}{'':>8}",
    ]
    for row in rows:
        lines.append(
            f"{row.label:<22}{row.total_flow_Lps:>11.5f}"
            f"{row.injection_rack_flow_Lps:>12.5f}"
            f"{row.injection_rack_outlet_C:>12.5f}{row.T_return_C:>11.5f}"
            f"{row.balance_residual_percent:>10.5f}{row.pump_flow_Lps:>11.5f}"
            f"{'ok' if row.solver_ok else '실패':>8}"
        )

    lines += [
        "",
        "── 각자의 기준 대비 Δ " + "─" * 53,
        "   「샘」 둘은 ⑴ 대비 · 「막힘」은 ⑷ 대비 (같은 코드 경로끼리 뺀다)",
        f"{'항목':<22}{'⑵ 환수측':>16}{'⑶ 공급측':>16}{'⑸ 막힘':>16}",
    ]
    fields = (
        ("총유량 [L/s]", "total_flow_Lps", 1.0e-9),
        ("주입랙 유량 [L/s]", "injection_rack_flow_Lps", 1.0e-9),
        ("주입랙 출구 [K]", "injection_rack_outlet_C", 1.0e-9),
        ("T_return [K]", "T_return_C", 1.0e-9),
        ("잔차 [%p]", "balance_residual_percent", 1.0e-9),
    )
    for name, field, tol in fields:
        d_ret = getattr(return_side, field) - getattr(normal, field)
        d_sup = getattr(supply_side, field) - getattr(normal, field)
        d_blk = getattr(blockage, field) - getattr(blockage_base, field)
        lines.append(
            f"{name:<22}"
            f"{d_ret:>+15.6f}{_sign(d_ret, tol):>1}"
            f"{d_sup:>+15.6f}{_sign(d_sup, tol):>1}"
            f"{d_blk:>+15.6f}{_sign(d_blk, tol):>1}"
        )
    lines += [
        "",
        "부호 읽는 임계 1e-9 — 이 판이 내는 Δ 보다 여러 자리 작다.",
    ]
    return "\n".join(lines)


def main() -> int:
    case, topology, massloss_Lps, rows = build_rows()
    print(format_report(case, topology, massloss_Lps, rows))
    if not all(row.solver_ok for row in rows):
        print("\n※ solver 실패가 있다 — 위 값을 쓰지 않는다 (절대 규칙 5).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
