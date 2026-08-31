"""누출 **질량손실**의 열·전이 관측 — 세션 5.7-D.

세션 5.6 (`leak_massloss.py`) 은 **수력 정상상태 한정**이었다. 이 파일은 그 위에
**2노드 온도 모델**을 얹어 온도와 시간축까지 본다. 판정 기준은 `PROCEED.md`
「세션 5.7-D … 판정 기준 선기재」에 **코드보다 먼저** 적었다.

**절대 규칙 8 예외다 — 사람이 승인했고 이 판까지 이어진다.** 조건은 세션 5.6 과
같다: **별도 코드 · 데이터셋 미변경 · 5-1 의 누출 정의는 K 근사 그대로 · 규칙 8
문언 무수정.** 이 모듈은 데이터셋 생성 경로에 **들어가지 않는다** — 저장소의 어느
모듈도 이 파일을 import 하지 않는다.

**5-1 「보충수 처리」(세션 5.7-C 확정)가 전제다.** 보충수를 모델링하지 않는다 —
조기감지 시나리오에 정의상 등장하지 않는다. 세션 5.6 이 수력 한정을 택한 이유가
이것이었고, 그것이 닫혔으므로 열·전이를 볼 수 있다.

**열모델 — 질량손실이 바꾸는 것은 딱 한 군데다**::

    환수(hot) 노드 :  M_hot·cp·dT_ret/dt = ṁ_sup·cp·(T_sup − T_ret) + Q_rack
    공급(cold) 노드:  M_cold·cp·dT_sup/dt = ṁ_ret·cp·(T_ret − T_sup) − Q_hx

- **환수 노드는 밀폐루프와 같다.** 유입 ṁ_sup, 유출 ṁ_ret + ṁ_leak = ṁ_sup 이라
  이 노드의 **질량은 정확히 보존된다.** 누출 유체가 T_ret 로 나가므로 유출
  엔탈피가 ṁ_sup·cp·T_ret 로 합쳐진다.
- **공급 노드만 ṁ_ret 로 바뀐다.** 열교환기를 지나는 것은 환수유량이므로
  `Q_hx = ε·C_min·(T_ret − T_2차)` 의 C_1차 도 ṁ_ret·cp 다.
- 재고가 줄어드는 것은 **공급 노드**다 (dM_cold/dt = −ṁ_leak).

**절대온도가 식에 남지 않는다.** 변질량 형태 d(M·T)/dt 에서 출발해 M 을 얼리면
T·dM/dt 항이 유출항의 절대온도 항과 **정확히 상쇄**되어 위 식이 된다. 온도 영점에
의존하는 항이 남았다면 상수 M 근사를 잘못 적용한 것이다.

**M 을 상수로 둔다**(판정 기준 「계통이 줄어드는 처리」). 감소를 넣으면 초기
보유량이라는 새 처리가 붙는다. 대신 `inventory_loss_percent` 가 그 근사의 크기를
케이스마다 낸다.

**누출 위치는 5-1 「누출 주입 지점」 그대로 랙 출구**이고, 5-1 이 헤더를 저항 0 의
공통 노드로 두므로 그 지점의 온도는 환수 헤더 온도 T_ret 다 — **새 숫자가 아니다.**

**게이트가 아니다.** 6장 기준 넷을 판정하지 않는다. `energy_balance` 잔차는 **이
별도 코드의 자기정합성 확인**이지 6장 ① 판정이 아니다.

**모든 수치는 가정값 기반이며 실측이 아니다.**
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from cdu_simul.assumptions import (
    ASSUMPTION_TAG,
    HEAT_EXCHANGER,
    LEAK,
    LOAD_PROFILE,
    PIPING,
    PLANT,
    SCENARIO,
)
from cdu_simul.dynamics import (
    INTEGRATION_ATOL,
    INTEGRATION_HORIZON_IN_TAU,
    INTEGRATION_RTOL,
    HoldupBound,
    holdup_bounds,
    storage_times_s,
    time_to_fraction_of_step_s,
)
from cdu_simul.fluid import (
    coolant_cp_Jkg_K,
    coolant_density_kgm3,
    coolant_enthalpy_Jkg,
)
from cdu_simul.hydraulics import (
    HydraulicCase,
    bulk_mean_temperature_C,
    rated_property_temperature_C,
)
from cdu_simul.leak_massloss import (
    SWEEP_FRACTIONS,
    LeakTopology,
    k_approx_results,
    leak_flow_bound_Lps,
    leak_topologies,
    solve_massloss,
)
from cdu_simul.model import (
    CduCase,
    default_cdu_cases,
    hx_capacity_terms,
    property_temperature_from_state,
    solve_cdu_steady_state,
)

_M3_PER_LITRE: float = 1.0e-3
_W_PER_KW: float = 1.0e3


# ─────────────────────────────────────────────────────────────────────────────
# 정상상태 — 질량손실 열모델
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MassLossThermal:
    """질량손실 열 정상상태 1건. solver 플래그를 함께 싣는다(절대 규칙 5)."""

    case: CduCase
    topology: LeakTopology
    leak_flow_Lps: float
    T_supply_C: float
    T_return_C: float
    rack_outlet_temps_C: tuple[float, ...]
    supply_flow_Lps: float
    return_flow_Lps: float
    pump_head_mAq: float
    property_eval_T_C: float
    hx_duty_kW: float
    rack_load_kW: float
    leak_enthalpy_kW: float
    outer_solver_ier: int
    outer_solver_message: str
    hydraulic_solver_converged: bool

    @property
    def solver_converged(self) -> bool:
        return self.outer_solver_ier == 1 and self.hydraulic_solver_converged

    @property
    def leak_rack_outlet_temp_C(self) -> float:
        """누출랙 콜드플레이트 출구온도 [℃] — 판정 기준 A."""
        return self.rack_outlet_temps_C[LEAK.injection_rack_index]

    @property
    def balance_residual_without_leak_percent(self) -> float:
        """leak 항을 **빼고** 본 잔차 [%] — 닫히지 않아야 정상이다."""
        return (self.hx_duty_kW - self.rack_load_kW) / self.rack_load_kW * 100.0

    @property
    def balance_residual_with_leak_percent(self) -> float:
        """leak 항을 **넣고** 본 잔차 [%] — 판정 기준 D.

        `Q_hx` 는 ε-NTU 경로, `Q_rack` 은 5장 입력, 누출 엔탈피는 **CoolProp 직접
        조회**다. 세 경로가 다르므로 항등식이 아니다.
        """
        return (
            (self.hx_duty_kW + self.leak_enthalpy_kW - self.rack_load_kW)
            / self.rack_load_kW
            * 100.0
        )


def _steady_at_property_temperature(
    T_property_C: float,
    case: CduCase,
    leak_flow_Lps: float,
    topology: LeakTopology,
    secondary_flow_Lps: float,
) -> tuple[float, float, tuple[float, ...], float, float, float, float]:
    """물성 온도가 주어졌을 때의 온도들 (순수 함수).

    반환: (T_sup, T_ret, 랙 출구온도들, Q_sup, 펌프양정, Q_hx [W], ρ).

    dT/dt = 0 을 모듈 docstring 의 두 식에 넣으면 닫힌 형태가 된다::

        ΔT    = Q_rack / (ṁ_sup·cp)
        T_ret = T_2차 + ṁ_ret·cp·ΔT / (ε·C_min)
        T_sup = T_ret − ΔT
    """
    flow = solve_massloss(case.hydraulic, leak_flow_Lps, topology, T_property_C)
    rho_kgm3 = coolant_density_kgm3(T_property_C)
    cp_Jkg_K = coolant_cp_Jkg_K(T_property_C)

    Q_supply_Lps = flow.supply_flow_Lps
    Q_return_Lps = Q_supply_Lps - leak_flow_Lps
    C_supply_W_K = Q_supply_Lps * _M3_PER_LITRE * rho_kgm3 * cp_Jkg_K
    C_return_W_K = Q_return_Lps * _M3_PER_LITRE * rho_kgm3 * cp_Jkg_K

    effectiveness, C_min_W_K = hx_capacity_terms(
        C_return_W_K, case.ntu, case.T_secondary_supply_C, secondary_flow_Lps
    )
    Q_rack_W = case.rack_load_kW * case.hydraulic.n_racks * _W_PER_KW

    dT_C = Q_rack_W / C_supply_W_K
    T_return_C = (
        case.T_secondary_supply_C
        + C_return_W_K * dT_C / (effectiveness * C_min_W_K)
    )
    T_supply_C = T_return_C - dT_C

    rack_outlet_temps_C = tuple(
        T_supply_C
        + case.rack_load_kW
        * _W_PER_KW
        / (Q_i * _M3_PER_LITRE * rho_kgm3 * cp_Jkg_K)
        for Q_i in flow.rack_flows_Lps
    )
    Q_hx_W = effectiveness * C_min_W_K * (T_return_C - case.T_secondary_supply_C)
    return (
        T_supply_C,
        T_return_C,
        rack_outlet_temps_C,
        Q_supply_Lps,
        flow.pump_head_mAq,
        Q_hx_W,
        rho_kgm3,
    )


def solve_massloss_steady(
    case: CduCase,
    leak_flow_Lps: float,
    topology: LeakTopology,
    secondary_flow_Lps: float = HEAT_EXCHANGER.secondary_flow_Lps,
) -> MassLossThermal:
    """질량손실 정상상태를 푼다 — 물성 온도 고정점 + quasi-steady 수력.

    구조는 `model.solve_cdu_steady_state` 와 같다(절대 규칙 4 하이브리드).
    `leak_flow_Lps = 0` 이면 수력이 K 근사 정상 케이스와 항등이고 공급=환수라
    열식도 밀폐루프와 항등이다 — 격리 확인이 이것을 쓴다.

    절대 규칙 5: 바깥 `fsolve` 의 `ier` 를 결과에 싣는다. 안쪽 수력이 실패하면
    `solve_massloss` 가 예외를 던진다.
    """

    def residual(x: np.ndarray) -> np.ndarray:
        T_prop_C = float(x[0])
        T_sup_C, T_ret_C, *_ = _steady_at_property_temperature(
            T_prop_C, case, leak_flow_Lps, topology, secondary_flow_Lps
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
        Q_supply_Lps,
        pump_head,
        Q_hx_W,
        rho_kgm3,
    ) = _steady_at_property_temperature(
        T_prop_C, case, leak_flow_Lps, topology, secondary_flow_Lps
    )

    m_leak_kgs = leak_flow_Lps * _M3_PER_LITRE * rho_kgm3
    dh_Jkg = coolant_enthalpy_Jkg(T_return_C) - coolant_enthalpy_Jkg(T_supply_C)
    return MassLossThermal(
        case=case,
        topology=topology,
        leak_flow_Lps=leak_flow_Lps,
        T_supply_C=T_supply_C,
        T_return_C=T_return_C,
        rack_outlet_temps_C=rack_outlet_temps_C,
        supply_flow_Lps=Q_supply_Lps,
        return_flow_Lps=Q_supply_Lps - leak_flow_Lps,
        pump_head_mAq=pump_head,
        property_eval_T_C=T_prop_C,
        hx_duty_kW=Q_hx_W / _W_PER_KW,
        rack_load_kW=case.rack_load_kW * case.hydraulic.n_racks,
        leak_enthalpy_kW=m_leak_kgs * dh_Jkg / _W_PER_KW,
        outer_solver_ier=int(ier),
        outer_solver_message=str(message).strip(),
        hydraulic_solver_converged=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 전이 — 누출을 t=0 에 계단으로 넣는다 (M 은 상수)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MassLossTransient:
    """질량손실 전이 1건. solver 플래그를 함께 싣는다(절대 규칙 5)."""

    case: CduCase
    topology: LeakTopology
    holdup: HoldupBound
    leak_flow_Lps: float
    t_s: np.ndarray
    T_supply_C: np.ndarray
    T_return_C: np.ndarray
    tau_theory_s: float
    t_end_s: float
    solver_success: bool
    solver_message: str

    @property
    def T_return_initial_C(self) -> float:
        return float(self.T_return_C[0])

    @property
    def T_return_final_C(self) -> float:
        return float(self.T_return_C[-1])

    @property
    def t63_s(self) -> float | None:
        """T_return 이 순변화의 63% 에 처음 도달한 시각 [s]. 도달 못하면 None."""
        return time_to_fraction_of_step_s(self, 0.63)

    @property
    def inventory_loss_percent(self) -> float:
        """관측 창 동안 빠져나간 질량 / M × 100 [%] — 상수 M 근사의 크기.

        판정 기준 「계통이 줄어드는 처리」가 요구하는 값이다. 관측 창 길이는
        기존 전이 설정(`INTEGRATION_HORIZON_IN_TAU`)을 그대로 쓴다.
        """
        return self._loss_percent_over(self.t_end_s)

    @property
    def inventory_loss_at_t63_percent(self) -> float:
        """t63 까지의 소실률 [%] — **신호가 서는 시간척도**에서의 유보 크기.

        전체 창(30τ)의 값과 나란히 읽는다: 전체 창이 크더라도 신호가 t63 안에서
        결정되면 그 구간의 근사 오차가 실제 유보다.
        """
        t63 = self.t63_s
        return float("nan") if t63 is None else self._loss_percent_over(t63)

    def _loss_percent_over(self, duration_s: float) -> float:
        rho_kgm3 = coolant_density_kgm3(
            property_temperature_from_state(
                self.T_return_initial_C, self.T_return_final_C, "bulk_mean"
            )
        )
        lost_kg = self.leak_flow_Lps * _M3_PER_LITRE * rho_kgm3 * duration_s
        return lost_kg / self.holdup.mass_kg * 100.0


def integrate_massloss_step(
    case: CduCase,
    leak_flow_Lps: float,
    topology: LeakTopology,
    holdup: HoldupBound,
    secondary_flow_Lps: float = HEAT_EXCHANGER.secondary_flow_Lps,
    horizon_in_tau: float = INTEGRATION_HORIZON_IN_TAU,
) -> MassLossTransient:
    """정격 운전 중 t=0 에 질량손실 누출을 계단으로 넣고 적분한다.

    초기조건은 **누출 전(Q_leak=0) 정상상태**다 — `dynamics.integrate_leak_step`
    과 같은 자극 형태다. 케이스마다 이 함수를 새로 호출해 초기조건을 명시적으로
    리셋한다(collaboration.md 결함유형 ④).

    M 의 노드 배분은 5-1 그대로 공급 50% · 환수 50% 이고, **M 은 상수다**
    (모듈 docstring · 판정 기준). 매 시점 수력을 다시 푼다(절대 규칙 4).

    절대 규칙 5: `solve_ivp` 의 `success` 를 결과에 싣고, 안쪽 수력이 실패하면
    `solve_massloss` 가 예외를 던진다.
    """
    initial = solve_massloss_steady(case, 0.0, topology, secondary_flow_Lps)
    if not initial.solver_converged:
        raise RuntimeError(
            f"{case.label} / {topology.label}: 초기 정상상태가 수렴하지 않았다 — "
            f"{initial.outer_solver_message}"
        )

    mass_hot_kg = PIPING.holdup_supply_node_fraction * holdup.mass_kg
    mass_cold_kg = PIPING.holdup_return_node_fraction * holdup.mass_kg
    Q_rack_W = case.rack_load_kW * case.hydraulic.n_racks * _W_PER_KW

    rho_initial = coolant_density_kgm3(initial.property_eval_T_C)
    m_dot_initial_kgs = initial.supply_flow_Lps * _M3_PER_LITRE * rho_initial
    tau_theory_s = holdup.mass_kg / m_dot_initial_kgs
    t_end_s = horizon_in_tau * tau_theory_s

    def rhs(_t: float, y: np.ndarray) -> list[float]:
        T_sup_C, T_ret_C = float(y[0]), float(y[1])
        T_prop_C = property_temperature_from_state(T_sup_C, T_ret_C, "bulk_mean")
        rho_kgm3 = coolant_density_kgm3(T_prop_C)
        cp_Jkg_K = coolant_cp_Jkg_K(T_prop_C)

        flow = solve_massloss(case.hydraulic, leak_flow_Lps, topology, T_prop_C)
        Q_supply_Lps = flow.supply_flow_Lps
        C_supply_W_K = Q_supply_Lps * _M3_PER_LITRE * rho_kgm3 * cp_Jkg_K
        C_return_W_K = (
            (Q_supply_Lps - leak_flow_Lps) * _M3_PER_LITRE * rho_kgm3 * cp_Jkg_K
        )
        effectiveness, C_min_W_K = hx_capacity_terms(
            C_return_W_K, case.ntu, case.T_secondary_supply_C, secondary_flow_Lps
        )
        Q_hx_W = effectiveness * C_min_W_K * (T_ret_C - case.T_secondary_supply_C)
        return [
            (C_return_W_K * (T_ret_C - T_sup_C) - Q_hx_W) / (mass_cold_kg * cp_Jkg_K),
            (C_supply_W_K * (T_sup_C - T_ret_C) + Q_rack_W) / (mass_hot_kg * cp_Jkg_K),
        ]

    solution = solve_ivp(
        rhs,
        t_span=(0.0, t_end_s),
        y0=[initial.T_supply_C, initial.T_return_C],
        method="RK45",
        rtol=INTEGRATION_RTOL,
        atol=INTEGRATION_ATOL,
        dense_output=True,
        t_eval=storage_times_s(t_end_s, tau_theory_s),
    )
    return MassLossTransient(
        case=case,
        topology=topology,
        holdup=holdup,
        leak_flow_Lps=leak_flow_Lps,
        t_s=solution.t,
        T_supply_C=solution.y[0],
        T_return_C=solution.y[1],
        tau_theory_s=tau_theory_s,
        t_end_s=t_end_s,
        solver_success=bool(solution.success),
        solver_message=str(solution.message).strip(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 다중 CDU — 누출을 CDU A 에만 (판정 기준 F)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PlantMassLoss:
    """2대 결합 해 1건. 누출은 CDU 0 에만 걸린다(5-1 「누출 주입 지점」·#34)."""

    cdu_results: tuple[MassLossThermal, ...]
    secondary_shares_Lps: tuple[float, ...]
    share_uses_return_flow: bool
    top_level_solver_ier: int
    top_level_solver_message: str

    @property
    def solver_converged(self) -> bool:
        return self.top_level_solver_ier == 1 and all(
            r.solver_converged for r in self.cdu_results
        )


def solve_plant_massloss(
    templates: tuple[CduCase, ...],
    leak_flow_Lps: float,
    topology: LeakTopology,
    share_uses_return_flow: bool,
) -> PlantMassLoss:
    """상위 레벨 연립으로 CDU 간 연동을 푼다 — 누출은 CDU 0 에만.

    구조는 `plant.solve_plant_steady_state` 와 같다: 미지수는 CDU 마다 물성
    평가온도 하나이고, 공유 2차측 배분이 총 1차측 유량에 걸려 식이 분리되지
    않는다(5-1 「공유 2차측 결합 방식」).

    **`share_uses_return_flow` 는 구조 자유도다**(판정 기준 「구조 자유도」).
    누출 CDU 는 ṁ_sup ≠ ṁ_ret 라 5-1 의 「1차측 유량」이 처음으로 갈린다.
    열교환기를 지나는 것은 환수유량이므로 `True` 가 물리적으로 정해진 읽기이지만,
    **값을 고르지 않고 양 끝을 둘 다 돌린다** — 분기의 양 끝이라 새 가정치가
    아니다(g 와 같은 취급).
    """
    leaks = (leak_flow_Lps,) + (0.0,) * (len(templates) - 1)

    def primary_flows(temps: tuple[float, ...]) -> tuple[float, ...]:
        flows = []
        for cdu, T_C, leak in zip(templates, temps, leaks, strict=True):
            flow = solve_massloss(cdu.hydraulic, leak, topology, T_C)
            Q_supply_Lps = flow.supply_flow_Lps
            flows.append(
                Q_supply_Lps - leak if share_uses_return_flow else Q_supply_Lps
            )
        return tuple(flows)

    def residuals(x: np.ndarray) -> np.ndarray:
        temps = tuple(float(v) for v in x)
        shares = PLANT.secondary_shares_Lps(primary_flows(temps))
        out = []
        for cdu, T_C, leak, share in zip(templates, temps, leaks, shares, strict=True):
            T_sup_C, T_ret_C, *_ = _steady_at_property_temperature(
                T_C, cdu, leak, topology, share
            )
            out.append(
                property_temperature_from_state(T_sup_C, T_ret_C, cdu.cp_rule) - T_C
            )
        return np.array(out)

    guess_C = bulk_mean_temperature_C(
        SCENARIO.T_primary_supply_C, SCENARIO.T_primary_return_C
    )
    solution, _info, ier, message = fsolve(
        residuals, np.full(len(templates), guess_C), full_output=True
    )
    temps = tuple(float(v) for v in solution)
    shares = PLANT.secondary_shares_Lps(primary_flows(temps))
    return PlantMassLoss(
        cdu_results=tuple(
            solve_massloss_steady(cdu, leak, topology, share)
            for cdu, leak, share in zip(templates, leaks, shares, strict=True)
        ),
        secondary_shares_Lps=shares,
        share_uses_return_flow=share_uses_return_flow,
        top_level_solver_ier=int(ier),
        top_level_solver_message=str(message).strip(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 스윕 — 판정 기준 A~G
# ─────────────────────────────────────────────────────────────────────────────
#: 5장 부하 양 끝 2수준 (방침 B). 중점을 고르지 않는다.
_LOAD_PERCENTS: tuple[float, ...] = (
    LOAD_PROFILE.idle_load_percent,
    LOAD_PROFILE.rated_load_percent,
)


def thermal_cases() -> list[CduCase]:
    """5장·5-1 양 끝 32조합 × 부하 2수준 = **64 케이스**.

    `model.default_cdu_cases` 를 그대로 쓴다 — 축을 새로 고르지 않는다(방침 B).
    """
    return [
        case
        for load_percent in _LOAD_PERCENTS
        for case in default_cdu_cases(load_percent)
    ]


def leak_sizes_Lps(hydraulic: HydraulicCase, T_property_C: float) -> list[float]:
    """누출 크기 스윕 [L/s] — 세션 5.6 과 같다. 새 값을 만들지 않는다."""
    bound = leak_flow_bound_Lps(k_approx_results(hydraulic, T_property_C))
    return [fraction * bound for fraction in SWEEP_FRACTIONS]


@dataclass(frozen=True)
class ThermalDeltas:
    """정상(누출 0) 대비 세 온도의 변화 — 판정 기준 A·B·C."""

    leak_rack_outlet_C: float
    T_return_C: float
    T_supply_C: float


def massloss_thermal_deltas(
    normal: MassLossThermal, leaked: MassLossThermal
) -> ThermalDeltas:
    """질량손실의 정상 대비 변화. 기준은 **같은 배치의 Q_leak=0 해**다."""
    return ThermalDeltas(
        leak_rack_outlet_C=(
            leaked.leak_rack_outlet_temp_C - normal.leak_rack_outlet_temp_C
        ),
        T_return_C=leaked.T_return_C - normal.T_return_C,
        T_supply_C=leaked.T_supply_C - normal.T_supply_C,
    )


def k_approx_thermal_deltas(case: CduCase, k_multiplier: float) -> ThermalDeltas:
    """K 근사의 정상 대비 변화 — **기존 모듈을 그대로 쓴다**(본문 무수정).

    누출 랙만 K 가 올라가므로 랙별 유량이 갈리고, 랙 출구온도도 갈린다.
    """
    from cdu_simul.hydraulics import apply_leak_to_rack

    normal = solve_cdu_steady_state(case)
    leaked_case = CduCase(
        hydraulic=apply_leak_to_rack(case.hydraulic, k_multiplier),
        T_secondary_supply_C=case.T_secondary_supply_C,
        ntu=case.ntu,
        load_percent=case.load_percent,
    )
    leaked = solve_cdu_steady_state(leaked_case)
    i = LEAK.injection_rack_index
    return ThermalDeltas(
        leak_rack_outlet_C=(
            leaked.thermal.rack_return_temps_C[i]
            - normal.thermal.rack_return_temps_C[i]
        ),
        T_return_C=leaked.thermal.T_return_C - normal.thermal.T_return_C,
        T_supply_C=leaked.thermal.T_supply_C - normal.thermal.T_supply_C,
    )


#: 부호를 "0" 으로 읽는 절대 임계 [K]. 세션 5.6 의 `_SIGN_ZERO_TOL` 와 같은 취지다.
_SIGN_ZERO_TOL_K: float = 1.0e-12


def sign_of(value: float) -> str:
    if value > _SIGN_ZERO_TOL_K:
        return "+"
    if value < -_SIGN_ZERO_TOL_K:
        return "-"
    return "0"


def sign_summary(values: list[float]) -> str:
    """부호 집합 요약 — 전 조합에서 같으면 한 글자, 갈리면 섞어 적는다."""
    signs = sorted({sign_of(v) for v in values})
    return signs[0] if len(signs) == 1 else "/".join(signs)


def span(values: list[float]) -> str:
    return f"{min(values):+.3e} ~ {max(values):+.3e}"


def _note() -> str:
    return (
        f"{ASSUMPTION_TAG}\n"
        "세션 5.7-D · 관측 판 — 게이트 아님 · 6장 기준 판정 아님.\n"
        "절대 규칙 8 예외(사람 승인) · 데이터셋·5-1 미변경 · M 은 상수.\n"
        "어느 모사가 실제에 가까운지 재지 않았다(실측 없음).\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 전수 실행 — 판정 기준 A~G · 반례 확인
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SteadySweep:
    """정상상태 전수 결과 — A·B·C·D·G 와 반례 확인이 여기서 나온다."""

    massloss: list[ThermalDeltas]
    k_approx: list[ThermalDeltas]
    #: 배치별 질량손실 부호 — G(배치 불변성)가 읽는다.
    massloss_by_topology: dict[str, list[ThermalDeltas]]
    balance_without_leak_percent: list[float]
    balance_with_leak_percent: list[float]
    isolation_max_abs_K: float
    solver_failures: list[str]
    n_solves: int


def run_steady_sweep() -> SteadySweep:
    """64 케이스 × 배치 6 × 누출 5수준. K 근사는 5장 3수준으로 같이 낸다."""
    massloss: list[ThermalDeltas] = []
    by_topology: dict[str, list[ThermalDeltas]] = {}
    without_leak: list[float] = []
    with_leak: list[float] = []
    failures: list[str] = []
    isolation_max = 0.0
    n_solves = 0

    for case in thermal_cases():
        sizes = leak_sizes_Lps(case.hydraulic, rated_property_temperature_C())
        closed_loop = solve_cdu_steady_state(case)
        for topology in leak_topologies():
            solutions = [
                solve_massloss_steady(case, size, topology) for size in sizes
            ]
            n_solves += len(solutions)
            for solution in solutions:
                if not solution.solver_converged:
                    failures.append(
                        f"{case.label} / {topology.label} / "
                        f"Q_leak={solution.leak_flow_Lps:g}: "
                        f"ier={solution.outer_solver_ier}"
                    )
            # 격리 확인 — 누출 0 에서 밀폐루프 해와 같아야 한다
            isolation_max = max(
                isolation_max,
                abs(solutions[0].T_return_C - closed_loop.thermal.T_return_C),
                abs(solutions[0].T_supply_C - closed_loop.thermal.T_supply_C),
            )
            for solution in solutions[1:]:
                deltas = massloss_thermal_deltas(solutions[0], solution)
                massloss.append(deltas)
                by_topology.setdefault(topology.label, []).append(deltas)
                without_leak.append(solution.balance_residual_without_leak_percent)
                with_leak.append(solution.balance_residual_with_leak_percent)

    k_approx = [
        k_approx_thermal_deltas(case, multiplier)
        for case in thermal_cases()
        for _label, multiplier in LEAK.k_multiplier_levels[1:]
    ]
    return SteadySweep(
        massloss=massloss,
        k_approx=k_approx,
        massloss_by_topology=by_topology,
        balance_without_leak_percent=without_leak,
        balance_with_leak_percent=with_leak,
        isolation_max_abs_K=isolation_max,
        solver_failures=failures,
        n_solves=n_solves,
    )


@dataclass(frozen=True)
class TransientSweep:
    """전이 전수 결과 — E 와 소실률 표가 여기서 나온다."""

    tau_ratios: list[float]
    t63_ratios: list[float]
    T_return_deltas_C: list[float]
    loss_full_window_percent: list[float]
    loss_at_t63_percent: list[float]
    solver_failures: list[str]
    n_integrations: int


def run_transient_sweep() -> TransientSweep:
    """64 케이스 × 배치 6 × M 양 끝. 누출 크기는 스윕 상한 하나(방향은 크기 무관)."""
    low, high = holdup_bounds()
    tau_ratios: list[float] = []
    t63_ratios: list[float] = []
    T_return_deltas: list[float] = []
    loss_full: list[float] = []
    loss_t63: list[float] = []
    failures: list[str] = []
    n = 0

    for case in thermal_cases():
        size = leak_sizes_Lps(case.hydraulic, rated_property_temperature_C())[-1]
        for topology in leak_topologies():
            runs = [
                integrate_massloss_step(case, size, topology, holdup)
                for holdup in (low, high)
            ]
            n += len(runs)
            for run, holdup in zip(runs, (low, high), strict=True):
                if not run.solver_success:
                    failures.append(
                        f"{case.label} / {topology.label} / {holdup.label}: "
                        f"{run.solver_message}"
                    )
                loss_full.append(run.inventory_loss_percent)
                loss_t63.append(run.inventory_loss_at_t63_percent)
            T_return_deltas.append(
                runs[0].T_return_final_C - runs[0].T_return_initial_C
            )
            tau_ratios.append(runs[1].tau_theory_s / runs[0].tau_theory_s)
            if runs[0].t63_s is not None and runs[1].t63_s is not None:
                t63_ratios.append(runs[1].t63_s / runs[0].t63_s)
    return TransientSweep(
        tau_ratios=tau_ratios,
        t63_ratios=t63_ratios,
        T_return_deltas_C=T_return_deltas,
        loss_full_window_percent=loss_full,
        loss_at_t63_percent=loss_t63,
        solver_failures=failures,
        n_integrations=n,
    )


@dataclass(frozen=True)
class PlantSweep:
    """다중 CDU 전수 결과 — F 와 그 구조 자유도 의존성."""

    #: 배분 읽기별로 (B 의 2차측 몫 변화, B 의 환수온도 변화)
    by_share_reading: dict[bool, tuple[list[float], list[float]]]
    monotonic_failures: list[str]
    solver_failures: list[str]
    n_solves: int


def run_plant_sweep() -> PlantSweep:
    """64 케이스 × 배치 6 × 누출 5수준 × 배분 읽기 2. 누출은 CDU 0 에만."""
    by_reading: dict[bool, tuple[list[float], list[float]]] = {
        True: ([], []),
        False: ([], []),
    }
    monotonic_failures: list[str] = []
    failures: list[str] = []
    n = 0

    for case in thermal_cases():
        sizes = leak_sizes_Lps(case.hydraulic, rated_property_temperature_C())
        for topology in leak_topologies():
            for uses_return in (True, False):
                runs = [
                    solve_plant_massloss((case, case), size, topology, uses_return)
                    for size in sizes
                ]
                n += len(runs)
                for run, size in zip(runs, sizes, strict=True):
                    if not run.solver_converged:
                        failures.append(
                            f"{case.label} / {topology.label} / "
                            f"share_ret={uses_return} / Q_leak={size:g}: "
                            f"ier={run.top_level_solver_ier}"
                        )
                shares = [run.secondary_shares_Lps[1] for run in runs]
                temps = [run.cdu_results[1].T_return_C for run in runs]
                by_reading[uses_return][0].append(shares[-1] - shares[0])
                by_reading[uses_return][1].append(temps[-1] - temps[0])
                if not _strictly_monotonic(shares):
                    monotonic_failures.append(
                        f"{case.label} / {topology.label} / share_ret={uses_return}"
                    )
    return PlantSweep(
        by_share_reading=by_reading,
        monotonic_failures=monotonic_failures,
        solver_failures=failures,
        n_solves=n,
    )


def _strictly_monotonic(values: list[float]) -> bool:
    """누출 0 을 제외한 구간에서 엄격 단조인가 (증가·감소 어느 쪽이든)."""
    # 인접 쌍이므로 길이가 1 다르다 — strict 를 쓰지 않는다.
    diffs = [b - a for a, b in zip(values, values[1:])]  # noqa: B905
    if all(abs(d) <= _SIGN_ZERO_TOL_K for d in diffs):
        return True
    return all(d > 0.0 for d in diffs) or all(d < 0.0 for d in diffs)


# ─────────────────────────────────────────────────────────────────────────────
# 보고 — 절대 규칙 11 표시를 반드시 넣는다
# ─────────────────────────────────────────────────────────────────────────────
def _delta_field(deltas: list[ThermalDeltas], field: str) -> list[float]:
    return [getattr(d, field) for d in deltas]


_THERMAL_QUANTITIES: tuple[tuple[str, str], ...] = (
    ("A 누출랙 출구온도", "leak_rack_outlet_C"),
    ("B CDU 환수온도", "T_return_C"),
    ("C CDU 공급온도", "T_supply_C"),
)


def format_sign_table(sweep: SteadySweep) -> str:
    """A·B·C 부호 대조표 — K 근사 대 질량손실 (순수 함수)."""
    lines = [
        "표 1. 정상상태 온도 신호의 부호 — K 근사 대 질량손실",
        "  (같은 조합의 누출 0 해 대비 변화. 부호만 대조한다)",
        "",
        f"{'양':<22}{'K 근사':>10}{'질량손실':>12}   {'질량손실 범위 [K]':>28}",
        "-" * 78,
    ]
    for title, field in _THERMAL_QUANTITIES:
        k_values = _delta_field(sweep.k_approx, field)
        m_values = _delta_field(sweep.massloss, field)
        lines.append(
            f"{title:<22}{sign_summary(k_values):>10}"
            f"{sign_summary(m_values):>12}   {span(m_values):>28}"
        )
    lines += [
        "-" * 78,
        f"K 근사 {len(sweep.k_approx)}건 · 질량손실 {len(sweep.massloss)}건"
        f" (정상 해 {sweep.n_solves}건 중 누출 있는 것)",
        "부호가 한 글자면 전 조합에서 일정하다 — 반례 확인(collaboration.md ⑥).",
    ]
    return "\n".join(lines)


def format_topology_table(sweep: SteadySweep) -> str:
    """G 배치 불변성 — 배치 6 각각에서 부호와 크기 (순수 함수)."""
    lines = [
        "표 2. 배치별 질량손실 열 신호 — 판정 기준 G (배치 불변성)",
        "",
        f"{'배치':<20}{'A 부호':>8}{'B 부호':>8}{'C 부호':>8}"
        f"   {'B 크기 범위 [K]':>28}",
        "-" * 78,
    ]
    for label, deltas in sweep.massloss_by_topology.items():
        signs = [
            sign_summary(_delta_field(deltas, field))
            for _title, field in _THERMAL_QUANTITIES
        ]
        lines.append(
            f"{label:<20}{signs[0]:>8}{signs[1]:>8}{signs[2]:>8}"
            f"   {span(_delta_field(deltas, 'T_return_C')):>28}"
        )
    lines += [
        "-" * 78,
        "세션 5.6 관측 ④: `g=0.0/펌프=공급` 배치에서 **수력** 다섯 양이 전부 0 이었다.",
        "그 배치의 행에 열 신호가 서 있으면 수력이 못 보는 것을 열이 본다는 뜻이다.",
    ]
    return "\n".join(lines)


def format_balance_table(sweep: SteadySweep) -> str:
    """D 에너지 balance — leak 항 없이/포함 (순수 함수)."""
    without = sweep.balance_without_leak_percent
    with_leak = sweep.balance_with_leak_percent
    return "\n".join(
        [
            "표 3. 에너지 balance 잔차 — 판정 기준 D",
            "  (별도 코드의 **자기정합성** 확인이다. 6장 기준 ① 판정이 아니다)",
            "",
            f"{'':<28}{'최소 [%]':>14}{'최대 [%]':>14}{'최대 |잔차| [%]':>18}",
            "-" * 78,
            f"{'leak 항 없이':<28}{min(without):>14.6f}{max(without):>14.6f}"
            f"{max(abs(v) for v in without):>18.6f}",
            f"{'leak 항 포함':<28}{min(with_leak):>14.6f}{max(with_leak):>14.6f}"
            f"{max(abs(v) for v in with_leak):>18.6f}",
            "-" * 78,
            "잔차 = (Q_hx + ṁ_leak·[h(T_ret) − h(T_sup)] − Q_rack) / Q_rack × 100.",
            "Q_hx 는 ε-NTU 경로 · Q_rack 은 5장 입력 · 엔탈피는 CoolProp 직접 조회 —",
            "세 경로가 다르므로 항등식이 아니다.",
            f"격리 확인(누출 0 = 밀폐루프 해): 최대 편차 "
            f"{sweep.isolation_max_abs_K:.3e} K",
        ]
    )


def format_transient_table(sweep: TransientSweep) -> str:
    """E 시간응답 + 소실률 표 (순수 함수)."""
    ratios = sweep.tau_ratios
    t63 = sweep.t63_ratios
    full = sweep.loss_full_window_percent
    at_t63 = [v for v in sweep.loss_at_t63_percent if v == v]
    return "\n".join(
        [
            "표 4. 전이 — 판정 기준 E 와 상수 M 근사의 크기",
            "",
            f"{'양':<34}{'최소':>14}{'최대':>14}",
            "-" * 78,
            f"{'τ 비 (M 상한/M 하한)':<34}{min(ratios):>14.6f}{max(ratios):>14.6f}",
            f"{'t63 비 (M 상한/M 하한)':<34}{min(t63):>14.6f}{max(t63):>14.6f}",
            f"{'T_return 순변화 [K]':<34}"
            f"{min(sweep.T_return_deltas_C):>14.6f}"
            f"{max(sweep.T_return_deltas_C):>14.6f}",
            f"{'소실률 · 전체 창 30τ [%]':<34}{min(full):>14.4f}{max(full):>14.4f}",
            f"{'소실률 · t63 까지 [%]':<34}{min(at_t63):>14.4f}{max(at_t63):>14.4f}",
            "-" * 78,
            f"적분 {sweep.n_integrations}건.",
            "소실률 = Q_leak × 시간 × ρ / M × 100 — **M 을 상수로 둔 근사의 크기**다.",
            "이 값이 크면 그만큼의 유보가 결과 해석에 붙는다(판정 기준 §「계통이 …」).",
        ]
    )


def format_plant_table(sweep: PlantSweep) -> str:
    """F 다중 CDU — 누출 없는 CDU B 의 반응 (순수 함수)."""
    lines = [
        "표 5. 다중 CDU — 판정 기준 F (누출은 CDU A 에만)",
        "  세션 5.5-D 의 K 근사 관측: B 의 2차측 몫 **증가** · B 의 환수온도 **감소**",
        "",
        f"{'배분이 보는 1차측 유량':<26}{'B 몫 부호':>12}{'B 환수온도 부호':>16}"
        f"   {'B 몫 변화 [L/s]':>22}",
        "-" * 82,
    ]
    for uses_return in (True, False):
        shares, temps = sweep.by_share_reading[uses_return]
        label = "Q_ret (HX 통과 · 정해진 읽기)" if uses_return else "Q_sup (대안 읽기)"
        lines.append(
            f"{label:<26}{sign_summary(shares):>12}"
            f"{sign_summary(temps):>16}   {span(shares):>22}"
        )
    lines += [
        "-" * 82,
        f"결합 해 {sweep.n_solves}건.",
        f"누출 크기에 대한 B 몫의 엄격 단조 위반: {len(sweep.monotonic_failures)}건",
    ]
    return "\n".join(lines)


def main() -> int:
    """전수 실행 후 표 다섯 개를 낸다. 관측 판이므로 판정하지 않는다."""
    print(_note())
    steady = run_steady_sweep()
    print(format_sign_table(steady), end="\n\n")
    print(format_topology_table(steady), end="\n\n")
    print(format_balance_table(steady), end="\n\n")

    transient = run_transient_sweep()
    print(format_transient_table(transient), end="\n\n")

    plant = run_plant_sweep()
    print(format_plant_table(plant), end="\n\n")

    failures = (
        steady.solver_failures + transient.solver_failures + plant.solver_failures
    )
    print(f"solver 실패 조합: {len(failures)}건")
    for failure in failures:
        print(f"  - {failure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
