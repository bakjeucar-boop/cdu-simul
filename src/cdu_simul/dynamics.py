"""온도 시간적분 모델 (세션 2) — 부하 스텝에 대한 T_return 응답.

범위 — 1-B 정상상태 모델에 **시간축만** 얹는다.

**압력-유량을 매 시점 푼다**(세션 3-B). 절대 규칙 4가 요구하는 "압력은
quasi-steady 대수, 온도는 시간적분"이 여기서 완성된다 — `solve_ivp` 의 우변이
호출될 때마다 그 시점의 1차측 벌크 평균온도로 `hydraulics.solve_flow_distribution`
을 다시 풀어 랙별 유량을 받는다. 유량을 상수로 얼려 두지 않는다.

**M 의 8랙 해석은 세션 2 그대로 두었다.** 5-1 이 등가길이를 "랙 1개 회로의 왕복
전체"로 읽으므로 `holdup_bounds()` 가 내는 M 은 **랙 1개 회로분**이다. 랙이 8개가
되면 계통 전체 보유량을 어떻게 읽어야 하는지(분기는 8벌·헤더는 공용) 5장에도
5-1 에도 답이 없다 — **새 숫자를 만들지 않고 세션 2 값을 그대로 쓴다**(절대 규칙 1).
그 결과 아래 τ·t63·t95 의 **절대값은 8랙에서 해석할 수 없다**. 이 판이 판정하는
방향성·비발산은 M 의 크기와 무관하므로(정상상태는 M 에 의존하지 않고, 전이의
부호도 M>0 이면 바뀌지 않는다) 게이트에는 영향이 없다.

**2차측 동특성을 만들지 않는다**(절대 규칙 7). 2차측 공급온도는 고정 경계조건이다.

**1-B `model.py` 를 수정하지 않는다.** ε 계산·cp 평가 규칙·정상상태 해는 전부
`model` 에서 가져다 쓴다 — 같은 물리를 두 번 적지 않는다(collaboration.md ④).

노드 구성 (2노드 루프):

    ┌── 랙 8개 병렬 (Q_총 유입) ──┐
    │                             ↓
  T_supply 노드               T_return 노드
    ↑                             │
    └──── 열교환기 (Q_hx 유출) ────┘

    M_hot ·cp·dT_return/dt = C·(T_supply - T_return) + Q_rack(t)
    M_cold·cp·dT_supply/dt = C·(T_return - T_supply) - Q_hx(T_return)

여기서 C = m_dot·cp [W/K], Q_hx = ε·C·(T_return - T_2차공급).
dT/dt = 0 을 넣으면 1-B 정상상태 식이 그대로 나온다 — 그래서 t→∞ 대조(C7)가
성립해야 한다.

**모든 수치는 가정값 기반이며 실측이 아니다.** 5장·5-1 값은 assumptions.py 에서만
읽는다(절대 규칙 2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.integrate import solve_ivp

from cdu_simul.assumptions import (
    ASSUMPTION_TAG,
    HEAT_EXCHANGER,
    LEAK,
    LOAD_PROFILE,
    PIPING,
    SCENARIO,
    SESSION_3B_CAVEAT,
    SESSION_5B_CAVEAT,
)
from cdu_simul.fluid import coolant_cp_Jkg_K, coolant_density_kgm3
from cdu_simul.hydraulics import (
    HydraulicCase,
    apply_leak_to_rack,
    solve_flow_distribution,
)
from cdu_simul.hydraulics import default_cases as default_hydraulic_cases
from cdu_simul.model import (
    CduCase,
    _property_temperature_from_state,
    hx_capacity_terms,
    solve_cdu_steady_state,
)

_M3_PER_LITRE: float = 1.0e-3
_W_PER_KW: float = 1.0e3
_M_PER_MM: float = 1.0e-3
_PERCENT: float = 1.0e-2

#: solve_ivp 허용오차. 물리 가정이 아니라 **수치 설정**이다.
#: C7(정상상태 대조)이 이 판의 핵심이라, 적분 오차가 대조 결과를 가리지 않도록
#: 기본값(rtol 1e-3 · atol 1e-6)보다 크게 조인다. 이 계는 상태가 2개뿐이라
#: 이렇게 조여도 계산 비용이 문제되지 않는다.
INTEGRATION_RTOL: float = 1.0e-10
INTEGRATION_ATOL: float = 1.0e-12

#: 저장 표본 추출 — **수치 설정이지 물리 가정이 아니다**(세션 5.5-B).
#: 적분 자체(RK45 · rtol · atol · 구간)는 손대지 않는다. `solve_ivp` 가 어느 시각의
#: 값을 **돌려줄지**만 정한다 — 해는 dense output 에서 같은 정확도로 나온다.
#:
#: 스텝 직후를 조밀하게 잡는 이유: 세션 2 방향성 게이트가 **스텝 직후 기울기**를
#: 쓰는데, 균등 간격으로 30τ 를 201점에 담으면 첫 구간이 0.15τ 나 되어 그 기울기가
#: 뭉개진다. 전이의 정보는 앞쪽에 몰려 있고 뒤쪽은 점근선이다.
#: 분할점 2τ 는 시간상수 몇 배 안에서 변화의 대부분이 끝나기 때문이고, 점 배분
#: 100/100 은 앞뒤에 같은 해상도를 주되 앞쪽 간격을 15배 좁히는 값이다.
#: 이 셋(분할점·앞 점수·뒤 점수)은 **5-1 에 기록하지 않는다** — 물리가 아니다.
STORAGE_SPLIT_IN_TAU: float = 2.0
STORAGE_POINTS_EARLY: int = 100
STORAGE_POINTS_LATE: int = 100

#: 적분 구간을 이론 체류시간 τ 의 몇 배까지 잡을지. **수치 설정**이다.
#: 가장 느린 모드의 시간상수가 τ/ε (ε<1) 이므로 30τ 는 최소 20 시간상수에
#: 해당한다 — 잔여 과도분이 e^-20 ≈ 2e-9 배로 줄어 t→∞ 대조에 영향을 주지 않는다.
INTEGRATION_HORIZON_IN_TAU: float = 30.0


# ─────────────────────────────────────────────────────────────────────────────
# C3. 계통 보유수량(열용량) — 순수 함수
# ─────────────────────────────────────────────────────────────────────────────
def storage_times_s(t_end_s: float, tau_s: float) -> np.ndarray:
    """저장할 시각 [s] — 스텝 직후를 조밀하게 잡는 **비균등** 표본 (순수 함수).

    [수치 설정: 세션 5.5-B. 물리 가정이 아니므로 5-1 에 기록하지 않는다]

    0 ~ `STORAGE_SPLIT_IN_TAU`·τ 에 `STORAGE_POINTS_EARLY` 점,
    그 뒤 `t_end_s` 까지 `STORAGE_POINTS_LATE` 점. 경계 시각은 한 번만 담는다.

    **적분 정확도와 무관하다** — `solve_ivp` 는 자기 스텝으로 풀고 여기 준 시각에서
    dense output 을 평가할 뿐이다. 그래서 이 함수를 바꿔도 게이트 결과가 바뀌지
    않아야 하고, 세션 5.5-B 가 그것을 확인했다.

    τ 가 0 이하이거나 구간이 비면 균등 표본으로 물러난다 — 그런 케이스는 현재
    없지만, 조용히 빈 배열을 내는 것보다 낫다.
    """
    split_s = min(STORAGE_SPLIT_IN_TAU * tau_s, t_end_s)
    if not (0.0 < split_s < t_end_s):
        return np.linspace(0.0, t_end_s, STORAGE_POINTS_EARLY + STORAGE_POINTS_LATE)
    early = np.linspace(0.0, split_s, STORAGE_POINTS_EARLY, endpoint=False)
    late = np.linspace(split_s, t_end_s, STORAGE_POINTS_LATE + 1)
    return np.concatenate([early, late])


def pipe_cross_section_area_m2(inner_diameter_m: float) -> float:
    """원형 배관 단면적 [m^2] (순수 함수)."""
    return math.pi * 0.25 * inner_diameter_m * inner_diameter_m


def system_coolant_mass_kg(
    inner_diameter_m: float, equivalent_length_m: float, density_kgm3: float
) -> float:
    """배관 보유 냉각액 질량 [kg] (순수 함수).

    M = ρ · (π/4 · D^2) · L

    **이 M 은 배관 보유량뿐이다.** 열교환기·CDU 내부·랙 콜드플레이트 보유량은
    5장에 없어 빠져 있다(프로젝트정리 5-1 「계통 보유수량 M」 한계 기록).
    따라서 실제보다 M 이 작고, 여기서 나오는 수렴시간도 실제보다 짧다.
    """
    return (
        density_kgm3
        * pipe_cross_section_area_m2(inner_diameter_m)
        * equivalent_length_m
    )


def holdup_reference_density_kgm3() -> float:
    """M 산출용 기준 밀도 [kg/m^3].

    1-B 와 같은 규약(1차측 벌크 평균온도에서 물성 평가)을 **5장 표의 1차측
    공급·환수 온도**에 적용한 값이다 — 새 규칙도, 새 숫자도 만들지 않는다.

    M 을 시간에 따라 변하게 하지 않고 하나로 고정하는 이유: M 은 계통에 담긴
    냉각액의 양이고, 이 온도대에서 ρ 변화는 0.5% 미만이라 시간상수에만 미미하게
    영향한다. 고정해야 5-1 이 요구한 "하한/상한 **두 값**"이 그대로 나온다.
    (평형 온도를 결정하는 cp·ρ 는 아래 미분방정식 안에서 매 시점 다시 평가한다 —
    그래야 t→∞ 극한이 1-B 해와 정확히 일치한다.)
    """
    reference_T_C = _property_temperature_from_state(
        SCENARIO.T_primary_supply_C, SCENARIO.T_primary_return_C, "bulk_mean"
    )
    return coolant_density_kgm3(reference_T_C)


@dataclass(frozen=True)
class HoldupBound:
    """M 범위의 한쪽 끝. 5-1 방침 (B) — 대표값을 고르지 않는다."""

    label: str
    inner_diameter_mm: float
    equivalent_length_m: float
    mass_kg: float


def holdup_bounds() -> tuple[HoldupBound, HoldupBound]:
    """M 의 하한·상한 두 값 (프로젝트정리 5-1).

    하한 = 전부 25A + 등가길이 하한, 상한 = 전부 80A + 등가길이 상한.
    구경과 등가길이가 **둘 다** 5장에서 범위로 주어졌으므로, M 의 실제 범위를
    내려면 양쪽 끝을 함께 취해야 한다. 중간값은 만들지 않는다.
    """
    rho_kgm3 = holdup_reference_density_kgm3()
    d_low_mm, d_high_mm = PIPING.holdup_bound_inner_diameters_mm
    return (
        HoldupBound(
            label="M 하한 (전부 25A · 등가길이 하한)",
            inner_diameter_mm=d_low_mm,
            equivalent_length_m=PIPING.equivalent_length_m.low,
            mass_kg=system_coolant_mass_kg(
                d_low_mm * _M_PER_MM, PIPING.equivalent_length_m.low, rho_kgm3
            ),
        ),
        HoldupBound(
            label="M 상한 (전부 80A · 등가길이 상한)",
            inner_diameter_mm=d_high_mm,
            equivalent_length_m=PIPING.equivalent_length_m.high,
            mass_kg=system_coolant_mass_kg(
                d_high_mm * _M_PER_MM, PIPING.equivalent_length_m.high, rho_kgm3
            ),
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# C4. 동적 모델
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LoadStepCase:
    """부하 스텝 시나리오 1건. 케이스마다 새로 만들어 초기조건을 리셋한다."""

    label: str
    holdup: HoldupBound
    load_before_percent: float
    load_after_percent: float
    T_secondary_supply_C: float
    ntu: float
    hydraulic: HydraulicCase
    #: 이 CDU 의 2차측 부피유량 [L/s]. Cr 은 여기서 유도된다(세션 5-B).
    secondary_flow_Lps: float = HEAT_EXCHANGER.secondary_flow_Lps

    def steady_case(self, load_percent: float) -> CduCase:
        """주어진 부하율에서의 결합 정상상태 케이스를 만든다 (물리 정의 재사용).

        세션 3-B 부터 초기조건은 **수력과 결합한** 정상상태다 — 유량이 5장 정격
        고정이 아니라 압력평형의 해이기 때문이다.
        """
        return CduCase(
            hydraulic=self.hydraulic,
            T_secondary_supply_C=self.T_secondary_supply_C,
            ntu=self.ntu,
            load_percent=load_percent,
        )

    @property
    def is_rising_step(self) -> bool:
        return self.load_after_percent > self.load_before_percent


@dataclass(frozen=True)
class TransientResult:
    """시간적분 결과 1건. solver 성공 플래그를 함께 들고 다닌다(절대 규칙 5)."""

    case: LoadStepCase
    t_s: np.ndarray
    T_supply_C: np.ndarray
    T_return_C: np.ndarray
    tau_theory_s: float
    t_end_s: float
    total_flow_initial_Lps: float
    total_flow_final_Lps: float
    hydraulic_solver_converged: bool
    solver_success: bool
    solver_message: str

    @property
    def T_return_initial_C(self) -> float:
        return float(self.T_return_C[0])

    @property
    def T_return_final_C(self) -> float:
        return float(self.T_return_C[-1])

    @property
    def T_supply_final_C(self) -> float:
        return float(self.T_supply_C[-1])


def _derivative(
    T_supply_C: float,
    T_return_C: float,
    load_kW: float,
    case: LoadStepCase,
    mass_hot_kg: float,
    mass_cold_kg: float,
    hydraulic: HydraulicCase | None = None,
    secondary_flow_Lps: float | None = None,
) -> tuple[float, float, float]:
    """2노드 온도 미분 (순수 함수). 반환: (dT_supply/dt, dT_return/dt, 총유량 L/s).

    cp·ρ 는 **매 시점 1차측 벌크 평균온도에서** 다시 평가한다 — 1-B 와 같은
    규약(프로젝트정리 5-1)이며, 그래야 dT/dt = 0 인 극한이 정상상태 해와 정확히
    일치한다. ε 계산은 1-B 함수를 그대로 쓴다.

    **압력-유량도 매 시점 그 온도에서 다시 푼다**(절대 규칙 4 · 세션 3-B) —
    quasi-steady 대수방정식이므로 시간미분이 없다. 유량은 상수가 아니다.
    `solve_flow_distribution` 이 `ier != 1` 이면 예외를 던진다(절대 규칙 5).

    `hydraulic` 을 주면 그것으로 푼다(기본값은 `case.hydraulic`). 누출 스텝은
    t0 전후로 **다른 수력 케이스**를 넘겨 K값 변화를 주입한다(세션 4).

    `secondary_flow_Lps` 를 주면 ε·C_min 을 그 유량에서 유도한다(세션 5 공유
    2차측). `None` 이면 세션 4까지와 같은 경로다 — `model.hx_capacity_terms` 가
    그 분기를 한 곳에서 처리하므로 물리를 두 번 적지 않는다.
    """
    T_property_C = _property_temperature_from_state(T_supply_C, T_return_C, "bulk_mean")
    rho_kgm3 = coolant_density_kgm3(T_property_C)
    cp_Jkg_K = coolant_cp_Jkg_K(T_property_C)

    hydraulic_case = case.hydraulic if hydraulic is None else hydraulic
    flow = solve_flow_distribution(hydraulic_case, T_property_C)
    m_dot_kgs = flow.total_flow_Lps * _M3_PER_LITRE * rho_kgm3
    C_W_K = m_dot_kgs * cp_Jkg_K

    effectiveness, C_min_W_K = hx_capacity_terms(
        C_W_K,
        case.ntu,
        case.T_secondary_supply_C,
        case.secondary_flow_Lps if secondary_flow_Lps is None else secondary_flow_Lps,
    )
    Q_rack_W = load_kW * _W_PER_KW
    Q_hx_W = effectiveness * C_min_W_K * (T_return_C - case.T_secondary_supply_C)

    dT_return_dt = (C_W_K * (T_supply_C - T_return_C) + Q_rack_W) / (
        mass_hot_kg * cp_Jkg_K
    )
    dT_supply_dt = (C_W_K * (T_return_C - T_supply_C) - Q_hx_W) / (
        mass_cold_kg * cp_Jkg_K
    )
    return dT_supply_dt, dT_return_dt, flow.total_flow_Lps


def integrate_load_step(
    case: LoadStepCase, horizon_in_tau: float = INTEGRATION_HORIZON_IN_TAU
) -> TransientResult:
    """부하 스텝에 대한 온도 응답을 `solve_ivp` 로 적분한다.

    초기조건은 **스텝 직전 부하의 1-B 정상상태 해**다. 케이스마다 이 함수를 새로
    호출해 초기조건을 명시적으로 리셋한다(collaboration.md 결함유형 ④ — 시나리오
    간 상태 이월 방지). t=0 에 부하가 스텝으로 바뀐다.

    M 의 노드 배분은 **공급 50% · 환수 50%** 다 [규약: 5-1 「계통 보유수량 M의
    노드 배분」 · 세션 3 확정] — `assumptions.py` 에서 읽는다(미해결 #20 종결).

    **M 자체는 세션 2 값 그대로다**(랙 1개 회로분). 8랙에서의 계통 전체 보유량은
    5장·5-1 에 답이 없어 새로 만들지 않았다 — 모듈 docstring 참조. 방향성·비발산은
    M 의 크기와 무관하므로 게이트에는 영향이 없고, τ·t63·t95 의 절대값만 해석할
    수 없다.

    `solve_ivp` 의 `success` 와 내부 수력 `fsolve` 의 `ier` 를 **둘 다** 확인한다
    (절대 규칙 5). 수력이 실패하면 `solve_flow_distribution` 이 예외를 던진다.
    """
    initial_cdu = solve_cdu_steady_state(case.steady_case(case.load_before_percent))
    if not initial_cdu.solver_converged:
        raise RuntimeError(
            f"{case.label}: 초기 정상상태가 수렴하지 않았다 — "
            f"{initial_cdu.outer_solver_message} / "
            f"{initial_cdu.thermal.solver_message} / {initial_cdu.flow.solver_message}"
        )
    initial = initial_cdu.thermal

    mass_hot_kg = PIPING.holdup_supply_node_fraction * case.holdup.mass_kg
    mass_cold_kg = PIPING.holdup_return_node_fraction * case.holdup.mass_kg
    load_after_kW = (
        case.steady_case(case.load_after_percent).rack_load_kW
        * case.hydraulic.n_racks
    )

    tau_theory_s = case.holdup.mass_kg / initial.m_dot_kgs
    t_end_s = horizon_in_tau * tau_theory_s
    total_flow_initial_Lps = initial_cdu.flow.total_flow_Lps

    def rhs(_t: float, y: np.ndarray) -> list[float]:
        dT_supply_dt, dT_return_dt, _total_flow_Lps = _derivative(
            float(y[0]), float(y[1]), load_after_kW, case, mass_hot_kg, mass_cold_kg
        )
        return [dT_supply_dt, dT_return_dt]

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

    final_flow = solve_flow_distribution(
        case.hydraulic,
        _property_temperature_from_state(
            float(solution.y[0][-1]), float(solution.y[1][-1]), "bulk_mean"
        ),
    )
    return TransientResult(
        case=case,
        t_s=solution.t,
        T_supply_C=solution.y[0],
        T_return_C=solution.y[1],
        tau_theory_s=tau_theory_s,
        t_end_s=t_end_s,
        total_flow_initial_Lps=total_flow_initial_Lps,
        total_flow_final_Lps=final_flow.total_flow_Lps,
        hydraulic_solver_converged=(
            initial_cdu.flow.solver_converged and final_flow.solver_converged
        ),
        solver_success=bool(solution.success),
        solver_message=str(solution.message).strip(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# C5. 부하 스텝 시나리오 (5장 부하 프로파일)
# ─────────────────────────────────────────────────────────────────────────────
def default_load_step_cases() -> list[LoadStepCase]:
    """부하 스텝 4케이스 = {상승, 하강} × {M 하한, M 상한}.

    부하는 5장 부하 프로파일의 양 끝(유휴 20% ↔ 정격 100%)을 그대로 쓴다.
    **극단 케이스(부하 0 / 최대)는 이 판에서 돌리지 않는다** — 세션 3 게이트다.

    2차측 공급온도·NTU·수력 조합은 이 표에서 **범위 하단으로 고정**한다. 이 판이
    변화시키는 것은 부하와 M 이고, 나머지는 그 둘을 분리해 보기 위해 고정한 것이다
    — 대표값을 골랐다는 뜻이 아니다. 방침 (B)가 걸리는 곳(세션 2·3 게이트)은
    `tests/test_dynamics.py` 가 양 끝 조합까지 전부 돌려 판정한다.
    """
    lower, upper = holdup_bounds()
    cases: list[LoadStepCase] = []
    for holdup in (lower, upper):
        for before_percent, after_percent, direction in (
            (LOAD_PROFILE.idle_load_percent, LOAD_PROFILE.rated_load_percent, "상승"),
            (LOAD_PROFILE.rated_load_percent, LOAD_PROFILE.idle_load_percent, "하강"),
        ):
            cases.append(
                LoadStepCase(
                    label=(
                        f"{direction} {before_percent:g}→{after_percent:g}%"
                        f" · {holdup.label}"
                    ),
                    holdup=holdup,
                    load_before_percent=before_percent,
                    load_after_percent=after_percent,
                    T_secondary_supply_C=SCENARIO.T_secondary_supply_C.low,
                    ntu=HEAT_EXCHANGER.ntu.low,
                    hydraulic=default_hydraulic_cases()[0],
                )
            )
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# 누출 스텝 (세션 4) — 정상 운전 중 t0 에 K값이 계단으로 오른다
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LeakStepCase:
    """누출 주입 시나리오 1건. 케이스마다 새로 만들어 초기조건을 리셋한다.

    부하는 **바뀌지 않는다** — 정격 정상운전 중에 누출만 계단으로 들어온다.
    부하 스텝(`LoadStepCase`)과 자극이 다르므로 별도 케이스로 둔다.

    누출은 배관 K값 배율로만 근사한다(절대 규칙 8 · 5장). `k_multiplier` 가 1.0
    이면 **아무 일도 일어나지 않는 정상 케이스**이고, 같은 경로로 돈다(세션 4 C2).
    """

    label: str
    holdup: HoldupBound
    hydraulic: HydraulicCase
    k_multiplier: float
    T_secondary_supply_C: float
    ntu: float
    #: 이 CDU 의 2차측 부피유량 [L/s]. Cr 은 여기서 유도된다(세션 5-B).
    secondary_flow_Lps: float = HEAT_EXCHANGER.secondary_flow_Lps
    load_percent: float = LOAD_PROFILE.rated_load_percent
    leak_rack_index: int = LEAK.injection_rack_index
    holdup_supply_fraction: float = PIPING.holdup_supply_node_fraction

    @property
    def hydraulic_after_leak(self) -> HydraulicCase:
        """t >= t0 에서 쓰는 수력 케이스 (K값 배율이 걸린 것)."""
        return apply_leak_to_rack(
            self.hydraulic, self.k_multiplier, self.leak_rack_index
        )

    def steady_case(self, hydraulic: HydraulicCase) -> CduCase:
        """주어진 수력 케이스에서의 결합 정상상태 케이스 (물리 정의 재사용)."""
        return CduCase(
            hydraulic=hydraulic,
            T_secondary_supply_C=self.T_secondary_supply_C,
            ntu=self.ntu,
            load_percent=self.load_percent,
        )


@dataclass(frozen=True)
class LeakTransientResult:
    """누출 스텝 적분 결과 1건. solver 플래그를 함께 싣는다(절대 규칙 5)."""

    case: LeakStepCase
    t_s: np.ndarray
    T_supply_C: np.ndarray
    T_return_C: np.ndarray
    total_flow_initial_Lps: float
    total_flow_final_Lps: float
    leak_rack_flow_initial_Lps: float
    leak_rack_flow_final_Lps: float
    pump_head_initial_mAq: float
    pump_head_final_mAq: float
    tau_theory_s: float
    t_end_s: float
    hydraulic_solver_converged: bool
    solver_success: bool
    solver_message: str

    @property
    def T_return_initial_C(self) -> float:
        return float(self.T_return_C[0])

    @property
    def T_return_final_C(self) -> float:
        return float(self.T_return_C[-1])


def integrate_leak_step(
    case: LeakStepCase, horizon_in_tau: float = INTEGRATION_HORIZON_IN_TAU
) -> LeakTransientResult:
    """정격 정상운전 중 t=0 에 누출을 계단으로 주입하고 적분한다.

    초기조건은 **누출 전 수력 케이스의 결합 정상상태**다. 케이스마다 이 함수를
    새로 호출해 초기조건을 명시적으로 리셋한다(collaboration.md 결함유형 ④).

    t>=0 에서는 K값 배율이 걸린 수력 케이스로 매 시점 압력평형을 다시 푼다 —
    부하는 그대로다. `solve_ivp` 의 `success` 와 수력 `fsolve` 의 `ier` 를 **둘 다**
    확인한다(절대 규칙 5).

    **전이 시간 규모의 절대값을 해석하지 않는다** — M 결손(#21)과 8랙 해석 부재
    (#31)가 둘 다 열려 있다.
    """
    before = solve_cdu_steady_state(case.steady_case(case.hydraulic))
    if not before.solver_converged:
        raise RuntimeError(f"{case.label}: 누출 전 정상상태가 수렴하지 않았다")

    hydraulic_after = case.hydraulic_after_leak
    mass_hot_kg = case.holdup_supply_fraction * case.holdup.mass_kg
    mass_cold_kg = (1.0 - case.holdup_supply_fraction) * case.holdup.mass_kg
    load_total_kW = case.steady_case(case.hydraulic).rack_load_kW * (
        case.hydraulic.n_racks
    )

    tau_theory_s = case.holdup.mass_kg / before.thermal.m_dot_kgs
    t_end_s = horizon_in_tau * tau_theory_s

    # 부하 스텝 자리를 빌려 쓰지 않고, 이 케이스의 물리 인자만 담은 대역을 만든다.
    load_step_view = LoadStepCase(
        label=case.label,
        holdup=case.holdup,
        load_before_percent=case.load_percent,
        load_after_percent=case.load_percent,
        T_secondary_supply_C=case.T_secondary_supply_C,
        ntu=case.ntu,
        hydraulic=hydraulic_after,
        secondary_flow_Lps=case.secondary_flow_Lps,
    )

    def rhs(_t: float, y: np.ndarray) -> list[float]:
        dT_supply_dt, dT_return_dt, _flow = _derivative(
            float(y[0]),
            float(y[1]),
            load_total_kW,
            load_step_view,
            mass_hot_kg,
            mass_cold_kg,
            hydraulic_after,
        )
        return [dT_supply_dt, dT_return_dt]

    solution = solve_ivp(
        rhs,
        t_span=(0.0, t_end_s),
        y0=[before.thermal.T_supply_C, before.thermal.T_return_C],
        method="RK45",
        rtol=INTEGRATION_RTOL,
        atol=INTEGRATION_ATOL,
        dense_output=True,
        t_eval=storage_times_s(t_end_s, tau_theory_s),
    )

    final_flow = solve_flow_distribution(
        hydraulic_after,
        _property_temperature_from_state(
            float(solution.y[0][-1]), float(solution.y[1][-1]), "bulk_mean"
        ),
    )
    index = case.leak_rack_index
    return LeakTransientResult(
        case=case,
        t_s=solution.t,
        T_supply_C=solution.y[0],
        T_return_C=solution.y[1],
        total_flow_initial_Lps=before.flow.total_flow_Lps,
        total_flow_final_Lps=final_flow.total_flow_Lps,
        leak_rack_flow_initial_Lps=before.flow.rack_flows_Lps[index],
        leak_rack_flow_final_Lps=final_flow.rack_flows_Lps[index],
        pump_head_initial_mAq=before.flow.pump_head_mAq,
        pump_head_final_mAq=final_flow.pump_head_mAq,
        tau_theory_s=tau_theory_s,
        t_end_s=t_end_s,
        hydraulic_solver_converged=(
            before.flow.solver_converged and final_flow.solver_converged
        ),
        solver_success=bool(solution.success),
        solver_message=str(solution.message).strip(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# C9. 수렴시간 관측 — 판정이 아니다
# ─────────────────────────────────────────────────────────────────────────────
class _StepTrajectory(Protocol):
    """`time_to_fraction_of_step_s` 가 실제로 쓰는 필드만 담은 구조적 타입.

    부하 스텝(`TransientResult`)과 누출 스텝(`LeakTransientResult`)이 같은 관측
    함수를 쓰는데, 둘은 상속 관계가 아니라 자극이 다른 별개의 결과다. 공통 조상을
    억지로 만들지 않고 **쓰는 필드만** 프로토콜로 적는다.
    """

    # frozen dataclass 의 필드는 읽기 전용이므로 프로토콜도 읽기 전용으로 적는다.
    @property
    def t_s(self) -> np.ndarray: ...

    @property
    def T_return_C(self) -> np.ndarray: ...

    @property
    def T_return_initial_C(self) -> float: ...

    @property
    def T_return_final_C(self) -> float: ...


def time_to_fraction_of_step_s(
    result: _StepTrajectory, fraction: float
) -> float | None:
    """T_return 이 스텝 전체 변화량의 `fraction` 에 처음 도달한 시각 [s].

    **관측용이다. 6장 「수렴시간」 기준을 판정하지 않는다** — M 이 배관 보유량만
    담고 있어 과소평가돼 있기 때문이다(5-1 한계 기록). 도달하지 못하면 None.
    """
    T0 = result.T_return_initial_C
    T_end = result.T_return_final_C
    total_change_C = T_end - T0
    if total_change_C == 0.0:
        return None
    progress = (result.T_return_C - T0) / total_change_C
    reached = np.nonzero(progress >= fraction)[0]
    if reached.size == 0:
        return None
    return float(result.t_s[reached[0]])


def format_results_table(results: list[TransientResult]) -> str:
    """세션 2 결과 표 (순수 함수). 절대 규칙 11 표시를 반드시 넣는다."""
    header = (
        f"{'case':<40}{'M':>9}{'tau':>9}{'t63':>9}{'t95':>9}"
        f"{'Tret 초기':>11}{'Tret 최종':>11}{'Q 초기':>9}{'Q 최종':>9}"
        f"{'방향':>7}{'ivp':>6}{'fsol':>6}"
    )
    units = (
        f"{'':<40}{'[kg]':>9}{'[s]':>9}{'[s]':>9}{'[s]':>9}"
        f"{'[C]':>11}{'[C]':>11}{'[L/s]':>9}{'[L/s]':>9}{'':>7}{'':>6}{'':>6}"
    )
    lines = [
        "세션 3-B · 8랙 CDU · 부하 스텝에 대한 T_return 응답 (수력 매 시점 결합)",
        "※ " + ASSUMPTION_TAG,
        "※ M 은 배관 보유량만 — 열교환기·CDU 내부·콜드플레이트 보유량 제외(과소평가).",
        "   게다가 **M 은 랙 1개 회로분**이고 8랙 계통 전체 보유량은 5장·5-1 에 답이",
        "   없어 만들지 않았다. 따라서 아래 tau·t63·t95 의 **절대값은 해석하지 않는다.**",
        "",
        header,
        units,
        "-" * len(header),
    ]
    for r in results:
        t63 = time_to_fraction_of_step_s(r, 0.63)
        t95 = time_to_fraction_of_step_s(r, 0.95)
        direction = "상승" if r.T_return_final_C > r.T_return_initial_C else "하강"
        lines.append(
            f"{r.case.label:<40}"
            f"{r.case.holdup.mass_kg:>9.2f}{r.tau_theory_s:>9.2f}"
            f"{(f'{t63:.2f}' if t63 is not None else '-'):>9}"
            f"{(f'{t95:.2f}' if t95 is not None else '-'):>9}"
            f"{r.T_return_initial_C:>11.3f}{r.T_return_final_C:>11.3f}"
            f"{r.total_flow_initial_Lps:>9.4f}{r.total_flow_final_Lps:>9.4f}"
            f"{direction:>7}{('OK' if r.solver_success else 'FAIL'):>6}"
            f"{('OK' if r.hydraulic_solver_converged else 'FAIL'):>6}"
        )
    lines += [
        "-" * len(header),
        "",
        f"적분 설정: RK45 · rtol={INTEGRATION_RTOL:g} · atol={INTEGRATION_ATOL:g} · "
        f"구간 {INTEGRATION_HORIZON_IN_TAU:g}τ",
        "tau = M / m_dot (이론 체류시간)",
        "t63·t95 = 스텝 전체 변화량의 63%·95% 도달 시각",
        "2차측 공급온도·NTU·수력 조합은 이 표에서 범위 하단 고정",
        "Q 초기·최종 = 매 시점 압력평형으로 다시 푼 총유량 (상수가 아니다)",
        "",
        "※ " + ASSUMPTION_TAG,
        SESSION_3B_CAVEAT,
        SESSION_5B_CAVEAT,
        "※ 이 표가 판정하는 feasibility 기준은 **T_return 방향성**과 **비발산**이다.",
        "   energy balance 는 model.py 쪽이 보고, 수렴시간은 위 M 한계 때문에",
        "   판정하지 않는다(미해결 #21).",
    ]
    return "\n".join(lines)


def main() -> int:
    import sys

    if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔 기본 인코딩 대비
        sys.stdout.reconfigure(encoding="utf-8")
    results = [integrate_load_step(case) for case in default_load_step_cases()]
    print(format_results_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
