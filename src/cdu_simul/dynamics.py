"""온도 시간적분 모델 (세션 2) — 부하 스텝에 대한 T_return 응답.

범위 — 1-B 정상상태 모델에 **시간축만** 얹는다.

**압력-유량을 풀지 않는다.** 유량은 5장 정격 고정이다. 단일 랙이라 분배할 곳이
없다 — 이것은 하이브리드 구조(절대 규칙 4)를 없애는 것이 아니라 **아직 만들지
않는 것**이며, `fsolve` 기반 압력평형은 세션 3에서 들어온다. 절대 규칙 4가 요구하는
"압력은 quasi-steady 대수, 온도는 시간적분"에서 이 판은 **온도 쪽만** 세운다.

**2차측 동특성을 만들지 않는다**(절대 규칙 7). 2차측 공급온도는 고정 경계조건이다.

**1-B `model.py` 를 수정하지 않는다.** ε 계산·cp 평가 규칙·정상상태 해는 전부
`model` 에서 가져다 쓴다 — 같은 물리를 두 번 적지 않는다(collaboration.md ④).

노드 구성 (2노드 루프):

    ┌── 랙 (Q_rack 유입) ──┐
    │                      ↓
  T_supply 노드        T_return 노드
    ↑                      │
    └── 열교환기 (Q_hx 유출) ┘

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

import numpy as np
from scipy.integrate import solve_ivp

from cdu_simul.assumptions import (
    ASSUMPTION_TAG,
    HEAT_EXCHANGER,
    LOAD_PROFILE,
    PIPING,
    SCENARIO,
    VALVE,
)
from cdu_simul.fluid import coolant_cp_Jkg_K, coolant_density_kgm3
from cdu_simul.model import (
    SteadyStateCase,
    _property_temperature_from_state,
    hx_effectiveness_counterflow,
    solve_steady_state,
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

#: 적분 구간을 이론 체류시간 τ 의 몇 배까지 잡을지. **수치 설정**이다.
#: 가장 느린 모드의 시간상수가 τ/ε (ε<1) 이므로 30τ 는 최소 20 시간상수에
#: 해당한다 — 잔여 과도분이 e^-20 ≈ 2e-9 배로 줄어 t→∞ 대조에 영향을 주지 않는다.
INTEGRATION_HORIZON_IN_TAU: float = 30.0


# ─────────────────────────────────────────────────────────────────────────────
# C3. 계통 보유수량(열용량) — 순수 함수
# ─────────────────────────────────────────────────────────────────────────────
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
    heat_capacity_ratio: float
    rack_load_kW: float
    rack_flow_Lps: float

    def steady_case(self, load_percent: float) -> SteadyStateCase:
        """주어진 부하율에서의 1-B 정상상태 케이스를 만든다 (물리 정의 재사용)."""
        return SteadyStateCase(
            T_secondary_supply_C=self.T_secondary_supply_C,
            ntu=self.ntu,
            rack_load_kW=self.rack_load_kW * load_percent * _PERCENT,
            rack_flow_Lps=self.rack_flow_Lps,
            heat_capacity_ratio=self.heat_capacity_ratio,
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
) -> tuple[float, float]:
    """2노드 온도 미분 (순수 함수). 반환: (dT_supply/dt, dT_return/dt) [K/s].

    cp·ρ 는 **매 시점 1차측 벌크 평균온도에서** 다시 평가한다 — 1-B 와 같은
    규약(프로젝트정리 5-1)이며, 그래야 dT/dt = 0 인 극한이 1-B 해와 정확히
    일치한다. ε 계산은 1-B 함수를 그대로 쓴다.
    """
    T_property_C = _property_temperature_from_state(T_supply_C, T_return_C, "bulk_mean")
    rho_kgm3 = coolant_density_kgm3(T_property_C)
    cp_Jkg_K = coolant_cp_Jkg_K(T_property_C)

    m_dot_kgs = case.rack_flow_Lps * _M3_PER_LITRE * rho_kgm3
    C_W_K = m_dot_kgs * cp_Jkg_K

    effectiveness = hx_effectiveness_counterflow(case.ntu, case.heat_capacity_ratio)
    Q_rack_W = load_kW * _W_PER_KW
    Q_hx_W = effectiveness * C_W_K * (T_return_C - case.T_secondary_supply_C)

    dT_return_dt = (C_W_K * (T_supply_C - T_return_C) + Q_rack_W) / (
        mass_hot_kg * cp_Jkg_K
    )
    dT_supply_dt = (C_W_K * (T_return_C - T_supply_C) - Q_hx_W) / (
        mass_cold_kg * cp_Jkg_K
    )
    return dT_supply_dt, dT_return_dt


def integrate_load_step(
    case: LoadStepCase, horizon_in_tau: float = INTEGRATION_HORIZON_IN_TAU
) -> TransientResult:
    """부하 스텝에 대한 온도 응답을 `solve_ivp` 로 적분한다.

    초기조건은 **스텝 직전 부하의 1-B 정상상태 해**다. 케이스마다 이 함수를 새로
    호출해 초기조건을 명시적으로 리셋한다(collaboration.md 결함유형 ④ — 시나리오
    간 상태 이월 방지). t=0 에 부하가 스텝으로 바뀐다.

    M 을 두 노드에 절반씩 나눈다: 5장 등가길이를 **왕복 전체**로 읽으라는 5-1
    규칙에서 공급측 구간과 환수측 구간이 그 왕복의 두 다리이기 때문이다. 배분
    비율 자체는 5장에도 5-1 에도 없다 — 미해결 목록에 올린다.

    `solve_ivp` 의 `success` 를 확인해 결과에 실어 보낸다(절대 규칙 5).
    """
    initial = solve_steady_state(case.steady_case(case.load_before_percent))
    if not initial.solver_converged:
        raise RuntimeError(
            f"{case.label}: 초기 정상상태가 수렴하지 않았다 — {initial.solver_message}"
        )

    mass_hot_kg = 0.5 * case.holdup.mass_kg
    mass_cold_kg = 0.5 * case.holdup.mass_kg
    load_after_kW = case.rack_load_kW * case.load_after_percent * _PERCENT

    tau_theory_s = case.holdup.mass_kg / initial.m_dot_kgs
    t_end_s = horizon_in_tau * tau_theory_s

    def rhs(_t: float, y: np.ndarray) -> list[float]:
        dT_supply_dt, dT_return_dt = _derivative(
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
        t_eval=np.linspace(0.0, t_end_s, 4001),
    )

    return TransientResult(
        case=case,
        t_s=solution.t,
        T_supply_C=solution.y[0],
        T_return_C=solution.y[1],
        tau_theory_s=tau_theory_s,
        t_end_s=t_end_s,
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

    2차측 공급온도·NTU 는 이 표에서 **범위 하단으로 고정**한다. 이 판이 변화시키는
    것은 부하와 M 이고, 열교환 조건은 그 둘을 분리해 보기 위해 고정한 것이다 —
    대표값을 골랐다는 뜻이 아니다. 방침 (B)가 걸리는 곳(세션 2 게이트)은
    `tests/test_dynamics.py` 가 2차측·NTU 양 끝 조합까지 전부 돌려 판정한다.
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
                    heat_capacity_ratio=(
                        HEAT_EXCHANGER.flow_ratio_primary_to_secondary
                    ),
                    rack_load_kW=SCENARIO.rack_it_load_kW,
                    rack_flow_Lps=VALVE.rated_flow_per_rack_Lps,
                )
            )
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# C9. 수렴시간 관측 — 판정이 아니다
# ─────────────────────────────────────────────────────────────────────────────
def time_to_fraction_of_step_s(result: TransientResult, fraction: float) -> float | None:
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
        f"{'Tret 초기':>11}{'Tret 최종':>11}{'방향':>7}{'ivp':>6}"
    )
    units = (
        f"{'':<40}{'[kg]':>9}{'[s]':>9}{'[s]':>9}{'[s]':>9}"
        f"{'[C]':>11}{'[C]':>11}{'':>7}{'':>6}"
    )
    lines = [
        "세션 2 · 부하 스텝에 대한 T_return 응답",
        "※ " + ASSUMPTION_TAG,
        "※ M 은 배관 보유량만 — 열교환기·CDU 내부·콜드플레이트 보유량 제외(과소평가).",
        "   따라서 아래 수렴시간은 실제보다 짧다. '합리적'이라고 판정하지 않는다.",
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
            f"{direction:>7}{('OK' if r.solver_success else 'FAIL'):>6}"
        )
    lines += [
        "-" * len(header),
        "",
        f"적분 설정: RK45 · rtol={INTEGRATION_RTOL:g} · atol={INTEGRATION_ATOL:g} · "
        f"구간 {INTEGRATION_HORIZON_IN_TAU:g}τ",
        "tau = M / m_dot (이론 체류시간)",
        "t63·t95 = 스텝 전체 변화량의 63%·95% 도달 시각",
        "2차측 공급온도·NTU 는 이 표에서 범위 하단 고정 (부하와 M 만 변화시킨다)",
        "",
        "※ " + ASSUMPTION_TAG,
        "※ 이 표가 판정하는 feasibility 기준은 **T_return 방향성 하나**다.",
        "   energy balance 는 세션 1-B에서 봤고, 수렴시간은 위 M 한계 때문에",
        "   판정하지 않으며, 극단 케이스(부하 0/최대)는 세션 3이다.",
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
