"""다중 CDU · 공유 2차측 결합 (세션 5).

범위 — 4장 「단계적 확장 전략」 4단계다.

    CDU 2대 ── 각자 8랙 · 자기 펌프 · 자기 열교환기
             └─ 공유 2차측 (총유량 고정 · 1차측 유량 비례 배분)

**연동의 실체는 고정된 총 2차측 유량이다** [규약: 5-1 「공유 2차측 결합 방식」].
총합이 정해져 있으므로 두 CDU 가 그것을 두고 경쟁한다 — 한쪽 부하가 오르면
온도·밀도가 변해 그쪽 1차측 유량이 이동하고, 비례 배분이 따라 이동해 **다른 쪽
CDU 의 Cr 이 바뀐다.** 그것이 다른 쪽 온도에 나타난다.

**상위 레벨 연립방정식으로 푼다.** CDU 마다 물성 온도 고정점 잔차를 하나씩 세우고
`fsolve` 로 **동시에** 0으로 만든다 — 한쪽을 풀고 다른 쪽에 넘기는 순차 방식이
아니다(세션 5 C2).

**2차측 공급온도는 고정이다** [5-1 「2차측 공급온도」 · 절대 규칙 7 의 연장].
냉각탑·드라이쿨러를 모델링하지 않는다. **한계: 그 결과 CDU 간 연동은 유량
경로로만 생기고 열 경로로는 생기지 않는다** — 연동이 작게 나오더라도 "CDU 간
상호작용이 작다"로 읽지 않는다.

**단일 CDU 경로를 지우지 않았다.** `model.solve_cdu_steady_state` 가 2차측 유량
없이 호출되면 세션 4까지와 **완전히 같은 수치**를 낸다.

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
    LOAD_PROFILE,
    PIPING,
    PLANT,
    SCENARIO,
    SESSION_5_CAVEAT,
    SESSION_5B_CAVEAT,
)
from cdu_simul.dynamics import (
    INTEGRATION_ATOL,
    INTEGRATION_HORIZON_IN_TAU,
    INTEGRATION_RTOL,
    HoldupBound,
    LoadStepCase,
    _derivative,
    holdup_bounds,
    storage_times_s,
)
from cdu_simul.hydraulics import default_cases as default_hydraulic_cases
from cdu_simul.model import (
    CduCase,
    CduSteadyStateResult,
    bulk_mean_temperature_C,
    cdu_property_temperature_residual,
    default_cdu_cases,
    energy_balance_residual_percent,
    solve_cdu_steady_state,
)


@dataclass(frozen=True)
class PlantCase:
    """CDU 여러 대와 공유 2차측 하나. 케이스마다 새로 만들어 초기조건을 리셋한다.

    CDU 대수는 `assumptions.PLANT.cdu_count` 에서 온다 — 이 파일에 숫자를 박지
    않는다. CDU 별 사양(펌프·랙 수·배관)은 **전부 같다**: 다르게 두려면 5장에
    없는 선택이 필요하다(세션 5 「하지 않을 것」).
    """

    label: str
    cdus: tuple[CduCase, ...]

    def __post_init__(self) -> None:
        if len(self.cdus) != PLANT.cdu_count:
            raise ValueError(
                f"CDU {len(self.cdus)} 대는 5-1 확정값 {PLANT.cdu_count} 대와 다르다"
            )

    @property
    def n_cdus(self) -> int:
        return len(self.cdus)


@dataclass(frozen=True)
class PlantSteadyStateResult:
    """플랜트 결합 해. 상위·하위 solver 플래그를 전부 싣는다(절대 규칙 5)."""

    case: PlantCase
    cdu_results: tuple[CduSteadyStateResult, ...]
    secondary_shares_Lps: tuple[float, ...]
    property_temps_C: tuple[float, ...]
    max_abs_residual_C: float
    top_level_solver_ier: int
    top_level_solver_message: str

    @property
    def top_level_converged(self) -> bool:
        return self.top_level_solver_ier == 1

    @property
    def solver_converged(self) -> bool:
        """상위 연립방정식과 각 CDU 의 하위 solver 가 **전부** 수렴했는가."""
        return self.top_level_converged and all(
            r.solver_converged for r in self.cdu_results
        )

    @property
    def secondary_share_fractions(self) -> tuple[float, ...]:
        total = sum(self.secondary_shares_Lps)
        return tuple(share / total for share in self.secondary_shares_Lps)

    def heat_capacity_ratios(self) -> tuple[float, ...]:
        """각 CDU 의 실제 Cr = C_min/C_max (유량에서 유도된 값)."""
        from cdu_simul.fluid import coolant_cp_Jkg_K, coolant_density_kgm3

        ratios = []
        for result, share_Lps in zip(
            self.cdu_results, self.secondary_shares_Lps, strict=True
        ):
            T_2nd_C = result.case.T_secondary_supply_C
            C_secondary = (
                share_Lps
                * 1.0e-3
                * coolant_density_kgm3(T_2nd_C)
                * coolant_cp_Jkg_K(T_2nd_C)
            )
            C_primary = result.thermal.m_dot_kgs * result.thermal.cp_Jkg_K
            ratios.append(
                min(C_primary, C_secondary) / max(C_primary, C_secondary)
            )
        return tuple(ratios)


def _primary_flows_at(
    case: PlantCase, property_temps_C: tuple[float, ...]
) -> tuple[float, ...]:
    """각 CDU 의 1차측 총유량 [L/s] — 그 시점의 물성 온도에서 quasi-steady."""
    from cdu_simul.hydraulics import solve_flow_distribution

    return tuple(
        solve_flow_distribution(cdu.hydraulic, T_C).total_flow_Lps
        for cdu, T_C in zip(case.cdus, property_temps_C, strict=True)
    )


def solve_plant_steady_state(case: PlantCase) -> PlantSteadyStateResult:
    """상위 레벨 연립방정식으로 CDU 간 연동을 푼다.

    미지수는 **CDU 마다 물성 평가온도 하나**다. 잔차도 CDU 마다 하나이므로
    n×n 연립이 된다::

        for i:  규칙온도_i(T_1..T_n) - T_i = 0

    잔차 하나를 계산하려면 **모든** CDU 의 1차측 유량이 필요하다 — 공유 2차측
    배분이 총유량 비례이기 때문이다. 그래서 이 식들은 분리되지 않고, `fsolve` 가
    **동시에** 푼다(순차 대입이 아니다).

    각 잔차 안에서는 그 CDU 의 헤더 압력평형이 quasi-steady 로 다시 풀린다
    (절대 규칙 4). 즉 3중 구조다: 상위 연립 → CDU 물성 온도 → 랙 유량 분배.

    초기값은 5장 1차측 공급·환수의 산술평균이다 — 출발점일 뿐 해를 정하지 않는다.
    **케이스마다 이 함수를 새로 호출해 초기조건을 명시적으로 리셋한다**
    (collaboration.md 결함유형 ④).

    절대 규칙 5: 상위 `fsolve` 의 `ier` 를 확인해 결과에 싣고, 하위 solver 플래그도
    함께 싣는다. 수력이 실패하면 `solve_flow_distribution` 이 예외를 던진다.
    """

    def residuals(x: np.ndarray) -> np.ndarray:
        temps = tuple(float(value) for value in x)
        shares = PLANT.secondary_shares_Lps(_primary_flows_at(case, temps))
        return np.array(
            [
                cdu_property_temperature_residual(cdu, T_C, share)
                for cdu, T_C, share in zip(case.cdus, temps, shares, strict=True)
            ]
        )

    initial_guess_C = bulk_mean_temperature_C(
        SCENARIO.T_primary_supply_C, SCENARIO.T_primary_return_C
    )
    solution, _info, ier, message = fsolve(
        residuals, np.full(case.n_cdus, initial_guess_C), full_output=True
    )

    property_temps_C = tuple(float(value) for value in solution)
    shares = PLANT.secondary_shares_Lps(_primary_flows_at(case, property_temps_C))
    return PlantSteadyStateResult(
        case=case,
        cdu_results=tuple(
            solve_cdu_steady_state(cdu, share)
            for cdu, share in zip(case.cdus, shares, strict=True)
        ),
        secondary_shares_Lps=shares,
        property_temps_C=property_temps_C,
        max_abs_residual_C=float(np.max(np.abs(residuals(solution)))),
        top_level_solver_ier=int(ier),
        top_level_solver_message=str(message).strip(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 부하 시나리오 — 5장 부하 프로파일(20~100%)을 지렛대로 쓴다
# ─────────────────────────────────────────────────────────────────────────────
def plant_case(
    label: str, load_percents: tuple[float, ...], template: CduCase
) -> PlantCase:
    """같은 사양의 CDU 여러 대를 부하율만 다르게 만든다.

    **대칭 경로다** — 전 CDU 가 같은 템플릿을 받는다. CDU 마다 다른 조건(예: 한
    CDU 에만 누출)이 필요하면 `plant_case_from_templates` 를 쓴다. 이 함수는 그
    함수에 같은 템플릿을 n 벌 넘기는 것과 **완전히 같다**(세션 5.5-D).
    """
    return plant_case_from_templates(
        label, load_percents, (template,) * len(load_percents)
    )


def plant_case_from_templates(
    label: str,
    load_percents: tuple[float, ...],
    templates: tuple[CduCase, ...],
) -> PlantCase:
    """CDU 마다 **다른 템플릿**을 받아 부하율만 덮어쓴다.

    필요해진 이유 [세션 5.5-D · 미해결 #34]: 다중 CDU 전이에서 누출을 **한 CDU 의
    랙 1개**에만 걸어야 한다. 5-1 「누출 주입 지점」이 "랙 1개에 주입한다 · 전 랙
    동시 누출은 5장에 근거가 없어 돌리지 않는다" 이므로, 두 CDU 의 랙 0 에 동시에
    거는 것은 그 규정 밖이다. 정상상태 경로(`dataset.steady_rows`)는 이미 CDU 0
    에만 걸고 있었고, 전이 경로만 걸지 못하고 있었다.

    **CDU 별 사양(펌프·랙 수·배관 규격)을 다르게 두라는 뜻이 아니다** — 그것은
    5장에 없는 선택이다(세션 5 「하지 않을 것」). 여기서 갈리는 것은 5장·5-1 이
    이미 정의한 시나리오 조건(누출 K값 배율)뿐이다.
    """
    from dataclasses import replace

    return PlantCase(
        label=label,
        cdus=tuple(
            replace(template, load_percent=percent)
            for template, percent in zip(templates, load_percents, strict=True)
        ),
    )


def default_load_scenarios() -> tuple[tuple[str, tuple[float, ...]], ...]:
    """대칭 2 + 비대칭 1. 부하율은 5장 부하 프로파일 양 끝만 쓴다.

    중간값을 만들지 않는다 — 5장이 주는 것은 유휴 20% 와 정격 100% 다.
    """
    idle = LOAD_PROFILE.idle_load_percent
    rated = LOAD_PROFILE.rated_load_percent
    return (
        ("대칭 100/100%", (rated, rated)),
        ("대칭 20/20%", (idle, idle)),
        ("비대칭 100/20%", (rated, idle)),
    )


def default_plant_cases() -> list[PlantCase]:
    """5장·5-1 범위 양 끝 32조합 × 부하 시나리오 3 = 96 케이스."""
    return [
        plant_case(f"{template.hydraulic.label} / NTU={template.ntu:g}"
                   f" / T2nd={template.T_secondary_supply_C:g}C / {name}",
                   percents, template)
        for template in default_cdu_cases()
        for name, percents in default_load_scenarios()
    ]


def format_plant_table(results: list[PlantSteadyStateResult]) -> str:
    """부하 시나리오별 결합 결과 표 (순수 함수).

    32조합 전부를 적으면 96행이 되므로, **수력·NTU·2차측을 범위 하단으로 고정한
    3케이스**를 낸다. 게이트(수렴)는 96 케이스 전수를 테스트가 판정한다.

    절대 규칙 11: 산출물에 "가정값 기반 — 실측 아님" 표시를 반드시 넣는다.
    """
    header = (
        f"{'시나리오':<18}{'CDU':>5}{'부하':>7}{'1차 Q':>10}{'2차 배분':>10}"
        f"{'배분비':>9}{'Cr':>11}{'T_sup':>9}{'T_ret':>9}{'balance':>11}{'solver':>8}"
    )
    units = (
        f"{'':<18}{'':>5}{'[%]':>7}{'[L/s]':>10}{'[L/s]':>10}{'[%]':>9}{'[-]':>11}"
        f"{'[C]':>9}{'[C]':>9}{'[%]':>11}{'':>8}"
    )
    lines = [
        f"세션 5 · CDU {PLANT.cdu_count}대 공유 2차측 결합 정상상태",
        "※ " + ASSUMPTION_TAG,
        f"총 2차측 유량 {PLANT.secondary_total_flow_Lps:g} L/s 고정"
        " · 1차측 유량 비례 배분",
        "",
        header,
        units,
        "-" * len(header),
    ]
    for result in results:
        ratios = result.heat_capacity_ratios()
        fractions = result.secondary_share_fractions
        for index, cdu_result in enumerate(result.cdu_results):
            lines.append(
                f"{(result.case.label if index == 0 else ''):<18}"
                f"{'AB'[index]:>5}{cdu_result.case.load_percent:>7.0f}"
                f"{cdu_result.flow.total_flow_Lps:>10.4f}"
                f"{result.secondary_shares_Lps[index]:>10.4f}"
                f"{fractions[index] * 100.0:>9.3f}{ratios[index]:>11.7f}"
                f"{cdu_result.thermal.T_supply_C:>9.4f}"
                f"{cdu_result.thermal.T_return_C:>9.4f}"
                f"{energy_balance_residual_percent(cdu_result.thermal):>11.5f}"
                f"{('OK' if result.solver_converged else 'FAIL'):>8}"
            )
    lines += [
        "-" * len(header),
        "",
        "수력 조합·NTU·2차측 공급온도는 이 표에서 범위 하단 고정.",
        "Cr = C_min/C_max — 배분된 2차측 유량에서 유도된 값이다",
        "  (5장의 명목 1:1 이 아니다).",
        "상위 연립 fsolve 최대 잔차 = "
        + " · ".join(f"{r.max_abs_residual_C:.2e} K" for r in results),
        "",
        "※ " + ASSUMPTION_TAG,
        SESSION_5_CAVEAT,
        SESSION_5B_CAVEAT,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 동적 결합 (C6) — 비대칭 부하 스텝에서 다른 CDU 가 반응하는가
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PlantLoadStepCase:
    """한 CDU 만 부하가 계단으로 바뀌는 시나리오. 나머지 CDU 는 부하 불변이다.

    **연동이 있으면 부하가 그대로인 CDU 도 움직인다** — 공유 2차측 배분이
    이동하기 때문이다. 그것을 보는 것이 이 케이스의 목적이다.

    **CDU 별 템플릿** [세션 5.5-D]: `templates` 를 주면 CDU 마다 다른 조건(누출
    K값 배율)을 건다. 주지 않으면 `template` 을 전 CDU 에 쓴다 — **기존 대칭
    경로가 그대로 남아 있고, 같은 템플릿을 n 벌 주는 것과 완전히 같다.**
    """

    label: str
    holdup: HoldupBound
    template: CduCase
    load_before_percents: tuple[float, ...]
    load_after_percents: tuple[float, ...]
    holdup_supply_fraction: float = PIPING.holdup_supply_node_fraction
    #: CDU 별 템플릿. `None` 이면 `template` 을 전 CDU 에 쓴다(대칭 경로).
    templates: tuple[CduCase, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.load_before_percents) != len(self.load_after_percents):
            raise ValueError("스텝 전후 부하 목록의 길이가 다르다")
        if self.templates is not None and len(self.templates) != len(
            self.load_after_percents
        ):
            raise ValueError("CDU 별 템플릿 수가 부하 목록 길이와 다르다")

    def cdu_templates(self) -> tuple[CduCase, ...]:
        """CDU 별 템플릿 — 주어지지 않았으면 `template` 을 복제한다."""
        if self.templates is None:
            return (self.template,) * len(self.load_after_percents)
        return self.templates

    def plant_at(self, load_percents: tuple[float, ...]) -> PlantCase:
        return plant_case_from_templates(
            self.label, load_percents, self.cdu_templates()
        )


@dataclass(frozen=True)
class PlantTransientResult:
    """플랜트 시간적분 결과. 하위 solver 플래그를 함께 싣는다(절대 규칙 5)."""

    case: PlantLoadStepCase
    t_s: np.ndarray
    T_supply_C: tuple[np.ndarray, ...]
    T_return_C: tuple[np.ndarray, ...]
    tau_theory_s: float
    t_end_s: float
    secondary_shares_initial_Lps: tuple[float, ...]
    secondary_shares_final_Lps: tuple[float, ...]
    hydraulic_solver_converged: bool
    solver_success: bool
    solver_message: str

    def T_return_change_C(self, index: int) -> float:
        return float(self.T_return_C[index][-1] - self.T_return_C[index][0])


def integrate_plant_load_step(
    case: PlantLoadStepCase,
    horizon_in_tau: float = INTEGRATION_HORIZON_IN_TAU,
) -> PlantTransientResult:
    """비대칭 부하 스텝을 적분한다 — 상태는 CDU 마다 2노드다.

    **매 시점** 각 CDU 의 헤더 압력평형을 quasi-steady 로 풀고(절대 규칙 4),
    그 유량으로 공유 2차측을 다시 배분한다. 배분이 시간에 따라 움직이는 것이
    연동의 경로다.

    초기조건은 스텝 직전 부하의 **플랜트 결합 정상상태**다. 케이스마다 이 함수를
    새로 호출해 초기조건을 명시적으로 리셋한다(collaboration.md 결함유형 ④).

    `solve_ivp` 의 `success` 와 매 시점 수력 `fsolve` 의 수렴을 **둘 다** 본다.
    **전이 시간 규모의 절대값은 해석하지 않는다**(#21 · #31).
    """
    before = solve_plant_steady_state(case.plant_at(case.load_before_percents))
    if not before.solver_converged:
        raise RuntimeError(f"{case.label}: 스텝 전 결합 정상상태가 수렴하지 않았다")

    after_plant = case.plant_at(case.load_after_percents)
    n = after_plant.n_cdus
    mass_hot_kg = case.holdup_supply_fraction * case.holdup.mass_kg
    mass_cold_kg = (1.0 - case.holdup_supply_fraction) * case.holdup.mass_kg
    loads_total_kW = tuple(
        cdu.rack_load_kW * cdu.hydraulic.n_racks for cdu in after_plant.cdus
    )

    tau_theory_s = case.holdup.mass_kg / before.cdu_results[0].thermal.m_dot_kgs
    t_end_s = horizon_in_tau * tau_theory_s

    def rhs(_t: float, y: np.ndarray) -> list[float]:
        temps = tuple(
            bulk_mean_temperature_C(float(y[2 * i]), float(y[2 * i + 1]))
            for i in range(n)
        )
        shares = PLANT.secondary_shares_Lps(_primary_flows_at(after_plant, temps))
        derivatives: list[float] = []
        for i, cdu in enumerate(after_plant.cdus):
            view = LoadStepCase(
                label=f"{case.label}#{i}",
                holdup=case.holdup,
                load_before_percent=cdu.load_percent,
                load_after_percent=cdu.load_percent,
                T_secondary_supply_C=cdu.T_secondary_supply_C,
                ntu=cdu.ntu,
                hydraulic=cdu.hydraulic,
                secondary_flow_Lps=shares[i],
            )
            dT_supply, dT_return, _flow = _derivative(
                float(y[2 * i]),
                float(y[2 * i + 1]),
                loads_total_kW[i],
                view,
                mass_hot_kg,
                mass_cold_kg,
                cdu.hydraulic,
                shares[i],
            )
            derivatives += [dT_supply, dT_return]
        return derivatives

    y0: list[float] = []
    for result in before.cdu_results:
        y0 += [result.thermal.T_supply_C, result.thermal.T_return_C]

    solution = solve_ivp(
        rhs,
        t_span=(0.0, t_end_s),
        y0=y0,
        method="RK45",
        rtol=INTEGRATION_RTOL,
        atol=INTEGRATION_ATOL,
        dense_output=True,
        # **저장 격자만 바꿨다 — 적분 설정(RK45·rtol·atol·구간)은 그대로다**
        # [세션 5.5-D · C3]. 종전에는 균등 2001점을 썼는데 `dynamics` 의 단일 CDU
        # 경로가 쓰는 `storage_times_s`(스텝 직후 조밀한 비균등 201점)와 격자가
        # 달라, `dataset.py` 가 뒤에서 가장 가까운 점을 골라 맞추고 있었다.
        # 여기서 같은 격자를 쓰면 그 보정이 필요 없어진다.
        # `t_eval` 은 `solve_ivp` 의 스텝 선택에 영향하지 않는다 — dense output 을
        # 어느 시각에서 평가할지만 정한다. 그래서 해 자체는 이동하지 않는다.
        t_eval=storage_times_s(t_end_s, tau_theory_s),
    )

    final_temps = tuple(
        bulk_mean_temperature_C(
            float(solution.y[2 * i][-1]), float(solution.y[2 * i + 1][-1])
        )
        for i in range(n)
    )
    final_shares = PLANT.secondary_shares_Lps(
        _primary_flows_at(after_plant, final_temps)
    )
    return PlantTransientResult(
        case=case,
        t_s=solution.t,
        T_supply_C=tuple(solution.y[2 * i] for i in range(n)),
        T_return_C=tuple(solution.y[2 * i + 1] for i in range(n)),
        tau_theory_s=tau_theory_s,
        t_end_s=t_end_s,
        secondary_shares_initial_Lps=before.secondary_shares_Lps,
        secondary_shares_final_Lps=final_shares,
        hydraulic_solver_converged=all(
            r.flow.solver_converged for r in before.cdu_results
        ),
        solver_success=bool(solution.success),
        solver_message=str(solution.message).strip(),
    )


def format_plant_transient_table(results: list[PlantTransientResult]) -> str:
    """비대칭 부하 스텝 결과 표 (순수 함수). 절대 규칙 11 표시를 넣는다."""
    header = (
        f"{'case':<34}{'CDU':>5}{'부하':>11}{'Tret 초기':>11}{'Tret 최종':>11}"
        f"{'ΔTret':>11}{'2차 배분 초기':>15}{'2차 배분 최종':>15}{'ivp':>6}{'fsol':>6}"
    )
    lines = [
        "세션 5 · 비대칭 부하 스텝에서의 CDU 간 연동 (동적)",
        "※ " + ASSUMPTION_TAG,
        "※ tau·전이 시간의 절대값은 해석하지 않는다 — #21 · #31.",
        "",
        header,
        "-" * len(header),
    ]
    for r in results:
        for i in range(len(r.T_return_C)):
            before_pct = r.case.load_before_percents[i]
            after_pct = r.case.load_after_percents[i]
            lines.append(
                f"{(r.case.label if i == 0 else ''):<34}"
                f"{'AB'[i]:>5}{f'{before_pct:g}→{after_pct:g}%':>11}"
                f"{float(r.T_return_C[i][0]):>11.5f}"
                f"{float(r.T_return_C[i][-1]):>11.5f}"
                f"{r.T_return_change_C(i):>11.5f}"
                f"{r.secondary_shares_initial_Lps[i]:>15.5f}"
                f"{r.secondary_shares_final_Lps[i]:>15.5f}"
                f"{('OK' if r.solver_success else 'FAIL'):>6}"
                f"{('OK' if r.hydraulic_solver_converged else 'FAIL'):>6}"
            )
    lines += [
        "-" * len(header),
        "",
        "부하가 바뀌지 않은 CDU 의 ΔTret 가 **연동의 크기**다.",
        "",
        "※ " + ASSUMPTION_TAG,
        SESSION_5_CAVEAT,
        SESSION_5B_CAVEAT,
    ]
    return "\n".join(lines)


def default_plant_load_step_cases() -> list[PlantLoadStepCase]:
    """비대칭 스텝 — A 만 20→100%. M 하한·상한 둘 다."""
    template = CduCase(
        hydraulic=default_hydraulic_cases()[0],
        T_secondary_supply_C=SCENARIO.T_secondary_supply_C.low,
        ntu=HEAT_EXCHANGER.ntu.low,
    )
    idle = LOAD_PROFILE.idle_load_percent
    rated = LOAD_PROFILE.rated_load_percent
    return [
        PlantLoadStepCase(
            label=f"A만 {idle:g}→{rated:g}% · {holdup.label}",
            holdup=holdup,
            template=template,
            load_before_percents=(idle, idle),
            load_after_percents=(rated, idle),
        )
        for holdup in holdup_bounds()
    ]


def main() -> int:
    import sys

    if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔 기본 인코딩 대비
        sys.stdout.reconfigure(encoding="utf-8")
    template = CduCase(
        hydraulic=default_hydraulic_cases()[0],
        T_secondary_supply_C=SCENARIO.T_secondary_supply_C.low,
        ntu=HEAT_EXCHANGER.ntu.low,
    )
    results = [
        solve_plant_steady_state(plant_case(name, percents, template))
        for name, percents in default_load_scenarios()
    ]
    print(format_plant_table(results))
    print()
    print(
        format_plant_transient_table(
            [
                integrate_plant_load_step(case)
                for case in default_plant_load_step_cases()
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
