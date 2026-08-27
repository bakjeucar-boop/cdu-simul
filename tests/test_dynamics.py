"""세션 2 게이트 — T_return 응답 방향성 + 정상상태 대조.

CLAUDE.md 절대 규칙 10: 사람이 눈으로 보지 않고 **테스트가 판정한다.**

**이 파일이 판정하는 feasibility 기준은 T_return 방향성 하나다.**
energy balance 는 세션 1-B 에서 봤고, 수렴시간은 M 이 배관 보유량만 담고 있어
과소평가된 상태라 판정하지 않으며(프로젝트정리 5-1 한계 기록), 극단 케이스
(부하 0/최대)는 세션 3 게이트다. 그 테스트들을 미리 만들지 않는다.

정상상태 대조(`test_transient_limit_matches_steady_state`)는 feasibility 기준이
아니라 **구현 일관성 검사**다 — 아래 그 테스트의 docstring 참조.
"""

from __future__ import annotations

import pytest

from cdu_simul.assumptions import HEAT_EXCHANGER, LOAD_PROFILE, SCENARIO, VALVE
from cdu_simul.dynamics import (
    HoldupBound,
    LoadStepCase,
    holdup_bounds,
    integrate_load_step,
)
from cdu_simul.model import SteadyStateCase, default_cases, solve_steady_state

#: 정상상태 대조 허용오차 [K].
#: 적분 허용오차(rtol 1e-10)에서 오는 수치 잡음은 실측 4.5e-9 K 수준이다.
#: 1e-6 K 는 그 잡음보다 200배 이상 느슨하면서, 두 경로가 실제로 어긋났을 때
#: 나타날 크기(4케이스의 온도 폭이 ~3 K 다)보다는 6자리 이상 엄격하다.
STEADY_STATE_TOLERANCE_C: float = 1.0e-6


def _direction_gate_cases() -> list[LoadStepCase]:
    """방향성 게이트용 케이스 — 5장 범위값의 **양 끝 조합 전부**.

    {상승, 하강} × {M 하한, 상한} × {2차측 27, 30℃} × {NTU 2, 3} = 16 케이스.
    결과 표(dynamics.format_results_table)는 2차측·NTU 를 하단으로 고정하지만,
    **게이트는 고정하지 않는다** — 방침 (B)(양 끝을 둘 다 돌린다)가 걸리는 곳은
    판정이기 때문이다.
    """
    lower, upper = holdup_bounds()
    cases: list[LoadStepCase] = []
    for holdup in (lower, upper):
        for T_secondary_C in (
            SCENARIO.T_secondary_supply_C.low,
            SCENARIO.T_secondary_supply_C.high,
        ):
            for ntu in (HEAT_EXCHANGER.ntu.low, HEAT_EXCHANGER.ntu.high):
                for before_percent, after_percent, direction in (
                    (
                        LOAD_PROFILE.idle_load_percent,
                        LOAD_PROFILE.rated_load_percent,
                        "상승",
                    ),
                    (
                        LOAD_PROFILE.rated_load_percent,
                        LOAD_PROFILE.idle_load_percent,
                        "하강",
                    ),
                ):
                    cases.append(
                        LoadStepCase(
                            label=(
                                f"{direction} · M={holdup.mass_kg:.0f}kg · "
                                f"T2nd={T_secondary_C:g}C · NTU={ntu:g}"
                            ),
                            holdup=holdup,
                            load_before_percent=before_percent,
                            load_after_percent=after_percent,
                            T_secondary_supply_C=T_secondary_C,
                            ntu=ntu,
                            heat_capacity_ratio=(
                                HEAT_EXCHANGER.flow_ratio_primary_to_secondary
                            ),
                            rack_load_kW=SCENARIO.rack_it_load_kW,
                            rack_flow_Lps=VALVE.rated_flow_per_rack_Lps,
                        )
                    )
    return cases


def _case_id(case: LoadStepCase) -> str:
    return case.label


DIRECTION_GATE_CASES = _direction_gate_cases()


@pytest.mark.parametrize("case", DIRECTION_GATE_CASES, ids=_case_id)
def test_solve_ivp_succeeded(case: LoadStepCase) -> None:
    """`solve_ivp` 의 성공 플래그를 확인한다 (절대 규칙 5).

    실패를 조용히 넘기고 마지막 값을 쓰지 않는다.
    """
    result = integrate_load_step(case)
    assert result.solver_success, (
        f"{case.label}: solve_ivp 가 실패했다 — {result.solver_message}"
    )


@pytest.mark.parametrize("case", DIRECTION_GATE_CASES, ids=_case_id)
def test_T_return_responds_in_physically_sound_direction(case: LoadStepCase) -> None:
    """**세션 2 게이트** — 부하 스텝에 T_return 이 타당한 방향으로 반응하는가.

    프로젝트정리 6장: "부하 증가 시 T_return 이 물리적으로 타당한 방향(상승)으로
    반응하는지". 부하 하강 스텝에서는 반대 방향이어야 한다.

    두 가지를 함께 본다 — 스텝 직후의 **반응 방향**(초기 기울기 부호)과
    적분 끝에서의 **순변화 방향**. 둘 중 하나만 맞으면 통과시키지 않는다.
    """
    result = integrate_load_step(case)
    assert result.solver_success, f"{case.label}: solve_ivp 실패 — 방향성 판정 불가"

    net_change_C = result.T_return_final_C - result.T_return_initial_C
    initial_slope_C = float(result.T_return_C[1] - result.T_return_C[0])

    if case.is_rising_step:
        assert net_change_C > 0.0, (
            f"{case.label}: 부하가 올랐는데 T_return 순변화가 {net_change_C:+.4f} K 다"
        )
        assert initial_slope_C > 0.0, (
            f"{case.label}: 부하가 올랐는데 스텝 직후 기울기가 "
            f"{initial_slope_C:+.3e} K 다"
        )
    else:
        assert net_change_C < 0.0, (
            f"{case.label}: 부하가 내렸는데 T_return 순변화가 {net_change_C:+.4f} K 다"
        )
        assert initial_slope_C < 0.0, (
            f"{case.label}: 부하가 내렸는데 스텝 직후 기울기가 "
            f"{initial_slope_C:+.3e} K 다"
        )


@pytest.mark.parametrize("steady_case", default_cases(), ids=lambda c: c.label)
def test_transient_limit_matches_steady_state(steady_case: SteadyStateCase) -> None:
    """t→∞ 시간적분 결과가 1-B 정상상태 해와 일치하는가.

    **feasibility 기준이 아니다 — 구현 일관성 검사다.** 두 값은 서로 다른
    *계산 경로*로 나온다(1-B: 물성 평가온도에 대한 대수 고정점 `fsolve` ·
    세션 2: 2노드 미분방정식 `solve_ivp` 시간적분). 그러나 동적 모델의 평형점은
    설계상 1-B 식과 같은 방정식이므로, 이 대조가 재는 것은 **적분 정확도와 두
    모듈의 구현 일관성**이지 물리 자체의 독립 확인이 아니다.
    미해결 #15("랙·HX 를 독립적으로 푼 뒤 대조")를 이 테스트가 닫지는 못한다.

    그래도 값이 있다: 동적 모델을 잘못 세우면(노드 배분·부호·물성 평가 규칙이
    어긋나면) 여기서 바로 드러난다.
    """
    steady = solve_steady_state(steady_case)
    assert steady.solver_converged, f"{steady_case.label}: 1-B 정상상태 미수렴"

    lower_holdup: HoldupBound = holdup_bounds()[0]
    transient_case = LoadStepCase(
        label=f"C7 대조 · {steady_case.label}",
        holdup=lower_holdup,
        load_before_percent=LOAD_PROFILE.idle_load_percent,
        load_after_percent=LOAD_PROFILE.rated_load_percent,
        T_secondary_supply_C=steady_case.T_secondary_supply_C,
        ntu=steady_case.ntu,
        heat_capacity_ratio=steady_case.heat_capacity_ratio,
        rack_load_kW=SCENARIO.rack_it_load_kW,
        rack_flow_Lps=steady_case.rack_flow_Lps,
    )
    transient = integrate_load_step(transient_case)
    assert transient.solver_success, f"{steady_case.label}: solve_ivp 실패"

    supply_gap_C = abs(transient.T_supply_final_C - steady.T_supply_C)
    return_gap_C = abs(transient.T_return_final_C - steady.T_return_C)

    assert supply_gap_C < STEADY_STATE_TOLERANCE_C, (
        f"{steady_case.label}: T_supply 가 {supply_gap_C:.3e} K 어긋났다 "
        f"(적분 {transient.T_supply_final_C:.9f}℃ vs 정상상태 {steady.T_supply_C:.9f}℃)"
    )
    assert return_gap_C < STEADY_STATE_TOLERANCE_C, (
        f"{steady_case.label}: T_return 이 {return_gap_C:.3e} K 어긋났다 "
        f"(적분 {transient.T_return_final_C:.9f}℃ vs 정상상태 {steady.T_return_C:.9f}℃)"
    )
