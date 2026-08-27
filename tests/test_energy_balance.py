"""energy balance 자동 검사 (세션 1-B 게이트).

CLAUDE.md 절대 규칙 10: feasibility 기준을 사람이 눈으로 보지 않고 **테스트가
판정한다.** 이 파일이 판정하는 것은 **energy balance 하나뿐이다.**
T_return 방향성(세션 2) · 수렴시간(세션 2) · 극단 케이스 발산(세션 3)은 여기서
판정하지 않으며, 그 테스트를 미리 만들지도 않는다.

판정 대상 4케이스 = 5장 범위값의 양 끝 조합(2차측 27/30℃ × NTU 2/3).
대표값을 고르지 않는다 — balance 는 보존법칙이므로 범위 안 어느 값에서도
성립해야 한다(미해결 #3 · 방침 (B)).
"""

from __future__ import annotations

import pytest

from cdu_simul.model import (
    SteadyStateCase,
    default_cases,
    energy_balance_residual_percent,
    hx_duty_identity_residual_percent,
    solve_steady_state,
)

#: energy balance 허용 오차 [%]
#: [기준값: 프로젝트정리 6장 「Feasibility 검증 기준」 — "에너지 balance 오차 <0.1%"]
ENERGY_BALANCE_TOLERANCE_PERCENT: float = 0.1


def _case_id(case: SteadyStateCase) -> str:
    return case.label


@pytest.mark.parametrize("case", default_cases(), ids=_case_id)
def test_steady_state_solver_converged(case: SteadyStateCase) -> None:
    """수치 solver 의 성공 플래그를 확인한다 (절대 규칙 5).

    `fsolve` 의 `ier` 를 보지 않고 결과값을 쓰지 않는다.
    """
    result = solve_steady_state(case)
    assert result.solver_converged, (
        f"{case.label}: fsolve 가 수렴하지 않았다 — {result.solver_message}"
    )


@pytest.mark.parametrize("case", default_cases(), ids=_case_id)
def test_energy_balance_residual_within_tolerance(case: SteadyStateCase) -> None:
    """energy balance 잔차가 6장 기준(<0.1%) 안에 드는지 판정한다.

    잔차 = ( m_dot · [h(T_return) - h(T_supply)] - Q_rack ) / Q_rack × 100

    왼쪽은 CoolProp **엔탈피** 경로로 다시 계산한 1차측 흡열량이고, 오른쪽은
    5장 랙 발열량이다. 모델은 상수 cp 선형화로 해를 구했으므로 두 경로가 다르며,
    이 잔차는 항등적으로 0이 아니다(`model.energy_balance_residual_percent`
    docstring 참조).
    """
    result = solve_steady_state(case)
    assert result.solver_converged, f"{case.label}: solver 미수렴 — 잔차 판정 불가"

    residual_percent = energy_balance_residual_percent(result)
    assert abs(residual_percent) < ENERGY_BALANCE_TOLERANCE_PERCENT, (
        f"{case.label}: energy balance 잔차 {residual_percent:.5f}% 가 "
        f"허용 {ENERGY_BALANCE_TOLERANCE_PERCENT}% 를 넘었다 "
        f"(T_supply={result.T_supply_C:.2f}℃, T_return={result.T_return_C:.2f}℃)"
    )


@pytest.mark.parametrize("case", default_cases(), ids=_case_id)
def test_hx_duty_residual_is_structural_identity(case: SteadyStateCase) -> None:
    """**게이트가 아니다.** HX duty 잔차가 구조상 항등적으로 0임을 문서화한다.

    모델이 T_return 을 ε-NTU 관계에서 역산해 정의하므로 같은 식으로 duty 를
    되돌리면 부동소수점 반올림만 남는다. 이 사실을 테스트로 박아 두는 이유는,
    다음 세션이 "HX duty 잔차가 0이니 balance 가 검증됐다"고 오해하지 않게
    하기 위해서다 — 이 잔차는 아무것도 증명하지 못한다.
    """
    result = solve_steady_state(case)
    identity_residual_percent = hx_duty_identity_residual_percent(result)
    assert abs(identity_residual_percent) < 1.0e-9, (
        f"{case.label}: 항등이어야 할 잔차가 {identity_residual_percent:.3e}% 다 — "
        "모델 구조가 바뀌었을 수 있으니 C7 판단을 다시 읽어야 한다"
    )
