"""1-B · 2 · 3 게이트가 누출 코드 도입 뒤에도 그대로 서는가 (세션 4 C7).

절대 규칙 6: 앞 단계의 feasibility 기준을 통과한 상태를 유지하지 못하면 다음
단계로 가지 않는다. 누출 주입은 **새 코드 경로를 하나 더 만든 것**이므로, 정상
운전 결과가 세션 3-B 와 달라지지 않았는지 먼저 고정한다.

앞 게이트 자체의 판정은 원래 자리에 그대로 있다 —
`test_energy_balance.py`(1-B) · `test_dynamics.py`(2) · `test_session3_gates.py`(3).
이 파일이 더하는 것은 **정상 경로의 불변성**이다:

  ① 배율 1.0 을 통과시킨 해가 누출 코드를 거치지 않은 해와 **부동소수점까지 같다**
  ② 누출 상태에서도 1-B 게이트(<0.1%)와 세션 3 게이트(수렴)가 선다

①이 깨지면 정상과 누출의 차이가 누출 때문인지 코드 경로 때문인지 갈라낼 수 없다
(세션 4 C2). ②가 깨지면 누출 신호를 읽을 자격이 없다.
"""

from __future__ import annotations

import pytest

from cdu_simul.leak import leak_case, leak_levels, steady_signals
from cdu_simul.model import (
    default_cdu_cases,
    energy_balance_residual_percent,
    solve_cdu_steady_state,
)

#: 6장 energy balance 기준 [%].
ENERGY_BALANCE_TOLERANCE_PERCENT: float = 0.1


@pytest.mark.parametrize("case", default_cdu_cases(), ids=lambda c: c.label)
def test_normal_path_is_bit_identical_to_direct_solve(case) -> None:  # type: ignore[no-untyped-def]
    """배율 1.0 경로의 해가 누출 코드를 거치지 않은 해와 **완전히 같다**.

    허용오차를 두지 않고 `==` 로 본다 — 배율 1.0 곱은 부동소수점 항등이므로
    해석적으로도 수치적으로도 **정확히 같아야** 한다. 조금이라도 다르면 누출
    경로가 K값 외의 무언가를 건드리고 있다는 뜻이다.
    """
    direct = solve_cdu_steady_state(case)
    through_leak_path = solve_cdu_steady_state(leak_case(case, leak_levels()[0]))

    assert direct.solver_converged and through_leak_path.solver_converged
    assert through_leak_path.thermal.T_supply_C == direct.thermal.T_supply_C
    assert through_leak_path.thermal.T_return_C == direct.thermal.T_return_C
    assert through_leak_path.thermal.hx_duty_kW == direct.thermal.hx_duty_kW
    assert through_leak_path.flow.total_flow_Lps == direct.flow.total_flow_Lps
    assert through_leak_path.flow.rack_flows_Lps == direct.flow.rack_flows_Lps
    assert through_leak_path.flow.pump_head_mAq == direct.flow.pump_head_mAq


@pytest.mark.parametrize("case", default_cdu_cases(), ids=lambda c: c.label)
def test_session3_gate_holds_under_leak(case) -> None:  # type: ignore[no-untyped-def]
    """세션 3 게이트 ㉠(수렴)이 누출 3수준에서도 선다.

    K값이 커지면 시스템곡선이 가팔라지므로 운전점이 옮겨간다 — `fsolve` 가 새
    운전점을 찾지 못하면 누출 시나리오 자체가 성립하지 않는다. 수력·열·결합
    고정점 세 solver 를 전부 본다(절대 규칙 5).
    """
    for signal in steady_signals(case):
        assert signal.leaked.flow.solver_ier == 1, (
            f"{signal.level.label}: 수력 fsolve ier={signal.leaked.flow.solver_ier}"
        )
        assert signal.leaked.thermal.solver_converged
        assert signal.leaked.outer_solver_converged


@pytest.mark.parametrize("case", default_cdu_cases(), ids=lambda c: c.label)
def test_1b_gate_holds_under_leak(case) -> None:  # type: ignore[no-untyped-def]
    """1-B 게이트(energy balance <0.1%)가 누출 3수준에서도 선다.

    K값은 압력-유량 쪽 값이고 보존법칙과 무관하므로 잔차가 크게 움직일 이유가
    없다 — 움직이면 누출 주입이 열식으로 새고 있다는 뜻이다.
    """
    for signal in steady_signals(case):
        residual_percent = energy_balance_residual_percent(signal.leaked.thermal)
        assert abs(residual_percent) < ENERGY_BALANCE_TOLERANCE_PERCENT, (
            f"{case.label} | {signal.level.label}: {residual_percent:+.6f} %"
        )
