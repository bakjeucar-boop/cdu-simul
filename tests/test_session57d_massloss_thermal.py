"""세션 5.7-D — 「샘」(질량손실) 열모델의 자기정합성 확인.

**게이트가 아니다.** 6장 기준을 하나도 판정하지 않는다. 이 파일이 고정하는 것은
`massloss_thermal` 이라는 **별도 관측 코드**가 두 가지를 지키는가 하나다:

1. **격리** — 「샘」 0 에서 기존 밀폐루프 해를 그대로 재현하는가(정상 경로 무오염)
2. **balance** — 「샘」 엔탈피 항을 **넣어야** 닫히는가(빼면 닫히지 않아야 한다)

전수 스윕은 `python -m cdu_simul.massloss_thermal` 이 돌린다 — 여기서는
대표 조합만 본다. 판정 기준은 `PROCEED.md` 「세션 5.7-D … 판정 기준 선기재」다.
"""

from __future__ import annotations

import pytest

from cdu_simul.hydraulics import rated_property_temperature_C
from cdu_simul.massloss import massloss_topologies
from cdu_simul.massloss_thermal import (
    massloss_sizes_Lps,
    solve_massloss_steady,
    thermal_cases,
)
from cdu_simul.model import solve_cdu_steady_state

#: 대표 조합 — 수력·부하 양 끝이 섞이도록 앞뒤에서 하나씩 집는다.
_CASES = [thermal_cases()[0], thermal_cases()[-1]]


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.label)
@pytest.mark.parametrize("topology", massloss_topologies(), ids=lambda t: t.label)
def test_massloss_zero_reproduces_closed_loop(case, topology) -> None:
    """「샘」 0 이면 밀폐루프 해와 **부동소수점까지** 같아야 한다.

    다르면 이 모듈이 정상 경로를 건드린 것이다. 배치(g·펌프 유량 위치)는 「샘」이
    0 이면 식에서 사라지므로 배치 6 전부에서 같아야 한다.
    """
    closed_loop = solve_cdu_steady_state(case).thermal
    massloss = solve_massloss_steady(case, 0.0, topology)

    assert massloss.solver_converged
    assert massloss.T_supply_C == pytest.approx(closed_loop.T_supply_C, abs=1e-9)
    assert massloss.T_return_C == pytest.approx(closed_loop.T_return_C, abs=1e-9)
    assert massloss.supply_flow_Lps == pytest.approx(
        massloss.return_flow_Lps, abs=0.0
    )


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.label)
@pytest.mark.parametrize("topology", massloss_topologies(), ids=lambda t: t.label)
def test_balance_closes_only_with_massloss_enthalpy(case, topology) -> None:
    """「샘」 엔탈피 항을 넣어야 balance 가 닫힌다 — 판정 기준 D.

    허용오차 0.1% 는 6장 기준 ①의 숫자를 **빌려 온 것**이지 6장을 판정하는 것이
    아니다(이 모듈은 게이트가 아니다). 항을 뺐을 때 닫히지 않는 것까지 함께
    고정해야 이 확인이 항등식이 아님이 드러난다.
    """
    size = massloss_sizes_Lps(case.hydraulic, rated_property_temperature_C())[-1]
    result = solve_massloss_steady(case, size, topology)

    assert result.solver_converged
    assert abs(result.balance_residual_with_massloss_percent) < 0.1
    assert abs(result.balance_residual_without_massloss_percent) > 0.1
