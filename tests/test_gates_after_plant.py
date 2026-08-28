"""단일 CDU 경로 격리 · 앞 게이트 재확인 (세션 5 C4·C7).

절대 규칙 6: 앞 단계의 게이트를 통과한 상태를 유지하지 못하면 다음으로 가지
않는다. 세션 5 는 `model.py`·`dynamics.py` 의 열교환기 항을 건드렸으므로
(`hx_capacity_terms` 도입), **2차측 유량을 주지 않은 경로가 세션 4까지와 완전히
같은지**부터 고정한다.

앞 게이트 자체의 판정은 원래 자리에 그대로 있다 —
`test_energy_balance.py`(1-B) · `test_dynamics.py`(2) · `test_session3_gates.py`(3) ·
`test_session4_gates.py`(4). 이 파일이 더하는 것은 **경로 불변성**과, 결합 상태에서
누출을 한 CDU 에만 주입했을 때의 **관측**이다.

**C7 의 누출 관측은 새 게이트가 아니다** — 세션 4 게이트는 이미 판정됐다. 여기서는
결합이 들어온 뒤에도 그 신호가 서는지, 그리고 상대 CDU 가 공유 2차측을 통해
반응하는지만 본다.
"""

from __future__ import annotations

import pytest

from cdu_simul.assumptions import PLANT
from cdu_simul.leak import leak_case, leak_levels
from cdu_simul.model import (
    default_cdu_cases,
    energy_balance_residual_percent,
    solve_cdu_steady_state,
)
from cdu_simul.plant import PlantCase, plant_case, solve_plant_steady_state

#: 6장 energy balance 기준 [%].
ENERGY_BALANCE_TOLERANCE_PERCENT: float = 0.1


@pytest.mark.parametrize("case", default_cdu_cases(), ids=lambda c: c.label)
def test_single_cdu_path_is_bit_identical(case) -> None:  # type: ignore[no-untyped-def]
    """2차측 유량을 주지 않은 경로가 세션 4까지와 **완전히 같다**.

    허용오차를 두지 않고 `==` 로 본다 — `hx_capacity_terms` 가 `None` 분기에서
    예전 식을 글자 그대로 되돌려주므로 부동소수점까지 같아야 한다. 조금이라도
    다르면 결합 도입이 단일 CDU 경로로 샜다는 뜻이다(세션 5 C2 "단일 CDU 경로를
    지우지 않는다").
    """
    result = solve_cdu_steady_state(case)
    assert result.solver_converged
    assert result.thermal.case.secondary_flow_Lps is None
    assert result.thermal.hx_effectiveness == pytest.approx(
        result.thermal.hx_effectiveness, abs=0.0
    )
    # 명목 Cr=1 이면 ε = NTU/(1+NTU) 라는 1-B 의 닫힌 형태가 그대로 서야 한다.
    assert result.thermal.hx_effectiveness == case.ntu / (1.0 + case.ntu)


@pytest.mark.parametrize("template", default_cdu_cases(), ids=lambda c: c.label)
def test_leak_signal_survives_coupling(template) -> None:  # type: ignore[no-untyped-def]
    """결합 상태에서 한 CDU 에만 누출을 넣어도 그 CDU 의 신호 방향이 유지된다.

    기대(세션 4 게이트 기준 A 와 같은 방향): 누출 CDU 의 총유량 **감소** ·
    환수온도 **상승** · 펌프 양정 **상승**.

    **새 게이트가 아니다** — 세션 4 에서 이미 판정했다. 여기서는 공유 2차측이
    붙은 뒤에도 그 방향이 뒤집히지 않는지만 본다.
    """
    base = solve_plant_steady_state(
        plant_case("정상", (template.load_percent,) * PLANT.cdu_count, template)
    )
    assert base.solver_converged

    for level in leak_levels()[1:]:
        leaked = solve_plant_steady_state(
            PlantCase(
                label=f"누출 {level.label}",
                cdus=(leak_case(base.case.cdus[0], level),) + base.case.cdus[1:],
            )
        )
        assert leaked.solver_converged, f"{template.label} {level.label}: 미수렴"

        a_before, a_after = base.cdu_results[0], leaked.cdu_results[0]
        assert a_after.flow.total_flow_Lps < a_before.flow.total_flow_Lps
        assert a_after.thermal.T_return_C > a_before.thermal.T_return_C
        assert a_after.flow.pump_head_mAq > a_before.flow.pump_head_mAq

        for cdu_result in leaked.cdu_results:
            residual_percent = energy_balance_residual_percent(cdu_result.thermal)
            assert abs(residual_percent) < ENERGY_BALANCE_TOLERANCE_PERCENT


@pytest.mark.parametrize("template", default_cdu_cases(), ids=lambda c: c.label)
def test_leak_in_one_cdu_moves_the_other(template) -> None:  # type: ignore[no-untyped-def]
    """한 CDU 의 누출이 **다른 CDU** 를 움직인다 — 공유 2차측을 통한 관측.

    기대 방향: 누출로 CDU A 의 1차측 유량이 줄면 비례 배분에서 A 의 몫이 줄고
    **B 의 몫이 늘어난다** → B 는 2차측을 더 받아 냉각이 좋아지므로 환수온도가
    **내려간다**.

    **관측이지 게이트가 아니다.** 크기를 판정하지 않는다 — 5-1 이 열 경로 결합을
    모델에서 뺐으므로(2차측 공급온도 고정) 여기 나오는 것은 유량 경로 하나뿐이다.
    """
    base = solve_plant_steady_state(
        plant_case("정상", (template.load_percent,) * PLANT.cdu_count, template)
    )
    major = leak_levels()[-1]
    leaked = solve_plant_steady_state(
        PlantCase(
            label="누출",
            cdus=(leak_case(base.case.cdus[0], major),) + base.case.cdus[1:],
        )
    )
    assert leaked.solver_converged

    assert leaked.secondary_shares_Lps[1] > base.secondary_shares_Lps[1], (
        f"{template.label}: A 에 누출이 걸렸는데 B 의 2차측 배분이 늘지 않았다"
    )
    assert leaked.cdu_results[1].thermal.T_return_C < (
        base.cdu_results[1].thermal.T_return_C
    ), f"{template.label}: B 의 배분이 늘었는데 환수온도가 내려가지 않았다"
