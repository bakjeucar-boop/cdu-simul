"""단일 CDU 경로 격리 · 앞 게이트 재확인 (세션 5 C4·C7).

절대 규칙 6: 앞 단계의 게이트를 통과한 상태를 유지하지 못하면 다음으로 가지
않는다.

**세션 5-B 에서 이 파일의 첫 테스트가 뒤집혔다.** 세션 5 는 "2차측 유량을 주지 않은
경로가 세션 4까지와 완전히 같다"를 고정했는데, 세션 5-B 가 그 전제(Cr=1 선언)를
폐기했다. 지금 고정하는 것은 **Cr 이 유도된다는 사실**이다 — 자세한 것은 아래
`test_single_cdu_uses_derived_cr` docstring 을 본다.

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

from cdu_simul.assumptions import HEAT_EXCHANGER, PLANT
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
def test_single_cdu_uses_derived_cr(case) -> None:  # type: ignore[no-untyped-def]
    """단일 CDU 경로가 **선언된 Cr=1 이 아니라 유도된 Cr** 을 쓴다 (세션 5-B).

    이 테스트는 세션 5 에서 "2차측 유량을 주지 않은 경로가 세션 4와 부동소수점까지
    같다"를 고정하고 있었다. **세션 5-B 가 그 전제를 폐기했다** — 5장 「1차:2차
    유량비 1:1」은 부피유량비이지 Cr 이 아니고, 양측 온도가 달라 Cr ≠ 1 이다.
    허용오차를 늘린 것이 아니라 **고정 대상이 바뀐 것**이므로 잠금 내용을 바꾼다.

    이제 고정하는 것은 셋이다:
      ① 2차측 유량이 `None` 이 아니라 5장에서 유도된 정격값(15.5 L/s)이다 —
         선언 Cr 이 들어올 자리가 코드에 남아 있지 않다
      ② ε 가 Cr=1 의 닫힌 형태 NTU/(1+NTU) 와 **다르다** (Cr<1 이므로)
      ③ 그럼에도 차이가 물성 차이에서 오는 크기(상대 1e-3 미만)여야 한다 —
         이보다 크면 유량이나 물성 평가점이 어긋난 것이므로 사람이 봐야 한다
    """
    result = solve_cdu_steady_state(case)
    assert result.solver_converged
    assert result.thermal.case.secondary_flow_Lps == (
        HEAT_EXCHANGER.secondary_flow_Lps
    )

    cr_one_effectiveness = case.ntu / (1.0 + case.ntu)
    assert result.thermal.hx_effectiveness != cr_one_effectiveness, (
        f"{case.label}: ε 가 아직 Cr=1 의 닫힌 형태다 — 선언 Cr 이 남아 있다"
    )
    relative_gap = abs(
        result.thermal.hx_effectiveness / cr_one_effectiveness - 1.0
    )
    assert relative_gap < 1.0e-3, (
        f"{case.label}: ε 가 Cr=1 형태에서 상대 {relative_gap:.3e} 벗어났다 — "
        "물성 차이만으로 설명되지 않는 크기다"
    )


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
