"""세션 5 게이트 — 다중 CDU 연동 수렴.

CLAUDE.md 절대 규칙 10: 사람이 눈으로 보지 않고 **테스트가 판정한다.**

`CLAUDE.md` 게이트 표의 세션 5 항목:

    "상위 레벨 연립방정식으로 CDU 간 연동 수렴"

═══════════════════════════════════════════════════════════════════════════════
**세션 5 게이트 판정 기준 (선기재)**

**기준 A — 상위 연립방정식 수렴.** 5장·5-1 범위 양 끝 32조합 × 부하 시나리오
3(대칭 100/100 · 대칭 20/20 · 비대칭 100/20) = **96 케이스 전부**에서
  · 상위 `fsolve` 의 `ier == 1`
  · 각 CDU 의 물성 온도 `fsolve` 수렴
  · 각 CDU 의 헤더 압력평형 `fsolve` 수렴
셋이 **동시에** 성립해야 한다. 하나라도 실패하면 결합 구조가 성립하지 않는다.

**기준 B — 연립이 실제로 연립인가.** 잔차가 **동시에** 0이어야 한다. 상위
`fsolve` 가 낸 해에서 모든 CDU 의 잔차 최대 절대값이 수치 잡음 수준이어야 한다.
순차 대입이었다면 마지막 CDU 만 0이고 앞쪽은 남는다.

**기준 C — 연동이 실제로 존재하는가.** 비대칭 부하에서 **부하가 바뀌지 않은
CDU 의 상태가 움직여야** 한다. 움직이지 않으면 "연동이 수렴했다"는 말이 공허하다
(두 CDU 가 독립이면 어떤 solver 든 수렴한다). 구체적으로 비대칭 100/20% 의 CDU B
가 대칭 20/20% 의 CDU B 와 **달라야** 한다.

**이 게이트가 판정하지 않는 것**
· 연동의 **크기**가 크다/작다 — 판단 기준이 없다. 5-1 이 열 경로 결합을 모델에서
  뺐으므로(2차측 공급온도 고정) 여기서 나오는 연동은 유량 경로 하나뿐이다
· 대칭 케이스가 세션 4 단일 CDU 결과를 재현하는가 — **재현하지 않는다.**
  `test_symmetric_case_deviation_from_single_cdu` 가 그 편차를 기록만 한다.
  원인은 5-1 두 행의 내부 모순이며 사람의 판정이 필요하다(미해결 #33)
· 수렴시간 — #21 · #31
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math

import pytest

from cdu_simul.assumptions import LOAD_PROFILE, PLANT
from cdu_simul.model import (
    default_cdu_cases,
    energy_balance_residual_percent,
    solve_cdu_steady_state,
)
from cdu_simul.plant import (
    default_plant_cases,
    default_plant_load_step_cases,
    integrate_plant_load_step,
    plant_case,
    solve_plant_steady_state,
)

#: 상위 연립 잔차 허용오차 [K].
#: `fsolve` 가 해라고 낸 점에서 남는 것은 수치 잡음뿐이라 1e-6 K 로 잡는다 —
#: 실측 잔차는 1e-13~1e-11 K 수준이고, 순차 대입이었다면 K 단위로 남는다.
SIMULTANEITY_TOLERANCE_C: float = 1.0e-6

#: 6장 energy balance 기준 [%].
ENERGY_BALANCE_TOLERANCE_PERCENT: float = 0.1

PLANT_CASES = default_plant_cases()


# ─────────────────────────────────────────────────────────────────────────────
# 기준 A — 수렴
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("case", PLANT_CASES, ids=lambda c: c.label)
def test_criterion_a_top_level_converges(case) -> None:  # type: ignore[no-untyped-def]
    """기준 A — 96 케이스 전부에서 상위·하위 solver 가 전부 수렴한다.

    절대 규칙 5: `ier` 를 확인하지 않고 결과값을 쓰지 않는다.
    """
    result = solve_plant_steady_state(case)
    assert result.top_level_solver_ier == 1, (
        f"{case.label}: 상위 fsolve ier={result.top_level_solver_ier} — "
        f"{result.top_level_solver_message}"
    )
    for index, cdu_result in enumerate(result.cdu_results):
        assert cdu_result.flow.solver_ier == 1, f"{case.label}: CDU {index} 수력 미수렴"
        assert cdu_result.thermal.solver_converged
        assert cdu_result.outer_solver_converged
    assert result.solver_converged


# ─────────────────────────────────────────────────────────────────────────────
# 기준 B — 연립이 실제로 연립인가
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("case", PLANT_CASES, ids=lambda c: c.label)
def test_criterion_b_residuals_vanish_simultaneously(case) -> None:  # type: ignore[no-untyped-def]
    """기준 B — 모든 CDU 의 잔차가 **동시에** 0이다.

    순차 대입(한쪽 풀고 다른 쪽에 넘기기)이었다면 마지막 CDU 만 0이고 앞쪽에
    잔차가 남는다. 최대 절대값을 보는 것이 그 구분이다.
    """
    result = solve_plant_steady_state(case)
    assert result.max_abs_residual_C < SIMULTANEITY_TOLERANCE_C, (
        f"{case.label}: 잔차 최대 {result.max_abs_residual_C:.3e} K"
    )
    assert math.isfinite(result.max_abs_residual_C)


# ─────────────────────────────────────────────────────────────────────────────
# 기준 C — 연동이 실제로 존재하는가
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("template", default_cdu_cases(), ids=lambda c: c.label)
def test_criterion_c_coupling_actually_exists(template) -> None:  # type: ignore[no-untyped-def]
    """기준 C — 비대칭 부하에서 **부하가 그대로인 CDU** 의 상태가 움직인다.

    비교: 대칭 20/20% 의 CDU B vs 비대칭 100/20% 의 CDU B. 두 경우 B 의 부하는
    똑같이 20% 인데, 상대 CDU 의 부하만 다르다. 값이 같다면 두 CDU 가 독립이라는
    뜻이고 「연동」이라는 말이 성립하지 않는다.

    **크기를 판정하지 않는다** — 0 이 아니라는 것만 본다.
    """
    idle = LOAD_PROFILE.idle_load_percent
    rated = LOAD_PROFILE.rated_load_percent
    symmetric = solve_plant_steady_state(
        plant_case("대칭", (idle,) * PLANT.cdu_count, template)
    )
    asymmetric = solve_plant_steady_state(
        plant_case("비대칭", (rated,) + (idle,) * (PLANT.cdu_count - 1), template)
    )
    assert symmetric.solver_converged and asymmetric.solver_converged

    b_symmetric = symmetric.cdu_results[-1]
    b_asymmetric = asymmetric.cdu_results[-1]
    assert b_symmetric.case.load_percent == b_asymmetric.case.load_percent == idle

    assert b_asymmetric.thermal.T_return_C != b_symmetric.thermal.T_return_C, (
        f"{template.label}: 상대 CDU 부하가 바뀌었는데 B 의 환수온도가 그대로다 — "
        "두 CDU 가 독립이라는 뜻이고 연동이 없다"
    )
    assert (
        asymmetric.secondary_shares_Lps[-1] != symmetric.secondary_shares_Lps[-1]
    ), f"{template.label}: 2차측 배분이 움직이지 않았다"


# ─────────────────────────────────────────────────────────────────────────────
# 앞 게이트 — 결합 상태에서도 energy balance 가 선다
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("case", PLANT_CASES, ids=lambda c: c.label)
def test_energy_balance_holds_under_coupling(case) -> None:  # type: ignore[no-untyped-def]
    """1-B 게이트(<0.1%)가 공유 2차측 결합 상태에서도 선다 (절대 규칙 6).

    부하 0 이 아닌 케이스만 본다 — 5장 부하 프로파일 하한이 20% 이므로 전부 해당한다.
    """
    result = solve_plant_steady_state(case)
    for index, cdu_result in enumerate(result.cdu_results):
        residual_percent = energy_balance_residual_percent(cdu_result.thermal)
        assert abs(residual_percent) < ENERGY_BALANCE_TOLERANCE_PERCENT, (
            f"{case.label} CDU {index}: {residual_percent:+.6f} %"
        )


# ─────────────────────────────────────────────────────────────────────────────
# C4 — 대칭 케이스와 단일 CDU 의 편차 (판정이 아니라 기록)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("template", default_cdu_cases(), ids=lambda c: c.label)
def test_symmetric_case_deviation_from_single_cdu(template) -> None:  # type: ignore[no-untyped-def]
    """대칭 케이스가 세션 4 단일 CDU 결과를 **재현하지 못한다** — 그 사실을 고정한다.

    세션 5 지시(C4)는 "비례 배분이므로 각자 정확히 1:1을 받는다"를 근거로 수치가
    같기를 기대했다. **성립하지 않는다.** 두 가지 이유가 겹친다:

    ① 총 2차측 유량이 5장 **정격**(15.5 L/s × 2)으로 고정인데 1차측 운전점은
       15.5056 L/s 다(세션 3-A 가 기록한 5장 자체의 반올림). 2차:1차 = 0.99964.
    ② 설령 부피유량이 정확히 같아도 **양쪽 온도가 다르므로** ρ·cp 가 달라
       Cr = 0.99863 이 된다 — 5장 「유량비 1:1」은 부피 기준이기 때문이다.

    즉 세션 4 의 Cr = 1 은 **유량에서 유도된 값이 아니라 5-1 「2차측 유체」가
    선언한 가정**이었다. 공유 2차측 배분에서 Cr 을 유도하는 한 어떤 배분 규칙으로도
    정확히 1 이 나오지 않는다.

    **이 테스트는 판정하지 않는다.** 편차가 유한하고 물리적으로 설명되는 크기
    (0.03 K 미만)임을 고정할 뿐이고, 5-1 두 행 중 무엇이 지배하는지는 **사람이
    정한다**(미해결 #33). 임계 0.03 K 는 통과 기준이 아니라 **회귀 방지선**이다 —
    이 값이 커지면 결합 구조가 다른 이유로 어긋난 것이므로 사람이 봐야 한다.
    """
    single = solve_cdu_steady_state(template)
    coupled = solve_plant_steady_state(
        plant_case("대칭", (template.load_percent,) * PLANT.cdu_count, template)
    )
    assert coupled.solver_converged

    for cdu_result in coupled.cdu_results:
        supply_gap_C = abs(cdu_result.thermal.T_supply_C - single.thermal.T_supply_C)
        return_gap_C = abs(cdu_result.thermal.T_return_C - single.thermal.T_return_C)
        assert supply_gap_C < 0.03, (
            f"{template.label}: T_supply 편차 {supply_gap_C:.6f} K"
        )
        assert return_gap_C < 0.03, (
            f"{template.label}: T_return 편차 {return_gap_C:.6f} K"
        )


# ─────────────────────────────────────────────────────────────────────────────
# C6 — 동적 결합
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "case", default_plant_load_step_cases(), ids=lambda c: c.label
)
def test_asymmetric_load_step_couples_dynamically(case) -> None:  # type: ignore[no-untyped-def]
    """비대칭 부하 스텝에서 **부하가 그대로인 CDU 가 반응한다** (동적 연동).

    기대: A 의 부하가 오르면 A 의 1차측 유량이 오르고, 고정된 총 2차측에서 A 의
    배분이 커진다 → B 의 배분이 **줄고** → B 의 Cr 이 바뀌어 B 의 환수온도가
    **오른다**(냉각이 조금 나빠진다).

    `solve_ivp` 의 `success` 와 매 시점 수력 `fsolve` 의 수렴을 **둘 다** 본다.
    **전이 시간 규모의 절대값은 판정하지 않는다**(#21 · #31).
    """
    result = integrate_plant_load_step(case)
    assert result.solver_success, (
        f"{case.label}: solve_ivp 실패 — {result.solver_message}"
    )
    assert result.hydraulic_solver_converged

    for series in result.T_supply_C + result.T_return_C:
        assert all(math.isfinite(T) for T in series), f"{case.label}: 비유한값"

    # A 는 부하가 올랐으므로 환수온도가 크게 오른다.
    assert result.T_return_change_C(0) > 0.0

    # B 는 부하가 그대로인데도 움직인다 — 이것이 연동이다.
    assert result.T_return_change_C(1) != 0.0, (
        f"{case.label}: 부하가 그대로인 CDU B 가 전혀 움직이지 않았다"
    )
    assert (
        result.secondary_shares_final_Lps[1]
        < result.secondary_shares_initial_Lps[1]
    ), (
        f"{case.label}: A 의 부하가 올랐는데 B 의 2차측 배분이 줄지 않았다"
    )
