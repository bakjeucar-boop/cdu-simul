"""세션 3 게이트 — 헤더 압력평형 유량분배 수렴 · 극단 케이스 비발산 (세션 3-B).

CLAUDE.md 절대 규칙 10: 사람이 눈으로 보지 않고 **테스트가 판정한다.**

`CLAUDE.md` 게이트 표의 세션 3 항목은 둘이다:

    "헤더 압력평형 기반 유량분배(`fsolve`) 수렴, 극단 케이스(부하 0/최대) 비발산"

세션 3-A 는 **열모델 없이** 수력만 8조합 돌렸으므로 게이트 판정이 아니었다
(미해결 #29). 이 파일이 **열결합 상태에서 32조합**으로 둘 다 판정한다.

**이 파일이 판정하지 않는 것**
· energy balance — `test_energy_balance.py` 가 같은 32조합으로 본다(1-B 게이트)
· T_return 방향성 — `test_dynamics.py` (세션 2 게이트)
· 수렴시간 — M 한계로 여전히 판정 불가(미해결 #21)

**과대해석 금지**: 8랙이 전부 동일 조건이라 열적으로는 1랙 집중모델과 동치다.
유량분배 결과는 잔여저항 배정 방식이 지배한다(총양정의 60~83% · 미해결 #24).
"""

from __future__ import annotations

import math

import pytest

from cdu_simul.assumptions import HEAT_EXCHANGER, LOAD_PROFILE, SCENARIO
from cdu_simul.dynamics import LoadStepCase, holdup_bounds, integrate_load_step
from cdu_simul.hydraulics import default_cases as default_hydraulic_cases
from cdu_simul.model import CduCase, default_cdu_cases, solve_cdu_steady_state

#: 극단 케이스에서 「부하 0」의 부하율 [%].
#: **5장 부하 프로파일(20~100%) 밖의 값이다.** 새 가정치를 만드는 것이 아니라
#: 6장이 발산 검사용으로 명시한 극단값이므로 5장 위반이 아니다(세션 3-B 지시).
ZERO_LOAD_PERCENT: float = 0.0

#: 부하 0 에서 1차측이 2차측 온도에 얼마나 붙어야 하는가 [K] — **정상상태** 판정용.
#: 해석적으로는 **정확히 0** 이다(아래 예상 참조) — 남는 것은 `fsolve` 수치 잔차뿐이라
#: 1e-9 K 로 잡는다. 모델이 실제로 어긋나면 K 단위로 벌어지므로 9자리 이상 엄격하다.
ZERO_LOAD_TOLERANCE_C: float = 1.0e-9

#: 시간적분 궤적의 하한 검사에 쓰는 여유 [K] — **정상상태 허용오차와 별개다.**
#: 지수감쇠는 해석적으로 2차측 온도 아래로 내려가지 않지만, `solve_ivp`(rtol 1e-10)
#: 는 점근선 근방에서 부호가 뒤집힌 수치 잡음을 남긴다 — 세션 2 가 같은 성질을
#: 1.4e-10 K 로 기록했고, 여기서 실측된 최대 침범은 1.5e-9 K 다. 1e-6 K 는 그
#: 잡음보다 3자리 위이면서 스텝 전체 변화량(약 15 K)보다는 7자리 아래라,
#: **물리적 언더슛이 생기면 반드시 걸린다.** 통과시키려고 늘린 값이 아니라
#: 정상상태 허용오차를 궤적 검사에 잘못 재사용한 것을 바로잡은 것이다.
#: (세션 2 `STEADY_STATE_TOLERANCE_C` 와 같은 근거·같은 크기다.)
TRAJECTORY_NOISE_FLOOR_C: float = 1.0e-6


def _extreme_load_cases(load_percent: float) -> list[CduCase]:
    """32조합을 주어진 부하율에서 만든다 (5장·5-1 범위 양 끝 전수)."""
    return default_cdu_cases(load_percent=load_percent)


# ─────────────────────────────────────────────────────────────────────────────
# 게이트 ㉠ — fsolve 수렴 (열결합 상태 · 32조합)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "case", default_cdu_cases(), ids=lambda c: c.label
)
def test_gate_flow_distribution_converges(case: CduCase) -> None:
    """세션 3 게이트 ㉠ — 32조합 전부에서 결합 해가 수렴한다.

    확인하는 solver 는 셋이다(절대 규칙 5):
    · 안쪽 수력 `fsolve` (헤더 압력평형) — `ier == 1`
    · 열 쪽 `fsolve` (물성 평가온도 고정점) — `ier == 1`
    · 바깥 결합 고정점 `fsolve` — `ier == 1`

    하나라도 실패하면 `CduSteadyStateResult.solver_converged` 가 False 다.
    """
    result = solve_cdu_steady_state(case)
    assert result.flow.solver_ier == 1, (
        f"{case.label}: 수력 fsolve ier={result.flow.solver_ier}"
    )
    assert result.thermal.solver_converged, (
        f"{case.label}: 열 fsolve 미수렴 — {result.thermal.solver_message}"
    )
    assert result.outer_solver_converged, (
        f"{case.label}: 결합 고정점 미수렴 — {result.outer_solver_message}"
    )
    assert result.solver_converged


# ─────────────────────────────────────────────────────────────────────────────
# 게이트 ㉡ — 극단 케이스 비발산 (부하 0% · 100%)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "case", _extreme_load_cases(ZERO_LOAD_PERCENT), ids=lambda c: c.label
)
def test_gate_zero_load_steady_state_matches_prediction(case: CduCase) -> None:
    """세션 3 게이트 ㉡ (정상상태 · 부하 0) — **예상을 먼저 적고 대조한다.**

    **사전 예상** (돌리기 전에 식에서 적은 것):
    발열량 Q_총 = 0 이면 `_state_at_property_temperature` 의 두 식이

        T_return = T_2차공급 + 0/(ε·C) = T_2차공급
        T_supply = T_return - 0/C      = T_2차공급

    가 되므로 **1차측 공급·환수가 둘 다 2차측 공급온도로 수렴하고 ΔT = 0** 이어야
    한다. 물리적으로도 그렇다 — 열원이 없으면 1차측은 열교환기를 통해 2차측 온도로
    끌려간다. ε·C 는 유량이 0이 아닌 한 0이 아니므로 **0으로 나누는 자리가 없다**
    (유량은 펌프가 만들며 부하와 무관하다).

    상대 잔차(`energy_balance_residual_percent`)는 Q=0 에서 정의되지 않는다 —
    그 함수는 명시적으로 `ValueError` 를 던지고, 이 게이트는 **비발산**으로 본다.
    """
    result = solve_cdu_steady_state(case)
    assert result.solver_converged, f"{case.label}: 부하 0 에서 미수렴"

    T_2nd_C = case.T_secondary_supply_C
    assert math.isfinite(result.thermal.T_supply_C)
    assert math.isfinite(result.thermal.T_return_C)
    assert abs(result.thermal.T_supply_C - T_2nd_C) < ZERO_LOAD_TOLERANCE_C, (
        f"{case.label}: T_supply {result.thermal.T_supply_C} vs 예상 {T_2nd_C}"
    )
    assert abs(result.thermal.T_return_C - T_2nd_C) < ZERO_LOAD_TOLERANCE_C, (
        f"{case.label}: T_return {result.thermal.T_return_C} vs 예상 {T_2nd_C}"
    )
    assert abs(result.thermal.dT_primary_C) < ZERO_LOAD_TOLERANCE_C


@pytest.mark.parametrize(
    "case",
    _extreme_load_cases(LOAD_PROFILE.rated_load_percent),
    ids=lambda c: c.label,
)
def test_gate_full_load_steady_state_does_not_diverge(case: CduCase) -> None:
    """세션 3 게이트 ㉡ (정상상태 · 부하 100%) — 발산하지 않는다.

    **사전 예상**: 5장 정격 부하이므로 1차측 온도가 5장 표의 32/42℃ 근방에
    있어야 하고, 환수온도는 2차측 공급온도보다 높아야 한다(열이 2차측으로 흘러야
    하므로). ΔT 는 양수여야 한다.

    발산의 정의를 넓게 잡는다: 값이 유한하고, 물리적으로 있을 수 없는 부호가
    나오지 않으며, 냉각액 물성이 정의된 범위를 크게 벗어나지 않는다. **"정확한가"
    는 보지 않는다** — 실측이 없다.
    """
    result = solve_cdu_steady_state(case)
    assert result.solver_converged, f"{case.label}: 부하 100% 에서 미수렴"

    assert math.isfinite(result.thermal.T_supply_C)
    assert math.isfinite(result.thermal.T_return_C)
    assert result.thermal.dT_primary_C > 0.0, f"{case.label}: ΔT 가 양수가 아니다"
    assert result.thermal.T_return_C > case.T_secondary_supply_C, (
        f"{case.label}: 환수온도가 2차측 공급온도보다 낮다 — 열이 거꾸로 흐른다"
    )
    assert result.thermal.T_supply_C > case.T_secondary_supply_C, (
        f"{case.label}: 공급온도가 2차측 공급온도보다 낮다"
    )
    assert 0.0 < result.thermal.T_return_C < 100.0, (
        f"{case.label}: 환수온도 {result.thermal.T_return_C} ℃ 가 물성 범위 밖이다"
    )
    assert result.flow.total_flow_Lps > 0.0


def _extreme_transient_cases() -> list[LoadStepCase]:
    """극단 부하 스텝 — {0%↔100%} × {M 하한, 상한}.

    수력 조합·2차측·NTU 는 하나로 고정한다. 정상상태 쪽이 32조합 전수를 보고,
    동적 쪽은 **발산 여부**를 보는 것이라 조합을 늘려도 같은 것을 반복한다.
    32조합 전수 적분은 1케이스당 약 0.7초라 실용적이지 않다(세션 3-B 실행시간 보고).
    """
    hydraulic = default_hydraulic_cases()[0]
    lower, upper = holdup_bounds()
    cases: list[LoadStepCase] = []
    for holdup in (lower, upper):
        for before, after, direction in (
            (LOAD_PROFILE.rated_load_percent, ZERO_LOAD_PERCENT, "100→0%"),
            (ZERO_LOAD_PERCENT, LOAD_PROFILE.rated_load_percent, "0→100%"),
        ):
            cases.append(
                LoadStepCase(
                    label=f"극단 {direction} · {holdup.label}",
                    holdup=holdup,
                    load_before_percent=before,
                    load_after_percent=after,
                    T_secondary_supply_C=SCENARIO.T_secondary_supply_C.low,
                    ntu=HEAT_EXCHANGER.ntu.low,
                    heat_capacity_ratio=(
                        HEAT_EXCHANGER.flow_ratio_primary_to_secondary
                    ),
                    hydraulic=hydraulic,
                )
            )
    return cases


@pytest.mark.parametrize(
    "case", _extreme_transient_cases(), ids=lambda c: c.label
)
def test_gate_extreme_load_transient_does_not_diverge(case: LoadStepCase) -> None:
    """세션 3 게이트 ㉡ (동적) — 극단 부하 스텝에서 시간적분이 발산하지 않는다.

    **사전 예상**: 100→0% 는 두 온도가 2차측 공급온도로 단조 접근하고, 0→100% 는
    정격 정상상태로 접근한다. 두 경우 모두 온도가 2차측 공급온도와 정격 환수온도
    사이에 머물러야 한다 — 이 계에는 열원이 랙 하나뿐이고 열침도 열교환기 하나뿐이라
    그 밖으로 나갈 경로가 없다.

    `solve_ivp` 의 `success` 와 매 시점 수력 `fsolve` 의 수렴을 **둘 다** 본다
    (절대 규칙 5). 수렴시간은 판정하지 않는다(미해결 #21).
    """
    result = integrate_load_step(case)
    assert result.solver_success, (
        f"{case.label}: solve_ivp 실패 — {result.solver_message}"
    )
    assert result.hydraulic_solver_converged, f"{case.label}: 수력 fsolve 미수렴"

    T_2nd_C = case.T_secondary_supply_C
    for name, series in (
        ("T_supply", result.T_supply_C),
        ("T_return", result.T_return_C),
    ):
        assert all(math.isfinite(T) for T in series), (
            f"{case.label}: {name} 에 비유한값이 있다"
        )
        assert min(series) > T_2nd_C - TRAJECTORY_NOISE_FLOOR_C, (
            f"{case.label}: {name} 최소 {min(series)} ℃ 가 2차측 {T_2nd_C} ℃ 아래다"
        )
        assert max(series) < 100.0, (
            f"{case.label}: {name} 최대 {max(series)} ℃ 가 물성 범위 밖이다"
        )

    if case.load_after_percent == ZERO_LOAD_PERCENT:
        assert abs(result.T_return_final_C - T_2nd_C) < TRAJECTORY_NOISE_FLOOR_C, (
            f"{case.label}: 부하 0 의 t→∞ 가 2차측 온도로 가지 않았다 "
            f"({result.T_return_final_C} ℃)"
        )
