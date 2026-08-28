"""energy balance 자동 검사 (세션 1-B 게이트).

CLAUDE.md 절대 규칙 10: feasibility 기준을 사람이 눈으로 보지 않고 **테스트가
판정한다.** 이 파일이 판정하는 것은 **energy balance 하나뿐이다.**
T_return 방향성(세션 2) · 수렴시간(세션 2) · 극단 케이스 발산(세션 3)은 여기서
판정하지 않으며, 그 테스트를 미리 만들지도 않는다.

판정 대상은 두 층이다.
· **열식만** 4케이스 = 2차측 27/30℃ × NTU 2/3 (유량은 5장 정격 고정)
· **수력 결합** 32조합 = 위 4 × 수력 8조합(양정 2 × 분기ΔP 2 × 밸브ΔP 2)
대표값을 고르지 않는다 — balance 는 보존법칙이므로 범위 안 어느 값에서도
성립해야 한다(미해결 #3 · 방침 (B)).

**세션 3-B 재확인**: `model.py` 를 세션 1-B 이후 처음 고친 판이므로, 8랙 확장과
수력 결합 뒤에도 1-B 게이트(<0.1%)가 성립하는지 32조합으로 다시 판정한다.
"""

from __future__ import annotations

import pytest

from cdu_simul.assumptions import SCENARIO
from cdu_simul.model import (
    CduCase,
    SteadyStateCase,
    default_cases,
    default_cdu_cases,
    energy_balance_residual_percent,
    hx_duty_identity_residual_percent,
    solve_cdu_steady_state,
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


# ─────────────────────────────────────────────────────────────────────────────
# 세션 3-B — 8랙 확장 · 수력 결합 뒤의 1-B 게이트 재판정 (32조합)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("case", default_cdu_cases(), ids=lambda c: c.label)
def test_coupled_energy_balance_residual_within_tolerance(case: CduCase) -> None:
    """수력과 결합한 8랙 CDU 에서도 energy balance 가 6장 기준(<0.1%) 안에 든다.

    **이 판의 첫 관문이다**(세션 3-B C4). `model.py` 를 세션 1-B 이후 처음
    건드렸으므로, 앞 게이트가 깨진 채로 세션 3 게이트를 보지 않는다(절대 규칙 6).

    잔차의 정의는 위 4케이스와 같다 — CoolProp 엔탈피 경로로 다시 계산한 흡열량과
    5장 총 발열량(랙당 × 8)의 차다. 유량이 압력평형의 해로 바뀌었을 뿐 잔차가
    재는 것은 그대로 **상수 cp 선형화의 정합성 하나**다. 통과가 "8랙 모델이
    정확하다"는 뜻이 아니다.
    """
    result = solve_cdu_steady_state(case)
    assert result.solver_converged, f"{case.label}: 결합 해 미수렴"

    residual_percent = energy_balance_residual_percent(result.thermal)
    assert abs(residual_percent) < ENERGY_BALANCE_TOLERANCE_PERCENT, (
        f"{case.label}: energy balance 잔차 {residual_percent:+.6f} % "
        f"(기준 <{ENERGY_BALANCE_TOLERANCE_PERCENT} %)"
    )


@pytest.mark.parametrize("case", default_cdu_cases(), ids=lambda c: c.label)
def test_eight_racks_match_single_rack_model(case: CduCase) -> None:
    """8랙 동일 조건의 온도가 같은 랙당 유량·부하의 1랙 모델과 일치한다.

    **feasibility 기준이 아니라 구현 일관성 검사다** — 세션 2 의 t→∞ 대조와 같은
    성격이다. 랙이 동일하면 총 발열량과 총 유량이 같은 배수로 커지므로 온도해가
    변하지 않는 것이 식에서 이미 따라 나온다. 이 검사가 재는 것은 **랙 축을 편
    구현이 그 성질을 깨지 않았는가**이지 물리의 독립 확인이 아니다.

    비교 대상을 5장 정격유량(1.94 L/s)이 아니라 **압력평형이 낸 랙당 유량**으로
    잡는다 — 두 모델에 같은 유량을 주어야 구현만 대조된다.

    **2차측 유량도 같은 배수로 나눈다**(세션 5-B). Cr 이 이제 양측 유량에서
    유도되므로(선언이 아니다), 1차측만 1/8 로 줄이고 2차측을 CDU 1대분 그대로
    두면 Cr 이 0.125 로 떨어져 아예 다른 열교환기를 비교하게 된다. 랙 축을 나누는
    것과 열교환기를 바꾸는 것은 다른 이야기다 — 두 모델이 **같은 Cr** 을 보도록
    2차측도 랙 수로 나눈다. 허용오차를 늘린 것이 아니라 비교 기준을 맞춘 것이다.
    """
    result = solve_cdu_steady_state(case)
    assert result.solver_converged, f"{case.label}: 결합 해 미수렴"
    assert result.thermal.case.n_racks == SCENARIO.racks_per_cdu

    single = solve_steady_state(
        SteadyStateCase(
            T_secondary_supply_C=case.T_secondary_supply_C,
            ntu=case.ntu,
            rack_loads_kW=(case.rack_load_kW,),
            rack_flows_Lps=(result.flow.mean_rack_flow_Lps,),
            secondary_flow_Lps=(
                result.thermal.case.secondary_flow_Lps / SCENARIO.racks_per_cdu
            ),
        )
    )
    assert single.solver_converged

    assert single.T_supply_C == pytest.approx(result.thermal.T_supply_C, abs=1.0e-9)
    assert single.T_return_C == pytest.approx(result.thermal.T_return_C, abs=1.0e-9)


@pytest.mark.parametrize("case", default_cdu_cases(), ids=lambda c: c.label)
def test_rack_return_temps_mix_to_header_temperature(case: CduCase) -> None:
    """랙별 환수온도를 유량가중 평균하면 환수 헤더 온도가 된다.

    합류 식(`_state_at_property_temperature` docstring)이 실제로 성립하는지의
    확인이다. 지금은 8랙이 동일해 모든 랙 온도가 같지만, 랙이 갈라지는 세션 4에서
    이 항등식이 신호의 근거가 된다 — 그 전에 고정해 둔다.
    """
    result = solve_cdu_steady_state(case)
    flows = result.flow.rack_flows_Lps
    temps = result.thermal.rack_return_temps_C
    assert len(flows) == len(temps) == SCENARIO.racks_per_cdu

    mixed_C = sum(f * T for f, T in zip(flows, temps, strict=True)) / sum(flows)
    assert mixed_C == pytest.approx(result.thermal.T_return_C, abs=1.0e-9)


def test_energy_balance_residual_undefined_at_zero_load() -> None:
    """부하 0 에서 상대 잔차 함수가 조용히 넘어가지 않고 예외를 던진다.

    0 으로 나누는 자리라 `ZeroDivisionError` 가 나면 원인을 읽기 어렵다. 극단
    케이스(6장)는 **비발산**으로 판정하고 이 잔차를 쓰지 않는다는 것을 함수가
    스스로 말하게 한다(`test_session3_gates.py` 참조).
    """
    zero_case = default_cdu_cases(load_percent=0.0)[0]
    result = solve_cdu_steady_state(zero_case)
    with pytest.raises(ValueError, match="부하 0"):
        energy_balance_residual_percent(result.thermal)
    with pytest.raises(ValueError, match="부하 0"):
        hx_duty_identity_residual_percent(result.thermal)
