"""세션 4 게이트 — 누출 신호 패턴 식별 가능성.

CLAUDE.md 절대 규칙 10: 사람이 눈으로 보지 않고 **테스트가 판정한다.**

`CLAUDE.md` 게이트 표의 세션 4 항목:

    "정상 대비 신호 패턴(온도상승·유량감소·펌프 운전점 이동) 식별 가능"

「식별 가능」은 그 자체로 판정할 수 없는 말이므로, **판정 기준을 코드보다 먼저
아래에 적고 그대로 판정한다**(세션 4 C6 · 세션 3-B C8 과 같은 방식).

═══════════════════════════════════════════════════════════════════════════════
**세션 4 게이트 판정 기준 (선기재)**

신호 3종을 본다: ① 누출 랙 유량 ② 누출 랙 출구온도 ③ 펌프 운전점(양정·총유량).

**기준 A — 부호 일관성.** 32조합 × 누출 3수준 **전부**에서 정상 대비 부호가
같아야 한다. 어느 조합에서 부호가 뒤집히면 그 신호는 누출의 지표가 될 수 없다.
  · 누출 랙 유량: **감소** (Δ < 0)  — 그 랙 저항이 커졌으므로
  · 누출 랙 출구온도: **상승** (Δ > 0) — 발열은 그대로인데 유량이 줄었으므로
  · 총유량: **감소** (Δ < 0)
  · 펌프 양정: **상승** (Δ > 0) — 정속 곡선 위에서 유량이 줄면 양정이 오른다
    [5-1 「펌프 운전 방식」]
  · 비누출 랙 유량: **증가** (Δ > 0) — 공통 경로 강하가 줄어 남는 양정이 커진다
    (3-A 방향성 검사에서 이미 확인된 방향)

**기준 B — 수준 간 단조.** 같은 조합 안에서 +5% < +20% < +50% 순으로 각 신호의
**크기(절대값)가 엄격히 커져야** 한다. 커지지 않으면 누출 정도를 신호에서 되읽을
수 없고, AI 학습 데이터셋으로서 수준 구분이 성립하지 않는다.

**기준 C — 잡음 대비.** 가장 작은 누출(+5%)의 신호가 수치 잡음보다 충분히 커야
한다. 잡음 크기의 근거는 **세션 3-B 가 실제로 관측한 값**이다:
  · 적분기 잡음 1.5e-9 K (부하 0 궤적의 점근선 침범) ~ 1e-6 K (그 여유 상한)
  · 수력 압력평형 잔차 ~1e-15 mAq · 랙 간 유량 균등성 잔차 상대 1e-9
임계는 그 잡음의 최소 1000배로 잡는다 — 아래 상수의 근거를 참조. 통과시키려고
고른 값이 아니며, 실제 +5% 신호는 임계보다 2~4자리 크다.

**이 게이트가 판정하지 않는 것**
· 신호 크기가 **실제 누출 감지에 충분한가** — 실측이 없으므로 판정 불가
· 전이 시간 규모가 빠른가/느린가 — M 결손(#21) · 8랙 해석 부재(#31)
· 누출의 물리 — K값 증가는 5장이 정의한 **대용(proxy)** 이고 질량 손실은 모델에 없다
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math

import pytest

from cdu_simul.assumptions import HEAT_EXCHANGER, SCENARIO
from cdu_simul.dynamics import LeakStepCase, holdup_bounds, integrate_leak_step
from cdu_simul.hydraulics import default_cases as default_hydraulic_cases
from cdu_simul.leak import (
    LeakSteadySignal,
    leak_levels,
    steady_signals,
)
from cdu_simul.model import default_cdu_cases

#: 온도 신호의 잡음 임계 [K].
#: 세션 3-B 가 관측한 적분기 잡음 상한 1e-6 K 의 **1000배**다. 정상상태 해는
#: `fsolve` 로 나오므로 적분기 잡음이 섞이지 않지만, 동적 신호와 같은 잣대를 쓰려고
#: 더 보수적인 쪽(적분기 잡음)을 기준으로 삼았다.
TEMPERATURE_NOISE_THRESHOLD_C: float = 1.0e-3

#: 유량 신호의 잡음 임계 [상대 %].
#: 세션 3-A 가 고정한 랙 간 유량 균등성 잔차는 상대 1e-9(= 1e-7 %)다. 임계
#: 1e-3 % 는 그보다 **10^4 배** 크다.
FLOW_NOISE_THRESHOLD_PERCENT: float = 1.0e-3

#: 펌프 양정 신호의 잡음 임계 [mAq].
#: 세션 3-A 가 관측한 압력평형식 잔차는 최대 8.9e-15 mAq 다. 임계 1e-4 mAq 는
#: 그보다 10^10 배 크고, 양정 20~30 mAq 대비 상대 5e-6 이다.
HEAD_NOISE_THRESHOLD_mAq: float = 1.0e-4

#: energy balance 허용 오차 [%] — 6장 기준. 누출 상태에서도 서야 한다.
ENERGY_BALANCE_TOLERANCE_PERCENT: float = 0.1


def _all_signals() -> list[LeakSteadySignal]:
    """32조합 × 누출 3수준. 파라미터화 id 를 위해 조합·수준 라벨을 함께 쓴다."""
    return [
        signal for case in default_cdu_cases() for signal in steady_signals(case)
    ]


def _signal_id(signal: LeakSteadySignal) -> str:
    return f"{signal.case_label} | {signal.level.label}"


ALL_SIGNALS = _all_signals()


# ─────────────────────────────────────────────────────────────────────────────
# 기준 A — 부호 일관성
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("signal", ALL_SIGNALS, ids=_signal_id)
def test_criterion_a_signal_signs_are_consistent(signal: LeakSteadySignal) -> None:
    """기준 A — 32조합 × 3수준 전부에서 다섯 신호의 부호가 예상과 같다.

    부호가 어느 조합에서든 뒤집히면 그 신호는 누출의 지표가 될 수 없다.
    기대 부호와 그 이유는 이 파일 머리말의 「판정 기준」에 미리 적어 두었다.
    """
    assert signal.solvers_converged, f"{_signal_id(signal)}: solver 미수렴"

    assert signal.leak_rack_flow_change_percent < 0.0, (
        f"누출 랙 유량이 줄지 않았다: {signal.leak_rack_flow_change_percent:+.6f} %"
    )
    assert signal.leak_rack_outlet_change_C > 0.0, (
        f"누출 랙 출구온도가 오르지 않았다: {signal.leak_rack_outlet_change_C:+.6f} K"
    )
    assert signal.total_flow_change_percent < 0.0, (
        f"총유량이 줄지 않았다: {signal.total_flow_change_percent:+.6f} %"
    )
    assert signal.pump_head_change_mAq > 0.0, (
        f"펌프 양정이 오르지 않았다: {signal.pump_head_change_mAq:+.6f} mAq"
    )
    assert signal.other_rack_flow_change_percent > 0.0, (
        f"비누출 랙 유량이 늘지 않았다: "
        f"{signal.other_rack_flow_change_percent:+.6f} %"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 기준 B — 수준 간 단조
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("case", default_cdu_cases(), ids=lambda c: c.label)
def test_criterion_b_signals_are_monotone_across_levels(case) -> None:  # type: ignore[no-untyped-def]
    """기준 B — 같은 조합 안에서 +5% < +20% < +50% 로 신호 크기가 엄격히 커진다.

    커지지 않으면 신호에서 누출 정도를 되읽을 수 없다 — AI 학습 데이터셋으로서
    수준 구분이 성립하지 않는다는 뜻이므로, 크기(절대값)의 **엄격 단조**로 본다.
    """
    signals = steady_signals(case)
    assert [s.level.k_multiplier for s in signals] == [1.05, 1.2, 1.5]

    for name, values in (
        ("누출 랙 유량", [abs(s.leak_rack_flow_change_percent) for s in signals]),
        ("누출 랙 출구온도", [abs(s.leak_rack_outlet_change_C) for s in signals]),
        ("총유량", [abs(s.total_flow_change_percent) for s in signals]),
        ("펌프 양정", [abs(s.pump_head_change_mAq) for s in signals]),
        ("비누출 랙 유량", [abs(s.other_rack_flow_change_percent) for s in signals]),
    ):
        for index in range(1, len(values)):
            assert values[index] > values[index - 1], (
                f"{case.label}: {name} 이 수준 {index} 에서 커지지 않았다 "
                f"({values[index - 1]:.6g} → {values[index]:.6g})"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 기준 C — 잡음 대비 (가장 작은 누출)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "signal",
    [s for s in ALL_SIGNALS if s.level.k_multiplier == 1.05],
    ids=_signal_id,
)
def test_criterion_c_smallest_leak_exceeds_noise(signal: LeakSteadySignal) -> None:
    """기준 C — 가장 작은 누출(+5%)의 신호가 수치 잡음보다 크다.

    임계의 근거는 이 파일 상단 상수 주석에 있다 — 전부 **세션 3-A/3-B 가 실제로
    관측한 잡음**의 1000배 이상이다. 통과시키려고 고른 값이 아니다.
    """
    assert abs(signal.leak_rack_flow_change_percent) > FLOW_NOISE_THRESHOLD_PERCENT
    assert abs(signal.other_rack_flow_change_percent) > FLOW_NOISE_THRESHOLD_PERCENT
    assert abs(signal.total_flow_change_percent) > FLOW_NOISE_THRESHOLD_PERCENT
    assert abs(signal.leak_rack_outlet_change_C) > TEMPERATURE_NOISE_THRESHOLD_C
    assert abs(signal.pump_head_change_mAq) > HEAD_NOISE_THRESHOLD_mAq


# ─────────────────────────────────────────────────────────────────────────────
# 누출 상태에서의 앞 게이트 (1-B)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("signal", ALL_SIGNALS, ids=_signal_id)
def test_energy_balance_holds_under_leak(signal: LeakSteadySignal) -> None:
    """누출 상태에서도 energy balance 가 6장 기준(<0.1%) 안에 든다.

    앞 게이트가 깨진 채로 세션 4 게이트를 읽지 않는다(절대 규칙 6). K값이
    바뀌어도 보존법칙은 그대로 서야 한다.
    """
    residual_percent = signal.energy_balance_residual_percent
    assert abs(residual_percent) < ENERGY_BALANCE_TOLERANCE_PERCENT, (
        f"{_signal_id(signal)}: {residual_percent:+.6f} %"
    )


def test_normal_level_is_identity() -> None:
    """정상(배율 1.0)이 K값을 바꾸지 않는다 — 같은 코드 경로의 항등성.

    정상과 누출이 다른 경로를 타면 결과 차이가 누출 때문인지 경로 때문인지
    갈라낼 수 없다(세션 4 C2). 배율 1.0 이 부동소수점까지 항등임을 고정한다.
    """
    from cdu_simul.hydraulics import apply_leak_to_rack

    normal = leak_levels()[0]
    assert normal.is_normal
    for case in default_hydraulic_cases():
        assert apply_leak_to_rack(case, normal.k_multiplier).rack_branch_K == (
            case.rack_branch_K
        )


# ─────────────────────────────────────────────────────────────────────────────
# 동적 신호 (C4)
# ─────────────────────────────────────────────────────────────────────────────
def _leak_transient_cases() -> list[LeakStepCase]:
    """누출 3수준 × M 하한·상한. 나머지는 결과표 관례대로 범위 하단 고정."""
    hydraulic = default_hydraulic_cases()[0]
    lower, upper = holdup_bounds()
    return [
        LeakStepCase(
            label=f"{level.label} · {holdup.label}",
            holdup=holdup,
            hydraulic=hydraulic,
            k_multiplier=level.k_multiplier,
            T_secondary_supply_C=SCENARIO.T_secondary_supply_C.low,
            ntu=HEAT_EXCHANGER.ntu.low,
        )
        for holdup in (lower, upper)
        for level in leak_levels()[1:]
    ]


LEAK_TRANSIENT_CASES = _leak_transient_cases()


@pytest.mark.parametrize("case", LEAK_TRANSIENT_CASES, ids=lambda c: c.label)
def test_leak_transient_solvers_and_direction(case: LeakStepCase) -> None:
    """누출 스텝 적분이 수렴하고, 전이의 방향이 정상상태 신호와 같다.

    `solve_ivp` 의 `success` 와 매 시점 수력 `fsolve` 의 수렴을 **둘 다** 본다
    (절대 규칙 5).

    방향 기대: 누출 랙 유량 **감소** · 총유량 **감소** · 펌프 양정 **상승** ·
    T_return **상승**. 정상상태 신호와 같은 방향이어야 한다 — 다르면 동적 모델이
    정상상태와 다른 물리를 쓰고 있다는 뜻이다.

    **전이 시간 규모의 절대값은 판정하지 않는다**(#21 · #31).
    """
    result = integrate_leak_step(case)
    assert result.solver_success, (
        f"{case.label}: solve_ivp 실패 — {result.solver_message}"
    )
    assert result.hydraulic_solver_converged, f"{case.label}: 수력 fsolve 미수렴"

    assert all(math.isfinite(T) for T in result.T_supply_C)
    assert all(math.isfinite(T) for T in result.T_return_C)

    assert result.leak_rack_flow_final_Lps < result.leak_rack_flow_initial_Lps
    assert result.total_flow_final_Lps < result.total_flow_initial_Lps
    assert result.pump_head_final_mAq > result.pump_head_initial_mAq
    assert result.T_return_final_C > result.T_return_initial_C
