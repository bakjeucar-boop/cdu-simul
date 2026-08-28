"""M 노드 배분 민감도 — 5-1 이 세션 4를 재검토 시점으로 지정한 항목.

프로젝트정리 5-1 「계통 보유수량 M의 노드 배분」은 **공급 50% · 환수 50%** 를
확정하면서 이렇게 적었다:

    배분비는 정상상태 해에 영향하지 않고 전이 파형에만 영향한다 — 수렴시간이
    이미 판정 불가이므로 현재 게이트를 바꾸지 않는다. 누출 신호가 전이 파형에
    걸리는 **세션 4에서 민감도를 확인**한다.

이 파일이 그 확인이다.

**40:60 과 60:40 은 새 가정치가 아니라 「시험값」이다.** 5-1 에 기록하지 않고
`assumptions.py` 에도 넣지 않는다 — 배분비를 바꿔 봤을 때 파형이 얼마나 움직이는지
보려는 것이지, 그 값을 채택하려는 것이 아니다. 그래서 이 시험 코드 안에만 둔다
(세션 4 C5).

**이 파일은 판정하지 않는다.** 50:50 을 유지할지 재검토할지는 **사람이 정한다** —
여기서는 수치만 낸다. 테스트가 고정하는 것은 두 가지뿐이다:
  ① 배분비를 바꿔도 **정상상태 해가 바뀌지 않는다**(5-1 의 진술이 실제로 성립하는가)
  ② 배분비를 바꿔도 **누출 신호의 방향이 뒤집히지 않는다**(게이트가 흔들리지 않는가)
파형이 얼마나 달라지는지의 **크기**는 `report_split_sensitivity()` 가 숫자로 낸다.
"""

from __future__ import annotations

import numpy as np
import pytest

from cdu_simul.assumptions import HEAT_EXCHANGER, SCENARIO
from cdu_simul.dynamics import (
    LeakStepCase,
    LeakTransientResult,
    holdup_bounds,
    integrate_leak_step,
)
from cdu_simul.hydraulics import default_cases as default_hydraulic_cases
from cdu_simul.leak import leak_levels

#: **시험값**이다 — 5-1 확정값(0.5)과 그 양옆. `assumptions.py` 에 넣지 않는다.
#: 40:60 · 60:40 을 고른 이유는 5-1 이 "비대칭 배분은 5장에 없는 배분비를 새로
#: 요구한다"고 적었기 때문이다 — 대칭에서 ±10%p 는 그 새 요구가 얼마나 결과를
#: 흔드는지 보기에 충분히 크고, 물리적으로 있을 법한 폭 안이다.
SUPPLY_FRACTION_TRIALS: tuple[float, ...] = (0.4, 0.5, 0.6)

#: 정상상태 해가 배분비에 의존하지 않는다는 5-1 진술의 허용오차 [K].
#: 해석적으로는 **정확히 0** 이다 — dT/dt = 0 을 넣으면 M 이 식에서 사라진다.
#: 남는 것은 적분기 잡음이라 세션 3-B 와 같은 근거로 1e-6 K 를 쓴다.
STEADY_INDEPENDENCE_TOLERANCE_C: float = 1.0e-6


def _case(
    supply_fraction: float, k_multiplier: float, holdup_index: int
) -> LeakStepCase:
    """배분비만 다른 누출 스텝 케이스. 나머지는 전부 같다."""
    holdup = holdup_bounds()[holdup_index]
    return LeakStepCase(
        label=f"공급 {supply_fraction:.0%} · K×{k_multiplier:g} · {holdup.label}",
        holdup=holdup,
        hydraulic=default_hydraulic_cases()[0],
        k_multiplier=k_multiplier,
        T_secondary_supply_C=SCENARIO.T_secondary_supply_C.low,
        ntu=HEAT_EXCHANGER.ntu.low,
        holdup_supply_fraction=supply_fraction,
    )


def _results_for(
    k_multiplier: float, holdup_index: int
) -> dict[float, LeakTransientResult]:
    return {
        fraction: integrate_leak_step(_case(fraction, k_multiplier, holdup_index))
        for fraction in SUPPLY_FRACTION_TRIALS
    }


@pytest.mark.parametrize("level", leak_levels()[1:], ids=lambda level: level.label)
def test_steady_state_is_independent_of_split(level) -> None:  # type: ignore[no-untyped-def]
    """배분비를 바꿔도 t→∞ 값이 바뀌지 않는다 — 5-1 진술의 확인.

    5-1 은 "배분비는 정상상태 해에 영향하지 않고 전이 파형에만 영향한다"고
    적었다. dT/dt = 0 을 넣으면 두 노드 방정식에서 M 이 사라지므로 해석적으로
    그렇다. 그것이 구현에서도 성립하는지 본다 — 성립하지 않으면 배분이 정상상태로
    새고 있다는 뜻이고, 그러면 5-1 의 「규약」 등급 자체가 흔들린다.
    """
    results = _results_for(level.k_multiplier, holdup_index=0)
    reference = results[0.5]
    for fraction, result in results.items():
        assert result.solver_success, f"공급 {fraction:.0%}: solve_ivp 실패"
        assert abs(
            result.T_return_final_C - reference.T_return_final_C
        ) < STEADY_INDEPENDENCE_TOLERANCE_C, (
            f"공급 {fraction:.0%}: t→∞ T_return 이 "
            f"{result.T_return_final_C - reference.T_return_final_C:+.3e} K 어긋났다"
        )


@pytest.mark.parametrize("level", leak_levels()[1:], ids=lambda level: level.label)
def test_leak_signal_direction_survives_split_change(level) -> None:  # type: ignore[no-untyped-def]
    """배분비를 바꿔도 누출 신호의 방향이 뒤집히지 않는다 — 게이트가 흔들리는가.

    세션 4 게이트(기준 A)가 배분비 규약에 의존한다면, 5-1 의 50:50 확정이
    게이트 판정을 좌우한다는 뜻이 된다. 그렇지 않음을 고정한다.
    """
    for fraction, result in _results_for(level.k_multiplier, holdup_index=0).items():
        assert result.T_return_final_C > result.T_return_initial_C, (
            f"공급 {fraction:.0%}: T_return 이 오르지 않았다"
        )
        assert result.total_flow_final_Lps < result.total_flow_initial_Lps
        assert result.pump_head_final_mAq > result.pump_head_initial_mAq


def report_split_sensitivity() -> str:
    """배분비별 전이 파형 차이를 **수치로만** 낸다 — 판정은 사람이 한다.

    비교 지표는 셋이다. 전부 50:50 을 기준으로 한 차이다.
      · t63·t95 — 스텝 전체 변화량의 63%·95% 도달 시각 [s]
      · 최대 편차 — 두 파형의 T_return 을 같은 시각에서 뺀 값의 최대 절대값 [K]
    **결론을 문서에 쓰지 않는다**(세션 4 C5).
    """
    from cdu_simul.dynamics import time_to_fraction_of_step_s

    lines = ["M 노드 배분 민감도 (시험값 — 5-1·assumptions.py 에 넣지 않는다)", ""]
    for holdup_index, holdup_label in ((0, "M 하한"), (1, "M 상한")):
        for level in leak_levels()[1:]:
            results = _results_for(level.k_multiplier, holdup_index)
            reference = results[0.5]
            lines.append(f"[{holdup_label} · {level.label}]")
            for fraction in SUPPLY_FRACTION_TRIALS:
                result = results[fraction]
                t63 = time_to_fraction_of_step_s(result, 0.63)
                t95 = time_to_fraction_of_step_s(result, 0.95)
                max_gap_C = float(
                    np.max(np.abs(result.T_return_C - reference.T_return_C))
                )
                lines.append(
                    f"  공급 {fraction:.0%} : "
                    f"t63={t63:.4f} s · t95={t95:.4f} s · "
                    f"50:50 대비 최대 편차 {max_gap_C:.3e} K · "
                    f"t→∞ {result.T_return_final_C:.9f} ℃"
                )
            lines.append("")
    return "\n".join(lines)
