"""세션 5.8 — N-CSTR 스윕 코드의 자기정합성 확인.

**게이트가 아니다.** 6장 기준을 하나도 판정하지 않는다. 이 파일이 고정하는 것은
`transport_lag` 이라는 **별도 관측 코드**가 두 가지를 지키는가다:

1. **N=2 항등** — 현재 `dynamics.integrate_leak_step` 과 같은 해를 내는가
   (N=2 가 대조 기준이므로 이것이 깨지면 스윕 전체가 무의미하다)
2. **정상상태 불변(판정 기준 A)** — N 을 바꿔도 양 끝 정상상태가 그대로인가

전수 스윕은 `python -m cdu_simul.transport_lag` 이 돌린다. 판정 기준은
`PROCEED.md` 「세션 5.8 … 판정 기준 선기재」다.
"""

from __future__ import annotations

import numpy as np
import pytest

from cdu_simul.assumptions import HEAT_EXCHANGER, LOAD_PROFILE, SCENARIO
from cdu_simul.dynamics import LeakStepCase, holdup_bounds, integrate_leak_step
from cdu_simul.hydraulics import default_cases
from cdu_simul.transport_lag import (
    LagCase,
    integrate_leak_step_n_cstr,
    leak_signal,
)

_HOLDUPS = holdup_bounds()
#: 대표 조합 — 수력 양 끝 두 모서리에서 하나씩.
_HYDRAULICS = (default_cases()[0], default_cases()[-1])


def _case(hydraulic, n_nodes: int, holdup=_HOLDUPS[0]) -> LagCase:
    return LagCase(
        label=f"{hydraulic.label}/N={n_nodes}",
        holdup=holdup,
        hydraulic=hydraulic,
        k_multiplier=1.5,
        T_secondary_supply_C=SCENARIO.T_secondary_supply_C.low,
        ntu=HEAT_EXCHANGER.ntu.low,
        load_percent=LOAD_PROFILE.rated_load_percent,
        n_nodes=n_nodes,
    )


@pytest.mark.parametrize("hydraulic", _HYDRAULICS, ids=lambda h: h.label)
def test_two_nodes_reproduces_current_model(hydraulic) -> None:
    """N=2 는 현재 2노드 모델과 같은 해여야 한다.

    같은 저장격자 위에서 비교한다. 허용오차 1e-9 K 는 두 경로가 서로 다른 순서로
    같은 식을 계산하며 쌓는 부동소수점 차만 남긴다 — 물리 차이가 아니다.
    """
    reference = integrate_leak_step(
        LeakStepCase(
            label="reference",
            holdup=_HOLDUPS[0],
            hydraulic=hydraulic,
            k_multiplier=1.5,
            T_secondary_supply_C=SCENARIO.T_secondary_supply_C.low,
            ntu=HEAT_EXCHANGER.ntu.low,
            load_percent=LOAD_PROFILE.rated_load_percent,
        )
    )
    n_cstr = integrate_leak_step_n_cstr(_case(hydraulic, 2))

    assert n_cstr.solver_success
    assert np.array_equal(reference.t_s, n_cstr.t_s)
    assert np.max(np.abs(reference.T_supply_C - n_cstr.T_supply_C)) < 1e-9
    assert np.max(np.abs(reference.T_return_C - n_cstr.T_return_C)) < 1e-9


@pytest.mark.parametrize("n_nodes", [4, 8, 16])
@pytest.mark.parametrize("hydraulic", _HYDRAULICS, ids=lambda h: h.label)
def test_steady_states_are_node_count_invariant(hydraulic, n_nodes: int) -> None:
    """판정 기준 A — 양 끝 정상상태가 N 에 불변이어야 한다.

    노드 분할은 보유량을 나누는 것이지 정상상태 방정식을 바꾸지 않는다. 깨지면
    스윕 구현에 결함이라는 뜻이고, 그때는 멈추고 보고한다.
    """
    base = integrate_leak_step_n_cstr(_case(hydraulic, 2))
    split = integrate_leak_step_n_cstr(_case(hydraulic, n_nodes))

    assert split.solver_success

    # 누출 **전** 정상상태는 대수적으로 푼 값이라 적분오차가 섞이지 않는다 —
    # N 불변성이 여기서 **완전히** 성립해야 한다.
    assert split.T_return_initial_C == base.T_return_initial_C
    assert split.T_supply_C[0] == base.T_supply_C[0]

    # 누출 **후** 값은 30τ 까지 **적분한 끝점**이다. N 이 커지면 상태수가 늘어
    # 적분기 허용오차가 그만큼 더 쌓인다(관측: 최대 ~3e-9 K, 부호는 N 에 대해
    # 무작위 — 구조적 차이가 아니라 잡음이다). 그래서 허용오차를 적분기 잡음
    # 바닥에 맞춘다. 신호 크기(~1.6e-2 K)보다 다섯 자리 이상 작다.
    integrator_noise_K = 1.0e-7
    assert split.T_return_final_C == pytest.approx(
        base.T_return_final_C, abs=integrator_noise_K
    )

    # 정상상태가 불변이면 정상상태 사이의 신호 넷도 불변이다.
    base_signal, split_signal = leak_signal(base), leak_signal(split)
    assert split_signal.total_flow_Lps == pytest.approx(
        base_signal.total_flow_Lps, abs=integrator_noise_K
    )
    assert split_signal.pump_head_mAq == pytest.approx(
        base_signal.pump_head_mAq, abs=integrator_noise_K
    )
    assert split_signal.rack_outlet_C == pytest.approx(
        base_signal.rack_outlet_C, abs=integrator_noise_K
    )
    assert split_signal.T_return_C == pytest.approx(
        base_signal.T_return_C, abs=integrator_noise_K
    )
