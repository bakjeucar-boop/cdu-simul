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
from cdu_simul.dynamics import (
    INTEGRATION_RTOL,
    LeakStepCase,
    holdup_bounds,
    integrate_leak_step,
)
from cdu_simul.hydraulics import default_cases
from cdu_simul.transport_lag import (
    LagCase,
    integrate_leak_step_n_cstr,
    leak_signal,
)

#: N=2 항등 비교의 여유 배수 [세션 7.72]. `solve_ivp` 가 **매 스텝** 죄는 것은
#: 국소오차 `rtol·|y| + atol` 이고, 두 경로가 서로 다른 스텝 열을 밟으며 그것을
#: 수천 번 쌓으므로 전역 차는 국소허용오차보다 크다. 몇 배까지 쌓이는지는 식으로
#: 세울 수 없어 **안전배수를 밝혀 적는다** — 실측 차는 상대로 1.1 × rtol 이다.
_TOL_SAFETY: float = 10.0

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

    같은 저장격자 위에서 비교한다. 허용오차는 두 경로가 서로 다른 순서로 같은
    식을 계산하며 쌓는 차만 남긴다 — 물리 차이가 아니다.

    **임계를 적분기 설정에서 세운다**(세션 7.72) — 관측값을 박지 않는다. 두 경로는
    서로 다른 적응 스텝 열을 밟으므로 남는 차는 `solve_ivp` 자신의 상대허용오차
    규모, 즉 `INTEGRATION_RTOL × |T|` 다. 세션 7.71 까지는 손으로 고른 1e-9 K 로
    두어도 넉넉했으나, UA 고정(#70)으로 ε 가 상태에 더 민감해지면서 두 스텝 열의
    벌어짐이 그 규모까지 커졌다 — **기준(「두 경로가 같은 해다」)은 그대로이고
    여유를 손으로 고르던 것을 적분기 설정에서 유도하도록 바꾼 것이다.**
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
    tol_K = _TOL_SAFETY * INTEGRATION_RTOL * float(np.max(np.abs(reference.T_return_C)))
    assert np.max(np.abs(reference.T_supply_C - n_cstr.T_supply_C)) < tol_K
    assert np.max(np.abs(reference.T_return_C - n_cstr.T_return_C)) < tol_K


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
