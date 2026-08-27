"""환경 스모크 테스트 (세션 1-A).

**이 테스트는 feasibility 게이트 판정이 아니다.** 세션 1-A에는 물리 모델이 없다
(CLAUDE.md 「개발 순서 — 게이트」 1-A 행: "게이트 없음").
여기서 보는 것은 세 가지뿐이다:
  1. 확정 스택 4종이 import 되는가
  2. Python 3.11+ 인가
  3. CoolProp 이 5장 냉각액에 대해 1차측 공급·환수 온도에서 유한한 물성을 주는가

물성값이 **물리적으로 타당한 크기인지는 판정하지 않는다** — 그 판단은 물리 모델이
생긴 뒤(1-B 이후) energy balance 검증에서 다룬다.
"""

from __future__ import annotations

import math
import sys

import pytest

from cdu_simul.assumptions import SCENARIO
from cdu_simul.fluid import coolant_cp_Jkg_K, coolant_density_kgm3


def test_python_version_is_3_11_or_newer() -> None:
    """CLAUDE.md 절대 규칙 12: Python 3.11+."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ 필요, 현재 {sys.version}"


def test_confirmed_stack_imports() -> None:
    """확정 스택 4종(NumPy·SciPy·CoolProp·Pandas) import (절대 규칙 3)."""
    import CoolProp  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import scipy  # noqa: F401


@pytest.mark.parametrize(
    "T_C",
    [SCENARIO.T_primary_supply_C, SCENARIO.T_primary_return_C],
)
def test_coolprop_returns_finite_properties_at_primary_temperatures(T_C: float) -> None:
    """5장 냉각액에 대해 1차측 공급(32℃)·환수(42℃)에서 유한한 밀도·비열 반환.

    값의 크기는 판정하지 않는다 — 조회가 성립하는지(유한한 수가 나오는지)만 본다.
    """
    rho_kgm3 = coolant_density_kgm3(T_C)
    cp_Jkg_K = coolant_cp_Jkg_K(T_C)

    assert math.isfinite(rho_kgm3), f"{T_C}℃ 에서 밀도가 유한하지 않다: {rho_kgm3}"
    assert math.isfinite(cp_Jkg_K), f"{T_C}℃ 에서 비열이 유한하지 않다: {cp_Jkg_K}"


def test_coolant_fluid_string_defined_in_assumptions_only() -> None:
    """유체 문자열의 단일 출처 확인 (collaboration.md ④).

    fluid.py 는 문자열을 자체 정의하지 않고 assumptions 의 값을 그대로 쓴다.
    """
    from cdu_simul import fluid

    # `__defaults__` 의 타입은 `tuple[Any, ...] | None` 이라 그대로 첨자하면
    # 타입검사에 걸린다. None 여부를 먼저 단언해 좁힌다 — 판정 내용은 그대로다.
    density_defaults = fluid.coolant_density_kgm3.__defaults__
    cp_defaults = fluid.coolant_cp_Jkg_K.__defaults__
    assert density_defaults is not None
    assert cp_defaults is not None

    assert density_defaults[-1] == SCENARIO.coolant_coolprop_id
    assert cp_defaults[-1] == SCENARIO.coolant_coolprop_id
