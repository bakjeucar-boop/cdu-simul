"""CoolProp 물성 래퍼 — 냉각액 물성 조회를 한 지점으로 모은다.

CLAUDE.md 코드 스타일: "CoolProp 호출은 별도 래퍼 함수로 감싸, 유체 문자열이
한 곳에서만 정의되게 한다."

**유체 문자열의 위치 결정(세션 1-A)**: `INCOMP::MPG-25%` 는 `assumptions.py` 의
`SCENARIO.coolant_coolprop_id` 에 두고, 이 모듈은 그것을 참조만 한다.
이유 — PG25 라는 선택(농도 포함)은 5장 표의 **가정치**이고 설계데이터 확보 시
교체 대상이다(예: PG30). 교체 지점이 assumptions 파일 하나여야 한다는 것이
절대 규칙 2 · collaboration.md ④ 의 요구다. fluid.py 에 문자열을 두면 교체
지점이 두 곳이 된다.

이 모듈의 함수는 전부 순수 함수다 — 전역 상태를 읽거나 쓰지 않는다.
"""

from __future__ import annotations

from CoolProp.CoolProp import PropsSI

from cdu_simul.assumptions import SCENARIO

#: 물성 조회용 기준 압력. 표준대기압 정의값(101325 Pa)이며 5장 가정치가 아니다.
#: [물리 상수: 표준대기압 정의값 — 가정값·설계값 아님]
#: 비압축성 유체(INCOMP)의 밀도·비열은 이 압력 근방에서 압력의존성이 무시할
#: 수준이므로 조회용 기준으로만 쓴다. 실제 계통 압력은 1-B 압력평형에서 다룬다.
REFERENCE_PRESSURE_Pa: float = 101325.0

#: 섭씨 → 켈빈 오프셋. 5장 표는 ℃, CoolProp API 는 K 를 쓴다.
#: **이 모듈이 ℃↔K 변환의 유일한 지점이다**(절대 규칙 9).
_CELSIUS_TO_KELVIN: float = 273.15


def celsius_to_kelvin(T_C: float) -> float:
    """섭씨를 켈빈으로 변환한다 (CoolProp 호출 경계 전용)."""
    return T_C + _CELSIUS_TO_KELVIN


def coolant_density_kgm3(
    T_C: float,
    P_Pa: float = REFERENCE_PRESSURE_Pa,
    fluid: str = SCENARIO.coolant_coolprop_id,
) -> float:
    """냉각액 밀도 [kg/m^3] 를 온도(℃)·압력(Pa)에서 조회한다."""
    return float(PropsSI("D", "T", celsius_to_kelvin(T_C), "P", P_Pa, fluid))


def coolant_cp_Jkg_K(
    T_C: float,
    P_Pa: float = REFERENCE_PRESSURE_Pa,
    fluid: str = SCENARIO.coolant_coolprop_id,
) -> float:
    """냉각액 정압비열 [J/(kg·K)] 를 온도(℃)·압력(Pa)에서 조회한다."""
    return float(PropsSI("C", "T", celsius_to_kelvin(T_C), "P", P_Pa, fluid))
