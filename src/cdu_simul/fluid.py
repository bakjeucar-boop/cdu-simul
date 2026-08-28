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


#: 비중(SG) 의 기준 밀도 [kg/m^3].
#: [정의값: SG 의 정의 — 가정값·설계값 아님. 5-1 빈칸 처리 순서 2번 등급]
#: SG 는 정의상 기준 물의 밀도에 대한 비이고, 배관 실무(Kv 식)의 관례 기준이
#: 1000 kg/m^3 이다. 15℃ 물의 실제 밀도(999.1)를 쓰지 않는다 — 그러면 5장에
#: 없는 숫자를 하나 더 들이게 되고, Kv 역산과 ΔP 계산 양쪽에 같은 인수로 들어가
#: 상쇄되므로 정격점 재현에도 기여하지 않는다.
SG_REFERENCE_DENSITY_kgm3: float = 1000.0


def coolant_specific_gravity(
    T_C: float,
    P_Pa: float = REFERENCE_PRESSURE_Pa,
    fluid: str = SCENARIO.coolant_coolprop_id,
) -> float:
    """냉각액 비중 SG [-] = rho(T) / 1000 (순수 함수).

    **SG 의 정의는 이 함수 하나에만 있다**(절대 규칙 2 · 코드 스타일 — CoolProp
    호출을 래퍼로 모은다). 밸브 Kv 역산과 밸브 ΔP 계산이 둘 다 여기서 SG 를
    받으므로, 두 자리에서 다른 SG 가 쓰일 수 없다(collaboration.md 결함유형 ③).

    세션 3-A 는 SG 를 5-1 에 숫자(1.0124)로 적었다가 CoolProp 값(1.01147)과
    0.09% 갈렸고, 그 결과 정격점 ΔP 가 5장 표값을 재현하지 못했다. 세션 3-A2 에서
    5-1 을 **역산 규칙**으로 바꾸고 숫자를 지웠다 — 이 함수가 그 규칙의 물성원이다.
    """
    return coolant_density_kgm3(T_C, P_Pa, fluid) / SG_REFERENCE_DENSITY_kgm3


def coolant_cp_Jkg_K(
    T_C: float,
    P_Pa: float = REFERENCE_PRESSURE_Pa,
    fluid: str = SCENARIO.coolant_coolprop_id,
) -> float:
    """냉각액 정압비열 [J/(kg·K)] 를 온도(℃)·압력(Pa)에서 조회한다."""
    return float(PropsSI("C", "T", celsius_to_kelvin(T_C), "P", P_Pa, fluid))


def coolant_enthalpy_Jkg(
    T_C: float,
    P_Pa: float = REFERENCE_PRESSURE_Pa,
    fluid: str = SCENARIO.coolant_coolprop_id,
) -> float:
    """냉각액 비엔탈피 [J/kg] 를 온도(℃)·압력(Pa)에서 조회한다.

    기준점(reference state)은 CoolProp 이 정하며 절대값 자체에는 의미가 없다 —
    **두 온도 사이의 차이**만 쓴다. energy balance 검사가 cp 선형화(cp·ΔT)와
    독립한 경로를 갖게 하려고 세션 1-B에서 추가했다.
    """
    return float(PropsSI("H", "T", celsius_to_kelvin(T_C), "P", P_Pa, fluid))
