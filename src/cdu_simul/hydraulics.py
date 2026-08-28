"""수력 모듈 — 8랙 헤더 압력평형 기반 유량분배 (세션 3-A).

범위 — 4장 「단계적 확장 전략」 3단계의 **앞 절반**이다.

    펌프곡선 · 랙 분기저항 · 밸브 · 계통 잔여저항 → fsolve 로 랙별 유량

**열모델과 결합하지 않는다.** 온도는 이 모듈에 없다 — 물성(밀도)만 5장 1차측
공급·환수의 벌크 평균온도에서 한 번 조회한다. 결합은 세션 3-B다. 따라서
6장 feasibility 기준(energy balance · T_return 방향성 · 수렴시간 · 극단 케이스)은
이 모듈이 **하나도 판정하지 않는다**.

**하이브리드 구조(절대 규칙 4)의 압력-유량 쪽이다.** 압력-유량 분배는 매 시점
quasi-steady 대수방정식으로 푼다 — 이 모듈이 그 대수방정식이다. 시간적분하지
않는다.

**모든 수치는 가정값 기반이며 실측이 아니다.** 5장·5-1 값은 assumptions.py
에서만 읽는다(절대 규칙 2) — 이 파일에 숫자를 박지 않는다.

**계통 구조(프로젝트정리 5-1 「계통 잔여저항」)**::

    펌프 ──┬── 계통 잔여저항 ──┬── [헤더] ──┬── 랙1 분기 + 밸브 ──┐
           │  (집중저항 하나)   │ 저항 0 의  ├── 랙2 분기 + 밸브 ──┤
           │                    │ 공통 노드  ├── ...               │
           └────────────────────┴────────────┴─────────────────────┘

헤더에 저항을 임의로 부여하지 않는다(5-1). 잔여저항을 HX·CDU 내부·헤더로
분해하지 않는다(5-1: 5장에 개별 값이 없어 분해하지 않는다).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve

from cdu_simul.assumptions import (
    ASSUMPTION_TAG,
    PASCAL_PER_MAQ,
    PIPING,
    PUMP,
    SCENARIO,
    SESSION_3B_CAVEAT,
    VALVE,
    PumpCurveCoefficients,
)
from cdu_simul.fluid import coolant_density_kgm3, coolant_specific_gravity

#: L/s → m^3/h. 밸브 Kv 식이 m^3/h 를 쓰고 5장 유량은 L/s 다 (절대 규칙 9).
_M3H_PER_LPS: float = 3.6
#: L/s → m^3/s. 분기저항 ΔP = K·ρv²/2 는 SI 로 푼다.
_M3S_PER_LPS: float = 1.0e-3
#: mm → m.
_M_PER_MM: float = 1.0e-3
#: bar → Pa. Kv 식(ΔP 를 bar 로 받는 관례)의 변환 지점.
_PA_PER_BAR: float = 1.0e5


def bulk_mean_temperature_C(T_supply_C: float, T_return_C: float) -> float:
    """1차측 벌크 평균온도 [℃] — 수력 물성을 평가할 온도 (순수 함수).

    [규약: 프로젝트정리 5-1 「수력 계산의 물성 평가 온도」 · 세션 3-A2 확정]

    cp·ρ 평가 규칙(5-1 · 세션 1-B)을 수력에도 그대로 적용한 것이다. 추가
    파라미터가 0이며, 공급측·환수측 중 하나를 고르면 5장에 없는 선택이 된다.
    **물리 가정이 아니라 수치처리 규약**이다.

    이 함수를 받는 쪽(`valve_dp_mAq` · `branch_dp_mAq` · `solve_flow_distribution`)
    은 **온도를 인자로 요구한다** — 기본값을 두지 않는 것이 의도다. `CLAUDE.md`
    규칙 4가 압력-유량을 "매 시점 quasi-steady"로 규정하므로, 열모델이 붙는
    세션 3-B 에서는 호출부가 **풀린 상태의** 공급·환수 온도를 넣게 된다.
    """
    return 0.5 * (T_supply_C + T_return_C)


def rated_property_temperature_C() -> float:
    """정격 물성 온도 [℃] — 5장 1차측 공급·환수의 벌크평균 (37℃).

    **Kv·K·잔여저항 역산의 기준점이다.** 이 셋은 기기·형상 특성이므로 정격 물성
    에서 **한 번 역산한 뒤 고정**한다(5-1) — 온도에 따라 변하는 것은 그것들이
    아니라 그것으로 계산한 ΔP 다.
    """
    return bulk_mean_temperature_C(
        SCENARIO.T_primary_supply_C, SCENARIO.T_primary_return_C
    )

def pump_head_mAq(Q_total_Lps: float, coeffs: PumpCurveCoefficients) -> float:
    """펌프 특성곡선 H = H0 - a*Q - b*Q^2 [mAq] (순수 함수).

    Q 는 **총유량** [L/s] 이다 — 펌프는 계통에 한 대다(5장).
    계수는 5-1 전사값이며 이 함수가 만들지 않는다.
    """
    return (
        coeffs.H0_mAq
        - coeffs.a_mAq_per_Lps * Q_total_Lps
        - coeffs.b_mAq_per_Lps2 * Q_total_Lps**2
    )


def valve_dp_mAq(
    Q_rack_Lps: float,
    Kv_max_m3h: float,
    opening_fraction: float,
    T_property_C: float,
) -> float:
    """랙 밸브 차압 [mAq] (순수 함수).

    개도 특성은 **선형** Kv(x) = Kv_max·x [규약: 5-1] 이다. Kv 정의식
    Kv = Q·sqrt(SG/ΔP[bar]) 를 ΔP 에 대해 푼 것이므로 ΔP = SG·(Q/Kv)^2 [bar] 다.

    SG 는 하드코딩하지 않고 CoolProp 래퍼(`fluid.coolant_specific_gravity`)에서
    얻는다 — SG 의 정의가 그 한 곳에만 있다(절대 규칙 2).

    **`T_property_C` 에 기본값을 두지 않는다.** 호출부가 어느 온도에서 물성을
    평가하는지 명시하게 강제하는 것이 의도다(세션 3-A2). 5-1 규약대로라면
    `bulk_mean_temperature_C(...)` 를 넣는다.
    """
    if opening_fraction <= 0.0:
        raise ValueError("개도 0 이하에서는 Kv 가 0이 되어 ΔP 가 정의되지 않는다")
    Kv_m3h = Kv_max_m3h * opening_fraction
    Q_m3h = Q_rack_Lps * _M3H_PER_LPS
    dP_bar = coolant_specific_gravity(T_property_C) * (Q_m3h / Kv_m3h) ** 2
    return dP_bar * _PA_PER_BAR / PASCAL_PER_MAQ


def branch_dp_mAq(
    Q_rack_Lps: float,
    K: float,
    T_property_C: float,
    inner_diameter_mm: float = PIPING.rack_branch_inner_diameter_mm,
) -> float:
    """랙 분기 배관 차압 [mAq] — ΔP = K·ρ·v²/2 (순수 함수).

    K 는 무차원이며 **25A 내경 기준 유속으로 정의**돼 있다(5-1) — 기본 내경을
    그대로 쓴다. 다른 구경을 넘기면 K 의 정의 기준이 어긋나므로 호출자가 그
    책임을 진다.

    **`T_property_C` 에 기본값을 두지 않는다** — `valve_dp_mAq` 와 같은 이유다.
    """
    rho_kgm3 = coolant_density_kgm3(T_property_C)
    d_m = inner_diameter_mm * _M_PER_MM
    area_m2 = math.pi * d_m**2 / 4.0
    v_ms = Q_rack_Lps * _M3S_PER_LPS / area_m2
    return K * rho_kgm3 * v_ms**2 / 2.0 / PASCAL_PER_MAQ


# ─────────────────────────────────────────────────────────────────────────────
# 밸브 Kv · 배관 K — 5-1 은 값이 아니라 **역산 규칙**을 준다 (세션 3-A2)
# ─────────────────────────────────────────────────────────────────────────────
def valve_Kv_max_m3h_from_rated_dP(dP_rated_mAq: float) -> float:
    """밸브 Kv_max [m^3/h] 를 5-1 역산 규칙 그대로 산출한다.

    [역산 규칙: 프로젝트정리 5-1 「밸브 Kv (개도 100% 기준)」 · 세션 3-A2]

    5장 조건(랙당 1.94 L/s · 개도 80% · ΔP 3~5 mAq)에 Kv = Q·sqrt(SG/ΔP[bar]) 를
    적용해 정격개도에서의 Kv 를 구하고, **선형 개도 특성**으로 100% 로 환산한다::

        Kv_80  = Q · sqrt(SG / ΔP)
        Kv_100 = Kv_80 / 개도분율

    **Kv 는 기기 특성이므로 정격 물성에서 한 번 역산한 뒤 고정한다**(5-1) —
    온도에 따라 변하는 것은 Kv 가 아니라 그것으로 계산한 ΔP 다. 그래서 이 함수는
    온도를 인자로 받지 않고 `rated_property_temperature_C()` 를 쓴다.

    SG 는 `fluid.coolant_specific_gravity` 한 곳에서만 온다 — 그 결과 이 Kv 로
    정격 조건의 ΔP 를 되계산하면 **역산의 역이므로 5장 표값이 항등적으로 재현된다**
    (`test_hydraulics.py` 의 항등성 검사). 세션 3-A 는 5-1 이 SG 를 숫자로 적어
    두는 바람에 이 항등성이 깨져 있었다(미해결 #27).

    5-1 참고값: ΔP 3 mAq → Kv ≈ 16.18 · 5 mAq → Kv ≈ 12.53 m^3/h.
    **그 숫자를 코드에 박지 않는다** — 이 함수가 규칙대로 재산출한다.
    """
    Q_m3h = VALVE.rated_flow_per_rack_Lps * _M3H_PER_LPS
    dP_bar = dP_rated_mAq * PASCAL_PER_MAQ / _PA_PER_BAR
    SG = coolant_specific_gravity(rated_property_temperature_C())
    Kv_at_rated_opening_m3h = Q_m3h * math.sqrt(SG / dP_bar)
    return Kv_at_rated_opening_m3h / VALVE.rated_opening_fraction


def branch_K_from_rated_dP(
    dP_rated_mAq: float,
    inner_diameter_mm: float = PIPING.rack_branch_inner_diameter_mm,
) -> float:
    """랙 분기 배관 K값 [-] 을 5-1 역산 규칙 그대로 산출한다.

    [역산 규칙: 프로젝트정리 5-1 「배관 K값 (랙 분기)」 · 세션 3-A2]

    5장 "랙당 ΔP 2~3 mAq @ 1.94 L/s" 와 5-1 배관 내경(25A)으로
    ΔP = K·ρ·v²/2 를 K 에 대해 푼 것이다. **K 는 무차원 형상 특성이므로 정격
    물성에서 한 번 역산한 뒤 고정한다**(5-1) — `valve_Kv_max_m3h_from_rated_dP`
    와 같은 이유로 온도를 인자로 받지 않는다.

    5-1 참고값: K ≈ 3.20 ~ 4.80. **그 숫자를 코드에 박지 않는다.**
    """
    dP_at_unit_K_mAq = branch_dp_mAq(
        VALVE.rated_flow_per_rack_Lps,
        K=1.0,
        T_property_C=rated_property_temperature_C(),
        inner_diameter_mm=inner_diameter_mm,
    )
    return dP_rated_mAq / dP_at_unit_K_mAq


# ─────────────────────────────────────────────────────────────────────────────
# 케이스 정의
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HydraulicCase:
    """수력 케이스 하나. 5장·5-1 범위의 양 끝 조합을 담는다.

    `branch_K_multipliers` 는 랙별 K 배수다. 기본(None)은 전부 1.0 — 8랙 동일
    조건이다. 누출 시나리오(절대 규칙 8: K값 변화로 근사)가 이 자리에 걸리지만
    **누출 구현은 세션 4**이며, 세션 3-A 에서는 방향성 검사에만 쓴다.
    """

    label: str
    pump: PumpCurveCoefficients
    branch_K: float
    valve_Kv_max_m3h: float
    opening_fraction: float = VALVE.rated_opening_fraction
    n_racks: int = SCENARIO.racks_per_cdu
    branch_K_multipliers: tuple[float, ...] | None = None

    @property
    def rack_branch_K(self) -> tuple[float, ...]:
        """랙별 K값 [-]."""
        if self.branch_K_multipliers is None:
            return (self.branch_K,) * self.n_racks
        if len(self.branch_K_multipliers) != self.n_racks:
            raise ValueError("branch_K_multipliers 길이가 랙 수와 다르다")
        return tuple(self.branch_K * m for m in self.branch_K_multipliers)


@dataclass(frozen=True)
class FlowDistributionResult:
    """유량분배 해. solver 성공 여부를 반드시 싣는다(절대 규칙 5)."""

    case: HydraulicCase
    rack_flows_Lps: tuple[float, ...]
    total_flow_Lps: float
    pump_head_mAq: float
    residual_dp_mAq: float
    rack_branch_dp_mAq: tuple[float, ...]
    rack_valve_dp_mAq: tuple[float, ...]
    residual_coeff_mAq_per_Lps2: float
    residual_share_at_rated_percent: float
    max_abs_equation_residual_mAq: float
    solver_ier: int
    solver_message: str
    solver_converged: bool

    @property
    def mean_rack_flow_Lps(self) -> float:
        return self.total_flow_Lps / self.case.n_racks


# ─────────────────────────────────────────────────────────────────────────────
# 계통 잔여저항 — 5-1 은 값이 아니라 **규칙**을 준다
# ─────────────────────────────────────────────────────────────────────────────
def residual_resistance_coeff_mAq_per_Lps2(case: HydraulicCase) -> float:
    """계통 잔여저항 계수 [mAq/(L/s)^2] 를 5-1 규칙 그대로 역산한다.

    [역산: 프로젝트정리 5-1 「계통 잔여저항 (HX 1차측 + CDU 내부배관 + 헤더)」]

    5-1 규칙::

        잔여저항 = H정격 - (랙 분기 ΔP + 밸브 ΔP)

    이것을 **총유량 경로의 집중저항 하나**로 배정한다. 헤더는 저항 0의 공통
    노드로 본다. 난류 영역이므로 ΔP ∝ Q^2 로 두고, 계수는 정격점 한 점에서
    정해진다::

        C_res = ΔP_res,정격 / Q_총,정격^2

    잔여저항은 독립 범위가 아니라 각 조합의 **종속값**이다 — 양정·분기ΔP·밸브ΔP
    가 정해지면 잔여분도 정해지므로 조합 수가 늘지 않는다(5-1).

    **두 정격 기준이 0.13% 어긋나 있다.** 역산의 총유량 기준은 5장 펌프 정격
    15.5 L/s 이고, 랙 분기·밸브 ΔP 는 5장 표대로 랙당 1.94 L/s 기준이다.
    8 × 1.94 = 15.52 ≠ 15.5 다. **이는 5장 자체의 반올림이며 어느 쪽도 고치지
    않는다** — 한쪽을 다른 쪽에 맞추면 5장에 없는 숫자를 만드는 것이 된다.
    이 어긋남이 운전점에 얼마나 전파되는지는 `format_results_table` 이 낸다.

    **한계(5-1)**: 잔여저항이 총양정의 60~83% 를 차지한다 — 세션 3의 유량분배
    결과는 사실상 이 배정 방식이 지배한다. 수렴·비발산 게이트 통과를 "유량분배
    값이 타당하다"로 읽지 않는다.

    **정격 물성에서 한 번만 역산한다** — Kv·K 와 같은 취급이며, 물성원도 같다
    (세션 3-A2). 반환값은 **정격 밀도 기준** 계수이고, 실제 ΔP 는
    `residual_dp_mAq` 가 그때의 ρ 로 스케일해 낸다 [5-1 「계통 잔여저항의 물성
    의존」 · 세션 3-B 확정] — 잔여저항도 집중저항이므로 분기·밸브와 같은 온도
    스케일을 따른다(미해결 #30 이 이것으로 닫혔다).
    """
    Q_rack_rated_Lps = VALVE.rated_flow_per_rack_Lps
    Q_total_rated_Lps = PUMP.rated_flow_Lps
    T_rated_C = rated_property_temperature_C()

    branch_dp = branch_dp_mAq(Q_rack_rated_Lps, case.branch_K, T_rated_C)
    valve_dp = valve_dp_mAq(
        Q_rack_rated_Lps, case.valve_Kv_max_m3h, case.opening_fraction, T_rated_C
    )
    head_rated_mAq = pump_head_mAq(Q_total_rated_Lps, case.pump)
    residual_dp_rated_mAq = head_rated_mAq - (branch_dp + valve_dp)
    if residual_dp_rated_mAq <= 0.0:
        raise ValueError(
            "잔여저항이 0 이하로 역산됐다 — 5장 조건이 정격점에서 닫히지 않는다: "
            f"{case.label} (H정격 {head_rated_mAq:.3f} mAq, "
            f"분기+밸브 {branch_dp + valve_dp:.3f} mAq)"
        )
    return residual_dp_rated_mAq / Q_total_rated_Lps**2


def residual_dp_mAq(
    Q_total_Lps: float,
    residual_coeff_mAq_per_Lps2: float,
    T_property_C: float,
) -> float:
    """계통 잔여저항 차압 [mAq] — 정격 기준 계수를 그때의 ρ 로 스케일한다.

    [규약: 프로젝트정리 5-1 「계통 잔여저항의 물성 의존」 · 세션 3-B 확정]

    ΔP_res(Q, T) = C_res · Q^2 · ρ(T)/ρ(정격)

    5-1 이 잔여저항을 **집중저항**으로 규정했고 집중저항의 물리는 K·ρv²/2 이므로,
    분기 ΔP·밸브 ΔP 와 같은 온도 스케일을 따른다. 나머지 두 저항이 이미 따르는
    물리를 세 번째에 적용하는 것이라 **추가 파라미터가 0**이다.

    **정격 온도(37℃)에서는 비가 정확히 1 이므로 세션 3-A2 결과가 바뀌지 않는다**
    — 이 변경의 격리 확인이 그것이다(세션 3-B C2).
    """
    density_ratio = coolant_density_kgm3(T_property_C) / coolant_density_kgm3(
        rated_property_temperature_C()
    )
    return residual_coeff_mAq_per_Lps2 * Q_total_Lps**2 * density_ratio


def residual_share_at_rated_percent(case: HydraulicCase) -> float:
    """정격점에서 잔여저항이 총양정에서 차지하는 몫 [%] (5-1 한계의 실측치)."""
    coeff = residual_resistance_coeff_mAq_per_Lps2(case)
    Q_total_rated_Lps = PUMP.rated_flow_Lps
    head_rated_mAq = pump_head_mAq(Q_total_rated_Lps, case.pump)
    return coeff * Q_total_rated_Lps**2 / head_rated_mAq * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# 헤더 압력평형 — fsolve
# ─────────────────────────────────────────────────────────────────────────────
def solve_flow_distribution(
    case: HydraulicCase,
    T_property_C: float,
    initial_guess_Lps: tuple[float, ...] | None = None,
) -> FlowDistributionResult:
    """헤더 압력평형을 `fsolve` 로 풀어 랙별 유량을 낸다.

    헤더는 **저항 0의 공통 노드**이므로(5-1) 랙 i 마다 같은 펌프 양정을 나눠
    받는다. 미지수는 랙별 유량 Q_i [L/s] 이고 식은 랙마다 하나다::

        H_pump(ΣQ) - C_res·(ΣQ)^2 - ΔP_분기,i(Q_i) - ΔP_밸브,i(Q_i) = 0

    초기값은 5장 랙당 정격유량(1.94 L/s)이다 — 출발점일 뿐 해를 정하지 않는다.
    **케이스마다 이 함수를 새로 호출해 초기조건을 명시적으로 리셋한다**
    (collaboration.md 결함유형 ④ — 시나리오 간 상태 이월 방지).

    `T_property_C` 는 **ΔP 계산에만** 쓴다 — Kv·K·잔여저항은 정격 물성에서 이미
    고정돼 있다(5-1). 기본값을 두지 않아 호출부가 물성 온도를 명시하게 한다.

    절대 규칙 5: `ier` 를 확인해 결과에 싣고, 실패하면 조용히 넘어가지 않고
    `RuntimeError` 를 던진다.
    """
    T_C = T_property_C
    K_per_rack = case.rack_branch_K
    residual_coeff = residual_resistance_coeff_mAq_per_Lps2(case)

    def equations(Q_racks_Lps: np.ndarray) -> np.ndarray:
        Q_total_Lps = float(np.sum(Q_racks_Lps))
        available_mAq = pump_head_mAq(
            Q_total_Lps, case.pump
        ) - residual_dp_mAq(Q_total_Lps, residual_coeff, T_C)
        return np.array(
            [
                available_mAq
                - branch_dp_mAq(float(Q_i), K_i, T_C)
                - valve_dp_mAq(
                    float(Q_i), case.valve_Kv_max_m3h, case.opening_fraction, T_C
                )
                for Q_i, K_i in zip(Q_racks_Lps, K_per_rack, strict=True)
            ]
        )

    guess = (
        np.full(case.n_racks, VALVE.rated_flow_per_rack_Lps)
        if initial_guess_Lps is None
        else np.array(initial_guess_Lps, dtype=float)
    )
    solution, _info, ier, message = fsolve(equations, guess, full_output=True)
    converged = ier == 1
    if not converged:
        raise RuntimeError(
            f"fsolve 가 수렴하지 않았다 (ier={ier}): {case.label} — "
            f"{str(message).strip()}"
        )

    rack_flows = tuple(float(q) for q in solution)
    total_flow_Lps = float(sum(rack_flows))
    return FlowDistributionResult(
        case=case,
        rack_flows_Lps=rack_flows,
        total_flow_Lps=total_flow_Lps,
        pump_head_mAq=pump_head_mAq(total_flow_Lps, case.pump),
        residual_dp_mAq=residual_dp_mAq(total_flow_Lps, residual_coeff, T_C),
        rack_branch_dp_mAq=tuple(
            branch_dp_mAq(q, k, T_C)
            for q, k in zip(rack_flows, K_per_rack, strict=True)
        ),
        rack_valve_dp_mAq=tuple(
            valve_dp_mAq(q, case.valve_Kv_max_m3h, case.opening_fraction, T_C)
            for q in rack_flows
        ),
        residual_coeff_mAq_per_Lps2=residual_coeff,
        residual_share_at_rated_percent=residual_share_at_rated_percent(case),
        max_abs_equation_residual_mAq=float(np.max(np.abs(equations(solution)))),
        solver_ier=int(ier),
        solver_message=str(message).strip(),
        solver_converged=converged,
    )


def default_cases() -> list[HydraulicCase]:
    """5장·5-1 범위 양 끝의 8조합 (방침 (B) — 양 끝을 둘 다 돌린다).

    세 축이 각각 두 끝을 가진다: 펌프 정격양정(20/30 mAq) · 랙 분기 ΔP(2/3 mAq)
    · 밸브 ΔP(3/5 mAq). 2 x 2 x 2 = 8 조합이다. 중점을 고르지 않는다.

    **K 와 Kv 는 여기서 5-1 역산 규칙으로 산출해 케이스에 고정한다**(세션 3-A2)
    — 5장이 주는 것은 ΔP 범위이고 K·Kv 는 그것의 역산값이다. 케이스 생성 시점에
    한 번 계산되므로 이후 온도가 바뀌어도 다시 계산되지 않는다.
    """
    cases = []
    for pump_coeffs in PUMP.curve_coefficient_bounds:
        for branch_dP_mAq in (PIPING.dP_per_rack_mAq.low, PIPING.dP_per_rack_mAq.high):
            for valve_dP_mAq in (
                VALVE.dP_at_rated_opening_mAq.low,
                VALVE.dP_at_rated_opening_mAq.high,
            ):
                cases.append(
                    HydraulicCase(
                        label=(
                            f"H{pump_coeffs.H0_mAq:.1f}"
                            f"/dPb{branch_dP_mAq:.0f}/dPv{valve_dP_mAq:.0f}"
                        ),
                        pump=pump_coeffs,
                        branch_K=branch_K_from_rated_dP(branch_dP_mAq),
                        valve_Kv_max_m3h=valve_Kv_max_m3h_from_rated_dP(valve_dP_mAq),
                    )
                )
    return cases


def format_results_table(results: list[FlowDistributionResult]) -> str:
    """8조합 결과 표를 문자열로 만든다 (순수 함수).

    절대 규칙 11: 산출물에 "가정값 기반 — 실측 아님" 표시를 반드시 넣는다.
    """
    header = (
        f"{'case':<26}{'H_op':>8}{'branch':>9}{'valve':>8}"
        f"{'res dP':>9}{'res/H':>8}{'Q_total':>10}{'Q_rack':>9}{'dev':>9}{'solver':>8}"
    )
    units = (
        f"{'':<26}{'[mAq]':>8}{'[mAq]':>9}{'[mAq]':>8}"
        f"{'[mAq]':>9}{'[%]':>8}{'[L/s]':>10}{'[L/s]':>9}{'[%]':>9}{'':>8}"
    )
    rated_Lps = PUMP.rated_flow_Lps
    lines = [
        "세션 3-A2 · 8랙 헤더 압력평형 유량분배 (개도 80% · 8랙 동일 조건)",
        "※ " + ASSUMPTION_TAG,
        "※ 열모델과 결합하지 않았다 — 6장 feasibility 기준을 하나도 판정하지 않는다.",
        "",
        header,
        units,
        "-" * len(header),
    ]
    for r in results:
        deviation_percent = (r.total_flow_Lps - rated_Lps) / rated_Lps * 100.0
        lines.append(
            f"{r.case.label:<26}"
            f"{r.pump_head_mAq:>8.3f}{r.rack_branch_dp_mAq[0]:>9.3f}"
            f"{r.rack_valve_dp_mAq[0]:>8.3f}{r.residual_dp_mAq:>9.3f}"
            f"{r.residual_share_at_rated_percent:>8.2f}{r.total_flow_Lps:>10.4f}"
            f"{r.mean_rack_flow_Lps:>9.4f}{deviation_percent:>9.4f}"
            f"{('OK' if r.solver_converged else 'FAIL'):>8}"
        )
    shares = [r.residual_share_at_rated_percent for r in results]
    lines += [
        "-" * len(header),
        "",
        f"물성(밀도·SG) 평가 온도: {rated_property_temperature_C():.1f} ℃ "
        "(5장 1차측 공급·환수 벌크평균 · 5-1 규약)",
        "  — Kv·K·잔여저항은 이 정격 물성에서 한 번 역산해 고정했다(5-1).",
        "     ΔP 계산 온도는 인자이며, 열모델이 붙는 세션 3-B 에서 풀린 상태가 들어온다.",
        f"잔여저항 몫(정격점): {min(shares):.2f} ~ {max(shares):.2f} %"
        " — 5-1 이 적은 60~83% 와 같은 범위다.",
        "  → **유량분배 결과는 사실상 이 배정 방식이 지배한다**(5-1 한계 · 미해결 #24).",
        "     이 표의 수렴을 '유량분배 값이 타당하다'로 읽지 않는다.",
        "dev: 운전점 총유량이 5장 펌프 정격 15.5 L/s 에서 벗어난 정도.",
        "  — 잔여저항을 15.5 L/s 기준으로, 분기·밸브 ΔP 를 랙당 1.94 L/s(×8=15.52)",
        "    기준으로 역산했으므로 두 기준이 0.13% 어긋나 있다(5장 자체의 반올림).",
        "    벗어남이 크든 작든 값을 그대로 적는다 — '충분히 작다'고 판단하지 않는다.",
        "",
        "※ " + ASSUMPTION_TAG,
        SESSION_3B_CAVEAT,
        "※ 이 표는 압력-유량 분배만 본다 — 열모델과 결합한 32조합은",
        "   `python -m cdu_simul.model` 이고, 세션 3 게이트는 그쪽이 판정한다.",
    ]
    return "\n".join(lines)


def main() -> int:
    import sys

    if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔 기본 인코딩 대비
        sys.stdout.reconfigure(encoding="utf-8")
    T_property_C = rated_property_temperature_C()
    results = [
        solve_flow_distribution(case, T_property_C) for case in default_cases()
    ]
    print(format_results_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
