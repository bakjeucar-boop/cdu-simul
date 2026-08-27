"""단일 랙 · 단일 CDU 정상상태 모델 (세션 1-B).

범위 — 4장 「단계적 확장 전략」 1단계의 최소 형태다.

    랙 1개(발열 80 kW) → 1차측 순환 → 열교환기 → 2차측 고정 경계조건

**열교환기 규모에 대한 해석**: NTU 는 무차원이므로 5장 값(2~3)을 그대로 쓰고,
유량은 단일 랙 정격을 쓴다. 즉 "랙 1개 규모로 축소한 CDU"를 본다. 이것은 4장
1단계의 해석이며 새 가정치가 아니다.

**압력-유량을 풀지 않는다.** 단일 랙이라 분배할 곳이 없다. 이것은 하이브리드
구조(절대 규칙 4)를 없애는 것이 아니라 **아직 만들지 않는 것**이다 — fsolve 기반
압력평형은 세션 3에서 도입한다. 랙 유량은 5장 정격유량을 주어진 값으로 쓴다.

**2차측 동특성을 만들지 않는다**(절대 규칙 7). 2차측 공급온도는 5장 범위의 고정
경계조건이다. 시간축(`solve_ivp`)은 이 모듈에 없다 — 세션 2다.

**모든 수치는 가정값 기반이며 실측이 아니다.** 5장 값은 assumptions.py 에서만
읽는다(절대 규칙 2) — 이 파일에 5장 숫자를 박지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.optimize import fsolve

from cdu_simul.assumptions import ASSUMPTION_TAG, HEAT_EXCHANGER, SCENARIO, VALVE
from cdu_simul.fluid import (
    coolant_cp_Jkg_K,
    coolant_density_kgm3,
    coolant_enthalpy_Jkg,
)

#: L/s → m^3/s. 5장 표는 L/s 로 적혀 있고 SI 계산은 m^3/s 를 쓴다 (절대 규칙 9).
_M3_PER_LITRE: float = 1.0e-3
#: kW → W.
_W_PER_KW: float = 1.0e3

#: cp·ρ 를 어느 온도에서 평가할지 정하는 규칙.
#: 기본값 = 1차측 벌크 평균온도
#: [가정값: 프로젝트정리 5-1 「파생 가정치」 · 세션 1-B 확정]
#: 5-1이 적은 대로 **물리 가정이 아니라 수치처리 규약**이다. 근거는
#: `solve_steady_state` docstring 참조. 인자로 남겨 둔 것은 재현·전환용이다.
CpRule = Literal["bulk_mean", "supply", "return"]
DEFAULT_CP_RULE: CpRule = "bulk_mean"


@dataclass(frozen=True)
class SteadyStateCase:
    """정상상태 계산 1건의 입력 조건. 케이스마다 새로 만들어 초기조건을 리셋한다."""

    T_secondary_supply_C: float
    ntu: float
    rack_load_kW: float
    rack_flow_Lps: float
    heat_capacity_ratio: float
    cp_rule: CpRule = DEFAULT_CP_RULE

    @property
    def label(self) -> str:
        return f"T2nd={self.T_secondary_supply_C:g}C / NTU={self.ntu:g}"


@dataclass(frozen=True)
class SteadyStateResult:
    """정상상태 해 1건. 수치 solver 성공 플래그를 함께 들고 다닌다(절대 규칙 5)."""

    case: SteadyStateCase
    T_supply_C: float
    T_return_C: float
    dT_primary_C: float
    m_dot_kgs: float
    property_eval_T_C: float
    cp_Jkg_K: float
    hx_effectiveness: float
    hx_duty_kW: float
    solver_converged: bool
    solver_message: str


def hx_effectiveness_counterflow(ntu: float, heat_capacity_ratio: float) -> float:
    """대향류 열교환기의 ε(유효도)를 NTU·열용량유량비에서 구한다 (순수 함수).

    유동배열 = **대향류**
    [가정값: 프로젝트정리 5-1 「파생 가정치」 · 세션 1-B 확정]

    근거(5-1 소절): 대향류·NTU 2·2차측 27℃ 조건에서 계산된 1차측 온도가
    32.15/42.44℃ 로 나와 5장 표의 1차측 32/42℃ 와 일치했다(세션 1-B 계산값).
    5장 표가 대향류를 상정해 만들어진 것으로 읽는다 — 새 가정 도입이 아니라
    **표에서 역으로 읽어낸 것**이다.
    """
    if heat_capacity_ratio == 1.0:
        return ntu / (1.0 + ntu)
    exponent = math.exp(-ntu * (1.0 - heat_capacity_ratio))
    return (1.0 - exponent) / (1.0 - heat_capacity_ratio * exponent)


def _state_at_property_temperature(
    property_eval_T_C: float, case: SteadyStateCase
) -> tuple[float, float, float, float, float, float]:
    """물성 평가온도가 주어졌을 때의 정상상태 온도들을 대수적으로 푼다.

    정상상태이므로 랙 발열량 = 열교환기 방열량이다. 그 조건에서

        T_return = T_2차공급 + Q / (ε · C)      (ε-NTU 관계)
        T_supply = T_return - Q / C             (랙 현열 상승)

    여기서 C = m_dot · cp [W/K] 이다. **T_return 을 위 첫 식으로 정의하므로
    ε-NTU duty 와 랙 발열량의 차는 구조상 항등적으로 0이 된다** — 그 성질은
    `hx_duty_identity_residual_percent` 에 적어 두었고 게이트 판정에 쓰지 않는다.

    반환: (T_supply_C, T_return_C, m_dot_kgs, cp_Jkg_K, ε, HX duty [W])
    """
    rho_kgm3 = coolant_density_kgm3(property_eval_T_C)
    cp_Jkg_K = coolant_cp_Jkg_K(property_eval_T_C)

    m_dot_kgs = case.rack_flow_Lps * _M3_PER_LITRE * rho_kgm3
    C_W_K = m_dot_kgs * cp_Jkg_K

    effectiveness = hx_effectiveness_counterflow(case.ntu, case.heat_capacity_ratio)
    Q_W = case.rack_load_kW * _W_PER_KW

    T_return_C = case.T_secondary_supply_C + Q_W / (effectiveness * C_W_K)
    T_supply_C = T_return_C - Q_W / C_W_K

    hx_duty_W = effectiveness * C_W_K * (T_return_C - case.T_secondary_supply_C)
    return T_supply_C, T_return_C, m_dot_kgs, cp_Jkg_K, effectiveness, hx_duty_W


def _property_temperature_from_state(
    T_supply_C: float, T_return_C: float, cp_rule: CpRule
) -> float:
    """cp·ρ 를 평가할 온도를 규칙에 따라 고른다."""
    if cp_rule == "bulk_mean":
        return 0.5 * (T_supply_C + T_return_C)
    if cp_rule == "supply":
        return T_supply_C
    if cp_rule == "return":
        return T_return_C
    raise ValueError(f"알 수 없는 cp 평가 규칙: {cp_rule}")


def solve_steady_state(case: SteadyStateCase) -> SteadyStateResult:
    """정상상태 해를 구한다.

    cp·ρ 가 온도에 따라 변하고 그 온도가 다시 해에 의존하므로 고정점 문제가 된다.
    scipy `fsolve` 로 풀고 **성공 플래그(`ier`)를 확인해** 결과에 실어 보낸다
    (절대 규칙 5).

    기본 규칙 `bulk_mean` 의 근거 [프로젝트정리 5-1 · 세션 1-B 확정]: 1차측
    흡열량의 참값은 엔탈피 적분 ∫cp(T)dT 이고, PG25 의 cp 가 이 온도대에서 거의
    선형이므로 **중점(=벌크 평균온도)에서 평가한 cp** 가 그 적분의 중점법 근사로서
    계통 편향 없이 가장 가깝다. 공급온도나 환수온도 한쪽에서 평가하면 치우친다.
    (5-1 기록: bulk_mean 0.005% 통과 · supply 0.309% 실패 · return 0.315% 실패 —
    규칙을 바꾸면 세션 1-B 게이트 판정이 뒤집힌다.)

    초기값은 5장 1차측 공급·환수 온도의 산술평균이다 — 출발점일 뿐 해를 정하지
    않는다. **케이스마다 이 함수를 새로 호출해 초기조건을 명시적으로 리셋한다**
    (collaboration.md 결함유형 ④ — 시나리오 간 상태 이월 방지).
    """

    def residual(x: list[float]) -> list[float]:
        T_prop_C = float(x[0])
        T_supply_C, T_return_C = _state_at_property_temperature(T_prop_C, case)[:2]
        rule_T_C = _property_temperature_from_state(T_supply_C, T_return_C, case.cp_rule)
        return [rule_T_C - T_prop_C]

    initial_guess_T_C = 0.5 * (
        SCENARIO.T_primary_supply_C + SCENARIO.T_primary_return_C
    )
    solution, _info, ier, message = fsolve(
        residual, [initial_guess_T_C], full_output=True
    )

    T_prop_C = float(solution[0])
    (
        T_supply_C,
        T_return_C,
        m_dot_kgs,
        cp_Jkg_K,
        effectiveness,
        hx_duty_W,
    ) = _state_at_property_temperature(T_prop_C, case)

    return SteadyStateResult(
        case=case,
        T_supply_C=T_supply_C,
        T_return_C=T_return_C,
        dT_primary_C=T_return_C - T_supply_C,
        m_dot_kgs=m_dot_kgs,
        property_eval_T_C=T_prop_C,
        cp_Jkg_K=cp_Jkg_K,
        hx_effectiveness=effectiveness,
        hx_duty_kW=hx_duty_W / _W_PER_KW,
        solver_converged=(ier == 1),
        solver_message=str(message).strip(),
    )


def energy_balance_residual_percent(result: SteadyStateResult) -> float:
    """energy balance 잔차 [%] — 세션 1-B 게이트가 판정하는 값.

    잔차 정의:

        잔차[%] = ( m_dot · [h(T_return) - h(T_supply)] - Q_rack ) / Q_rack × 100

    - 왼쪽 항: 해로 나온 두 온도에서 **CoolProp 엔탈피를 직접 조회**해 얻은 1차측
      흡열량. 모델이 해를 구할 때 쓴 경로(상수 cp 선형화, cp·ΔT)를 쓰지 않는다.
    - 오른쪽 항: 5장 랙 발열량(입력값).

    두 항이 **서로 다른 경로**로 계산되므로 이 잔차는 항등적으로 0이 아니다 —
    모델이 쓴 상수 cp 근사가 실제 엔탈피 변화와 얼마나 어긋나는지를 잰다.
    구조상 항등적으로 0인 잔차는 `hx_duty_identity_residual_percent` 쪽이며
    그것은 게이트 판정에 쓰지 않는다.
    """
    dh_Jkg = coolant_enthalpy_Jkg(result.T_return_C) - coolant_enthalpy_Jkg(
        result.T_supply_C
    )
    q_enthalpy_kW = result.m_dot_kgs * dh_Jkg / _W_PER_KW
    q_rack_kW = result.case.rack_load_kW
    return (q_enthalpy_kW - q_rack_kW) / q_rack_kW * 100.0


def hx_duty_identity_residual_percent(result: SteadyStateResult) -> float:
    """HX duty 와 랙 발열량의 차 [%] — **구조상 항등적으로 0이다.**

    `_state_at_property_temperature` 가 T_return 을 Q = ε·C·(T_return - T_2차)
    에서 역산해 정의하므로, 같은 식으로 duty 를 되돌리면 부동소수점 반올림만
    남는다. 통과해도 아무것도 증명하지 못한다 — 게이트 판정에 쓰지 않고, 항등임을
    눈으로 확인하려고 남겨둔다(C7).
    """
    q_rack_kW = result.case.rack_load_kW
    return (result.hx_duty_kW - q_rack_kW) / q_rack_kW * 100.0


def default_cases(cp_rule: CpRule = DEFAULT_CP_RULE) -> list[SteadyStateCase]:
    """5장 범위값의 **양 끝**을 조합한 4케이스 (미해결 #3 · 방침 (B)).

    2차측 공급온도 {하단, 상단} × NTU {하단, 상단}. 대표값을 고르지 않는다 —
    balance 는 보존법칙이므로 범위 안 어느 값에서도 성립해야 한다. 부하는
    100%(5장 랙당 발열량) 고정이다. 부하 0/최대 스윕은 세션 3 게이트이며 여기서
    돌리지 않는다.
    """
    return [
        SteadyStateCase(
            T_secondary_supply_C=T_secondary_C,
            ntu=ntu,
            rack_load_kW=SCENARIO.rack_it_load_kW,
            rack_flow_Lps=VALVE.rated_flow_per_rack_Lps,
            heat_capacity_ratio=HEAT_EXCHANGER.flow_ratio_primary_to_secondary,
            cp_rule=cp_rule,
        )
        for T_secondary_C in (
            SCENARIO.T_secondary_supply_C.low,
            SCENARIO.T_secondary_supply_C.high,
        )
        for ntu in (HEAT_EXCHANGER.ntu.low, HEAT_EXCHANGER.ntu.high)
    ]


def format_results_table(results: list[SteadyStateResult]) -> str:
    """4케이스 결과 표를 문자열로 만든다 (순수 함수).

    절대 규칙 11: 산출물에 "가정값 기반 — 실측 아님" 표시를 반드시 넣는다.
    """
    header = (
        f"{'case':<22}{'T_supply':>10}{'T_return':>10}{'dT':>8}"
        f"{'HX duty':>10}{'balance res':>14}{'solver':>9}"
    )
    units = (
        f"{'':<22}{'[C]':>10}{'[C]':>10}{'[K]':>8}{'[kW]':>10}{'[%]':>14}{'':>9}"
    )
    lines = [
        "세션 1-B · 단일 랙 · 단일 CDU 정상상태 결과",
        "※ " + ASSUMPTION_TAG,
        "",
        header,
        units,
        "-" * len(header),
    ]
    for r in results:
        lines.append(
            f"{r.case.label:<22}"
            f"{r.T_supply_C:>10.2f}{r.T_return_C:>10.2f}{r.dT_primary_C:>8.2f}"
            f"{r.hx_duty_kW:>10.2f}{energy_balance_residual_percent(r):>14.5f}"
            f"{('OK' if r.solver_converged else 'FAIL'):>9}"
        )
    lines += [
        "-" * len(header),
        "",
        f"물성 평가 규칙: cp·ρ 를 '{results[0].case.cp_rule}' 온도에서 평가",
        "balance 잔차: (m_dot·[h(T_return)-h(T_supply)] - Q_rack) / Q_rack × 100",
        "  — CoolProp 엔탈피 경로로 다시 계산한 흡열량과 5장 랙 발열량의 차이다.",
        "  — 모델이 해를 구할 때 쓴 상수 cp 경로와 독립이다.",
        "",
        "※ " + ASSUMPTION_TAG,
        "※ 이 표는 energy balance 만 본다. T_return 방향성·수렴시간·극단 케이스는",
        "   세션 2·3의 것이며 여기서 판정하지 않았다.",
    ]
    return "\n".join(lines)


def main() -> int:
    import sys

    if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔 기본 인코딩 대비
        sys.stdout.reconfigure(encoding="utf-8")
    results = [solve_steady_state(case) for case in default_cases()]
    print(format_results_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
