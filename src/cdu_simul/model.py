"""다중 랙 · 단일 CDU 정상상태 모델 (세션 1-B → 세션 3-B 에서 8랙 확장).

범위 — 4장 「단계적 확장 전략」 2단계다.

    랙 N개(5장: 8개) → 유량가중 합류 → 1차측 순환 → 열교환기 → 2차측 고정 경계

**랙 수는 `assumptions.py` 에서 읽는다**(5장 `racks_per_cdu`) — 이 파일에 숫자를
박지 않는다. 세션 1-B 는 랙 1개였고, 그때의 식은 N=1 의 특수경우로 그대로 남아
있다(랙이 동일하면 합류식이 1랙 식과 같아진다 — `_state_at_property_temperature`).

**열교환기 규모에 대한 해석**: NTU 는 무차원이므로 5장 값(2~3)을 그대로 쓴다.
유량이 N배가 되면 UA 도 함께 커지는 것으로 읽는 것이며, 5장이 NTU 를 무차원으로
준 이상 새 가정치가 아니다. 그 결과 **동일 랙 N개의 온도해는 1랙 해와 같다** —
Q 와 C 가 같은 배수로 커지기 때문이다.

**압력-유량은 `hydraulics.py` 가 푼다**(절대 규칙 4 — 하이브리드 구조).
`solve_cdu_steady_state` 가 물성 온도 고정점 안에서 매번 헤더 압력평형을
quasi-steady 로 풀어 랙별 유량을 받는다. 온도는 이 모듈이 대수적으로 풀고,
시간적분은 `dynamics.py` 다.

**2차측 동특성을 만들지 않는다**(절대 규칙 7). 2차측 공급온도는 5장 범위의 고정
경계조건이다.

**부하는 CDU 전체 일괄이다** [규약: 5-1 「부하 프로파일의 랙 배분」 · 세션 3-B].
8랙에 동일 부하를 준다 — 랙별 배분비는 5장에 없다. 랙 간 비대칭은 누출 K값이
들어오는 세션 4에서 처음 생긴다.

**모든 수치는 가정값 기반이며 실측이 아니다.** 5장 값은 assumptions.py 에서만
읽는다(절대 규칙 2) — 이 파일에 5장 숫자를 박지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.optimize import fsolve

from cdu_simul.assumptions import (
    ASSUMPTION_TAG,
    HEAT_EXCHANGER,
    LOAD_PROFILE,
    SCENARIO,
    SESSION_3B_CAVEAT,
    SESSION_5B_CAVEAT,
    VALVE,
)
from cdu_simul.fluid import (
    coolant_cp_Jkg_K,
    coolant_density_kgm3,
    coolant_enthalpy_Jkg,
)
from cdu_simul.hydraulics import (
    FlowDistributionResult,
    HydraulicCase,
    bulk_mean_temperature_C,
    solve_flow_distribution,
)
from cdu_simul.hydraulics import default_cases as default_hydraulic_cases

#: L/s → m^3/s. 5장 표는 L/s 로 적혀 있고 SI 계산은 m^3/s 를 쓴다 (절대 규칙 9).
_M3_PER_LITRE: float = 1.0e-3
#: kW → W.
_W_PER_KW: float = 1.0e3
#: % → 분율.
_PERCENT: float = 1.0e-2

#: cp·ρ 를 어느 온도에서 평가할지 정하는 규칙.
#: 기본값 = 1차측 벌크 평균온도
#: [가정값: 프로젝트정리 5-1 「파생 가정치」 · 세션 1-B 확정]
#: 5-1이 적은 대로 **물리 가정이 아니라 수치처리 규약**이다. 근거는
#: `solve_steady_state` docstring 참조. 인자로 남겨 둔 것은 재현·전환용이다.
CpRule = Literal["bulk_mean", "supply", "return"]
DEFAULT_CP_RULE: CpRule = "bulk_mean"


@dataclass(frozen=True)
class SteadyStateCase:
    """정상상태 계산 1건의 입력 조건. 케이스마다 새로 만들어 초기조건을 리셋한다.

    **랙별 튜플이다**(세션 3-B). 세션 1-B 는 스칼라 1랙이었고, 그것은 길이 1
    튜플의 특수경우로 그대로 표현된다 — 물리식을 바꾼 것이 아니라 랙 축을 편 것이다.
    `uniform(...)` 이 5장 랙 수만큼 동일 랙을 만드는 통상 경로다.
    """

    T_secondary_supply_C: float
    ntu: float
    rack_loads_kW: tuple[float, ...]
    rack_flows_Lps: tuple[float, ...]
    #: 이 CDU 에 배분된 2차측 **부피**유량 [L/s].
    #: **Cr 은 이 유량과 양측 물성에서 유도된다 — 선언하지 않는다**
    #: [규약: 프로젝트정리 5-1 「2차측 유체」 · 세션 5-B 확정].
    #: 세션 1-B~5 는 `heat_capacity_ratio=1` 을 필드로 들고 다녔는데, 그것이
    #: 세션 5 C4 가 드러낸 오류다(5장 「1:1」은 부피유량비이지 Cr 이 아니다).
    secondary_flow_Lps: float = HEAT_EXCHANGER.secondary_flow_Lps
    cp_rule: CpRule = DEFAULT_CP_RULE

    def __post_init__(self) -> None:
        if len(self.rack_loads_kW) != len(self.rack_flows_Lps):
            raise ValueError("랙별 부하와 유량의 길이가 다르다")
        if not self.rack_loads_kW:
            raise ValueError("랙이 하나도 없다")

    @classmethod
    def uniform(
        cls,
        T_secondary_supply_C: float,
        ntu: float,
        rack_load_kW: float,
        rack_flow_Lps: float,
        n_racks: int = SCENARIO.racks_per_cdu,
        cp_rule: CpRule = DEFAULT_CP_RULE,
        secondary_flow_Lps: float = HEAT_EXCHANGER.secondary_flow_Lps,
    ) -> SteadyStateCase:
        """동일한 랙 `n_racks` 개로 케이스를 만든다.

        기본 랙 수는 5장 `racks_per_cdu` 다 — 이 파일에 숫자를 박지 않는다.
        부하 일괄 배분은 5-1 「부하 프로파일의 랙 배분」 규약이다.
        """
        return cls(
            T_secondary_supply_C=T_secondary_supply_C,
            ntu=ntu,
            rack_loads_kW=(rack_load_kW,) * n_racks,
            rack_flows_Lps=(rack_flow_Lps,) * n_racks,
            secondary_flow_Lps=secondary_flow_Lps,
            cp_rule=cp_rule,
        )

    @property
    def n_racks(self) -> int:
        return len(self.rack_loads_kW)

    @property
    def total_load_kW(self) -> float:
        return sum(self.rack_loads_kW)

    @property
    def total_flow_Lps(self) -> float:
        return sum(self.rack_flows_Lps)

    @property
    def label(self) -> str:
        return f"T2nd={self.T_secondary_supply_C:g}C / NTU={self.ntu:g}"


@dataclass(frozen=True)
class SteadyStateResult:
    """정상상태 해 1건. 수치 solver 성공 플래그를 함께 들고 다닌다(절대 규칙 5)."""

    case: SteadyStateCase
    T_supply_C: float
    T_return_C: float
    rack_return_temps_C: tuple[float, ...]
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
    # Cr = 1 은 아래 일반식이 0/0 이 되는 **수학적 특이점**이라 극한식을 쓴다.
    # **가정으로서의 Cr=1 은 세션 5-B 에서 전부 제거됐다** — 여기 남은 것은 유도된
    # Cr 이 우연히 정확히 1 이 될 때(양측 온도·유량이 같아지는 극단 케이스, 예:
    # 부하 0)를 위한 가드이지 5장의 「1:1」을 되살리는 경로가 아니다.
    if heat_capacity_ratio == 1.0:
        return ntu / (1.0 + ntu)
    exponent = math.exp(-ntu * (1.0 - heat_capacity_ratio))
    return (1.0 - exponent) / (1.0 - heat_capacity_ratio * exponent)


@dataclass(frozen=True)
class _PrimaryState:
    """`_state_at_property_temperature` 의 반환 묶음 (내부용)."""

    T_supply_C: float
    T_return_C: float
    rack_return_temps_C: tuple[float, ...]
    m_dot_kgs: float
    cp_Jkg_K: float
    effectiveness: float
    hx_duty_W: float


def hx_capacity_terms(
    C_primary_W_K: float,
    ntu: float,
    T_secondary_supply_C: float,
    secondary_flow_Lps: float,
) -> tuple[float, float]:
    """열교환기 유효도 ε 와 C_min 을 낸다 (순수 함수). 반환: (ε, C_min [W/K]).

    **물리를 한 곳에만 적는다**(collaboration.md ④) — 정상상태(`model`)와
    시간적분(`dynamics`)이 둘 다 이 함수를 쓴다.

    **Cr 을 선언하지 않고 매번 유도한다**
    [규약: 프로젝트정리 5-1 「2차측 유체」 · 세션 5-B 확정]::

        C_2차 = Q_2차 · ρ(T_2차공급) · cp(T_2차공급)
        Cr    = C_min / C_max          (ε-NTU 정의)

    5-1 「2차측 유체」가 정하는 것은 셋이다: 2차측 유체는 **1차측과 동일한 PG25**,
    5장 「1차:2차 유량비 1:1」은 **부피유량 기준**으로 읽는다, 2차측 부피유량은
    1차측 **정격** 15.5 L/s 에 고정한다. Cr 은 그 셋에서 **유도**되는 값이지
    5-1 이 주는 값이 아니다.

    세션 1-B~5 는 5장 「1차:2차 유량비 1:1」을 Cr=1 로 **선언해** 썼다. 그것이
    세션 5 C4 가 드러낸 오류다 — 부피유량이 같아도 1차측(벌크 ~37℃)과 2차측
    (27~30℃)의 ρ·cp 가 달라 Cr ≈ 0.9986 이 된다. 이 함수에는 이제 **Cr=1 을
    가정하는 경로가 없다.**

    **어느 쪽이 C_min 인지도 매번 판정한다.** (ρcp)₂₇ < (ρcp)₃₇ 이므로 부피유량이
    같으면 **2차측이 C_min** 이다 — 세션 5-B 이전 코드는 1차측을 C_min 으로
    가정하고 있었다.

    **2차측 물성을 공급온도에서 평가한다.** 2차측 출구온도는 모델에 없으므로
    (5-1 「2차측 공급온도」 — 냉각탑을 모델링하지 않는다) 벌크평균을 만들 수 없다.
    선택이 아니라 강제다. 5-1 의 cp·ρ 벌크평균 규약은 1차측에만 적용된다.
    """
    C_secondary_W_K = (
        secondary_flow_Lps
        * _M3_PER_LITRE
        * coolant_density_kgm3(T_secondary_supply_C)
        * coolant_cp_Jkg_K(T_secondary_supply_C)
    )
    C_min_W_K = min(C_primary_W_K, C_secondary_W_K)
    C_max_W_K = max(C_primary_W_K, C_secondary_W_K)
    return (
        hx_effectiveness_counterflow(ntu, C_min_W_K / C_max_W_K),
        C_min_W_K,
    )


def _state_at_property_temperature(
    property_eval_T_C: float, case: SteadyStateCase
) -> _PrimaryState:
    """물성 평가온도가 주어졌을 때의 정상상태 온도들을 대수적으로 푼다.

    랙 N개가 공급 헤더에서 갈라져 각자 가열된 뒤 환수 헤더에서 **유량가중으로
    혼합**된다::

        T_return,i = T_supply + Q_i / (m_dot_i · cp)      (랙 i 현열 상승)
        T_return   = Σ(m_dot_i · T_return,i) / Σ m_dot_i  (혼합)
                   = T_supply + ΣQ_i / (Σm_dot_i · cp)

    cp 가 랙 간 공통이므로 혼합 결과는 **총 발열량과 총 유량만으로 결정된다** —
    랙이 동일하든 아니든 그렇다. 그래서 아래는 총량으로 풀고, 랙별 환수온도는
    해가 나온 뒤 되돌려 계산한다(누출로 랙이 갈라지는 세션 4에서 쓸 값이다).

    정상상태이므로 총 발열량 = 열교환기 방열량이다. 그 조건에서

        T_return = T_2차공급 + Q_총 / (ε · C_총)   (ε-NTU 관계)
        T_supply = T_return - Q_총 / C_총          (현열 상승)

    여기서 C_총 = m_dot_총 · cp [W/K] 이다. **T_return 을 위 첫 식으로 정의하므로
    ε-NTU duty 와 랙 발열량의 차는 구조상 항등적으로 0이 된다** — 그 성질은
    `hx_duty_identity_residual_percent` 에 적어 두었고 게이트 판정에 쓰지 않는다.
    """
    rho_kgm3 = coolant_density_kgm3(property_eval_T_C)
    cp_Jkg_K = coolant_cp_Jkg_K(property_eval_T_C)

    m_dot_kgs = case.total_flow_Lps * _M3_PER_LITRE * rho_kgm3
    C_W_K = m_dot_kgs * cp_Jkg_K

    effectiveness, C_min_W_K = hx_capacity_terms(
        C_W_K,
        case.ntu,
        case.T_secondary_supply_C,
        case.secondary_flow_Lps,
    )
    Q_W = case.total_load_kW * _W_PER_KW

    T_return_C = case.T_secondary_supply_C + Q_W / (effectiveness * C_min_W_K)
    T_supply_C = T_return_C - Q_W / C_W_K

    rack_return_temps_C = tuple(
        T_supply_C
        + load_kW * _W_PER_KW / (flow_Lps * _M3_PER_LITRE * rho_kgm3 * cp_Jkg_K)
        for load_kW, flow_Lps in zip(
            case.rack_loads_kW, case.rack_flows_Lps, strict=True
        )
    )

    hx_duty_W = effectiveness * C_min_W_K * (T_return_C - case.T_secondary_supply_C)
    return _PrimaryState(
        T_supply_C=T_supply_C,
        T_return_C=T_return_C,
        rack_return_temps_C=rack_return_temps_C,
        m_dot_kgs=m_dot_kgs,
        cp_Jkg_K=cp_Jkg_K,
        effectiveness=effectiveness,
        hx_duty_W=hx_duty_W,
    )


def property_temperature_from_state(
    T_supply_C: float, T_return_C: float, cp_rule: CpRule
) -> float:
    """cp·ρ 를 평가할 온도를 규칙에 따라 고른다 (순수 함수).

    **공개 이름이다.** `dynamics` 가 이 함수를 쓴다 — cp 평가 규칙
    [프로젝트정리 5-1 「cp·ρ 평가 온도」]을 정상상태와 시간적분 두 곳에 적지
    않으려는 것이 원래 의도이고(collaboration.md ④ 「물리를 한 곳에만 적는다」),
    세션 2 는 `model.py` 수정 금지라 비공개 이름인 채로 import 하고 있었다
    (미해결 #22). 세션 5.7 에서 이름만 승격했다 — **본문은 그대로다.**
    """
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
        state = _state_at_property_temperature(T_prop_C, case)
        rule_T_C = property_temperature_from_state(
            state.T_supply_C, state.T_return_C, case.cp_rule
        )
        return [rule_T_C - T_prop_C]

    initial_guess_T_C = 0.5 * (
        SCENARIO.T_primary_supply_C + SCENARIO.T_primary_return_C
    )
    solution, _info, ier, message = fsolve(
        residual, [initial_guess_T_C], full_output=True
    )

    T_prop_C = float(solution[0])
    state = _state_at_property_temperature(T_prop_C, case)

    return SteadyStateResult(
        case=case,
        T_supply_C=state.T_supply_C,
        T_return_C=state.T_return_C,
        rack_return_temps_C=state.rack_return_temps_C,
        dT_primary_C=state.T_return_C - state.T_supply_C,
        m_dot_kgs=state.m_dot_kgs,
        property_eval_T_C=T_prop_C,
        cp_Jkg_K=state.cp_Jkg_K,
        hx_effectiveness=state.effectiveness,
        hx_duty_kW=state.hx_duty_W / _W_PER_KW,
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
    q_rack_kW = result.case.total_load_kW
    if q_rack_kW == 0.0:
        raise ValueError(
            "부하 0 에서는 상대 잔차가 정의되지 않는다 (0 으로 나눈다) — "
            "극단 케이스(6장)는 비발산으로 판정하고 이 잔차를 쓰지 않는다"
        )
    q_enthalpy_kW = result.m_dot_kgs * dh_Jkg / _W_PER_KW
    return (q_enthalpy_kW - q_rack_kW) / q_rack_kW * 100.0


def hx_duty_identity_residual_percent(result: SteadyStateResult) -> float:
    """HX duty 와 랙 발열량의 차 [%] — **구조상 항등적으로 0이다.**

    `_state_at_property_temperature` 가 T_return 을 Q = ε·C·(T_return - T_2차)
    에서 역산해 정의하므로, 같은 식으로 duty 를 되돌리면 부동소수점 반올림만
    남는다. 통과해도 아무것도 증명하지 못한다 — 게이트 판정에 쓰지 않고, 항등임을
    눈으로 확인하려고 남겨둔다(C7).
    """
    q_rack_kW = result.case.total_load_kW
    if q_rack_kW == 0.0:
        raise ValueError("부하 0 에서는 상대 잔차가 정의되지 않는다 (0 으로 나눈다)")
    return (result.hx_duty_kW - q_rack_kW) / q_rack_kW * 100.0


def default_cases(cp_rule: CpRule = DEFAULT_CP_RULE) -> list[SteadyStateCase]:
    """5장 범위값의 **양 끝**을 조합한 4케이스 (미해결 #3 · 방침 (B)).

    2차측 공급온도 {하단, 상단} × NTU {하단, 상단}. 대표값을 고르지 않는다 —
    balance 는 보존법칙이므로 범위 안 어느 값에서도 성립해야 한다. 부하는
    100%(5장 랙당 발열량) 고정이고 랙 수는 5장 `racks_per_cdu`(8개)다.

    **유량은 5장 랙당 정격(1.94 L/s)을 그대로 쓴다 — 수력을 풀지 않는다.**
    수력과 결합한 32조합은 `default_cdu_cases()` 이고 세션 3 게이트는 그쪽이
    판정한다. 이 4케이스는 수력과 무관하게 열식만 보는 세션 1-B 의 자리를
    유지하려고 남긴다(랙 수만 5장대로 8개가 됐다).
    """
    return [
        SteadyStateCase.uniform(
            T_secondary_supply_C=T_secondary_C,
            ntu=ntu,
            rack_load_kW=SCENARIO.rack_it_load_kW,
            rack_flow_Lps=VALVE.rated_flow_per_rack_Lps,
            cp_rule=cp_rule,
        )
        for T_secondary_C in (
            SCENARIO.T_secondary_supply_C.low,
            SCENARIO.T_secondary_supply_C.high,
        )
        for ntu in (HEAT_EXCHANGER.ntu.low, HEAT_EXCHANGER.ntu.high)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 수력↔열 결합 (세션 3-B) — 하이브리드 구조의 두 반쪽을 물린다 (절대 규칙 4)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CduCase:
    """CDU 1대 정상상태 케이스 — 수력 조건 + 열 조건.

    부하는 **CDU 전체 일괄**이다 [규약: 5-1 「부하 프로파일의 랙 배분」 · 세션 3-B]
    — `load_percent` 를 8랙에 동일하게 나눈다. 랙별 배분비는 5장에 없다.
    """

    hydraulic: HydraulicCase
    T_secondary_supply_C: float
    ntu: float
    load_percent: float = LOAD_PROFILE.rated_load_percent
    cp_rule: CpRule = DEFAULT_CP_RULE

    @property
    def rack_load_kW(self) -> float:
        """랙당 발열량 [kW] — 5장 랙당 발열량 × 부하율."""
        return SCENARIO.rack_it_load_kW * self.load_percent * _PERCENT

    @property
    def label(self) -> str:
        return (
            f"{self.hydraulic.label} / NTU={self.ntu:g}"
            f" / T2nd={self.T_secondary_supply_C:g}C"
            f" / load={self.load_percent:g}%"
        )


@dataclass(frozen=True)
class CduSteadyStateResult:
    """결합 해 1건. 수력·열 두 solver 의 성공 플래그를 함께 싣는다(절대 규칙 5)."""

    case: CduCase
    thermal: SteadyStateResult
    flow: FlowDistributionResult
    property_eval_T_C: float
    outer_solver_converged: bool
    outer_solver_message: str

    @property
    def solver_converged(self) -> bool:
        """수력·열·바깥 고정점이 **전부** 수렴했는가."""
        return (
            self.outer_solver_converged
            and self.thermal.solver_converged
            and self.flow.solver_converged
        )


def cdu_thermal_case_at(
    case: CduCase,
    T_property_C: float,
    secondary_flow_Lps: float = HEAT_EXCHANGER.secondary_flow_Lps,
) -> tuple[SteadyStateCase, FlowDistributionResult]:
    """물성 온도가 주어졌을 때의 열 케이스와 수력 해를 만든다 (순수 함수).

    수력은 그 온도에서 **매번 quasi-steady 로 다시 푼다**(절대 규칙 4).
    상위 레벨(다중 CDU)이 이 함수를 통해 CDU 하나의 상태를 들여다본다 —
    `plant.py` 가 비공개 이름을 import 하지 않도록 공개해 둔다.
    """
    flow = solve_flow_distribution(case.hydraulic, T_property_C)
    thermal_case = SteadyStateCase(
        T_secondary_supply_C=case.T_secondary_supply_C,
        ntu=case.ntu,
        rack_loads_kW=(case.rack_load_kW,) * case.hydraulic.n_racks,
        rack_flows_Lps=flow.rack_flows_Lps,
        secondary_flow_Lps=secondary_flow_Lps,
        cp_rule=case.cp_rule,
    )
    return thermal_case, flow


def cdu_property_temperature_residual(
    case: CduCase,
    T_property_C: float,
    secondary_flow_Lps: float = HEAT_EXCHANGER.secondary_flow_Lps,
) -> float:
    """물성 온도 고정점의 잔차 [K] — 규칙이 주는 온도와 넣은 온도의 차 (순수 함수).

    0 이면 그 온도가 이 CDU 의 자기정합 해다. 상위 레벨 연립방정식은 CDU 마다
    이 잔차를 하나씩 세워 **동시에** 0으로 만든다(`plant.solve_plant_steady_state`).
    """
    thermal_case, _flow = cdu_thermal_case_at(case, T_property_C, secondary_flow_Lps)
    state = _state_at_property_temperature(T_property_C, thermal_case)
    rule_T_C = property_temperature_from_state(
        state.T_supply_C, state.T_return_C, case.cp_rule
    )
    return rule_T_C - T_property_C


def solve_cdu_steady_state(
    case: CduCase,
    secondary_flow_Lps: float = HEAT_EXCHANGER.secondary_flow_Lps,
) -> CduSteadyStateResult:
    """수력과 열을 결합해 CDU 정상상태를 푼다.

    구조 (절대 규칙 4 — 하이브리드)::

        바깥 fsolve: 물성 평가온도 T_prop 의 고정점
          └ 안쪽 fsolve: 그 T_prop 에서 헤더 압력평형 → 랙별 유량 (quasi-steady)
              └ 대수식: 그 유량으로 정상상태 온도

    압력-유량은 **매번 quasi-steady 대수방정식**으로 다시 풀고, 온도는 대수적으로
    (정상상태이므로) 푼다. 시간적분은 이 함수에 없다 — `dynamics.py` 다.

    물성 온도는 5-1 규약대로 1차측 벌크 평균온도이며, 수력과 열이 **같은 온도를**
    본다 — 세션 3-A2 가 수력 쪽 물성 온도를 인자로 뺀 것이 이 자리를 만들기
    위해서였다(미해결 #28).

    초기값은 5장 1차측 공급·환수의 산술평균이다 — 출발점일 뿐 해를 정하지 않는다.
    **케이스마다 이 함수를 새로 호출해 초기조건을 명시적으로 리셋한다**
    (collaboration.md 결함유형 ④ — 시나리오 간 상태 이월 방지).

    절대 규칙 5: 바깥 `fsolve` 의 `ier`, 안쪽 수력 `fsolve` 의 `ier`, 열 쪽
    `fsolve` 의 `ier` 를 **전부** 확인한다. 수력이 실패하면
    `solve_flow_distribution` 이 예외를 던지고, 나머지는 결과에 플래그로 실린다.
    """

    def residual(x: list[float]) -> list[float]:
        return [
            cdu_property_temperature_residual(
                case, float(x[0]), secondary_flow_Lps
            )
        ]

    initial_guess_T_C = bulk_mean_temperature_C(
        SCENARIO.T_primary_supply_C, SCENARIO.T_primary_return_C
    )
    solution, _info, ier, message = fsolve(
        residual, [initial_guess_T_C], full_output=True
    )

    T_prop_C = float(solution[0])
    thermal_case, flow = cdu_thermal_case_at(case, T_prop_C, secondary_flow_Lps)
    thermal = solve_steady_state(thermal_case)
    return CduSteadyStateResult(
        case=case,
        thermal=thermal,
        flow=flow,
        property_eval_T_C=T_prop_C,
        outer_solver_converged=(ier == 1),
        outer_solver_message=str(message).strip(),
    )


def default_cdu_cases(
    load_percent: float = LOAD_PROFILE.rated_load_percent,
) -> list[CduCase]:
    """5장·5-1 범위 양 끝의 **32조합** (방침 (B) — 양 끝을 둘 다 돌린다).

    수력 8조합(양정 2 × 분기ΔP 2 × 밸브ΔP 2) × NTU 2 × 2차측 2 = 32.
    중점을 고르지 않는다. 부하는 인자로 받는다 — 극단 케이스(0% / 100%)를
    같은 32조합 위에서 돌리기 위해서다(6장 발산 검사).
    """
    return [
        CduCase(
            hydraulic=hydraulic,
            T_secondary_supply_C=T_secondary_C,
            ntu=ntu,
            load_percent=load_percent,
        )
        for hydraulic in default_hydraulic_cases()
        for ntu in (HEAT_EXCHANGER.ntu.low, HEAT_EXCHANGER.ntu.high)
        for T_secondary_C in (
            SCENARIO.T_secondary_supply_C.low,
            SCENARIO.T_secondary_supply_C.high,
        )
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
        f"열식 단독 정상상태 (랙 {results[0].case.n_racks}개 · 유량 5장 정격 고정)",
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
        "※ 이 표는 수력을 풀지 않는다 — 유량이 5장 정격 고정이다.",
        "   수력과 결합한 32조합은 아래 「세션 3-B」 표이며, 세션 3 게이트는 그쪽이다.",
    ]
    return "\n".join(lines)


def format_cdu_results_table(results: list[CduSteadyStateResult]) -> str:
    """수력 결합 32조합 결과 표를 문자열로 만든다 (순수 함수).

    절대 규칙 11: 산출물에 "가정값 기반 — 실측 아님" 표시를 반드시 넣는다.
    """
    header = (
        f"{'case':<44}{'Q_total':>10}{'Q_rack':>9}{'T_sup':>9}{'T_ret':>9}"
        f"{'dT':>8}{'duty':>10}{'balance':>11}{'solver':>8}"
    )
    units = (
        f"{'':<44}{'[L/s]':>10}{'[L/s]':>9}{'[C]':>9}{'[C]':>9}"
        f"{'[K]':>8}{'[kW]':>10}{'[%]':>11}{'':>8}"
    )
    lines = [
        f"세션 3-B · 8랙 CDU 정상상태 (수력 결합) · {len(results)}조합",
        "※ " + ASSUMPTION_TAG,
        "",
        header,
        units,
        "-" * len(header),
    ]
    worst_residual_percent = 0.0
    for r in results:
        residual_percent = energy_balance_residual_percent(r.thermal)
        worst_residual_percent = max(worst_residual_percent, abs(residual_percent))
        lines.append(
            f"{r.case.label:<44}"
            f"{r.flow.total_flow_Lps:>10.4f}{r.flow.mean_rack_flow_Lps:>9.4f}"
            f"{r.thermal.T_supply_C:>9.3f}{r.thermal.T_return_C:>9.3f}"
            f"{r.thermal.dT_primary_C:>8.3f}{r.thermal.hx_duty_kW:>10.2f}"
            f"{residual_percent:>11.5f}"
            f"{('OK' if r.solver_converged else 'FAIL'):>8}"
        )
    lines += [
        "-" * len(header),
        "",
        f"energy balance 최대 |잔차| = {worst_residual_percent:.5f} % "
        "(6장 기준 <0.1% — 세션 1-B 게이트를 8랙에서 재판정한 값)",
        "solver: 수력 fsolve · 열 fsolve · 결합 고정점 fsolve 셋 다 ier==1 이어야 OK",
        "물성 평가: 수력과 열이 같은 1차측 벌크 평균온도를 본다 (5-1 규약)",
        "",
        "※ " + ASSUMPTION_TAG,
        SESSION_3B_CAVEAT,
        SESSION_5B_CAVEAT,
    ]
    return "\n".join(lines)


def main() -> int:
    import sys

    if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔 기본 인코딩 대비
        sys.stdout.reconfigure(encoding="utf-8")
    print(format_results_table([solve_steady_state(c) for c in default_cases()]))
    print()
    print(
        format_cdu_results_table(
            [solve_cdu_steady_state(c) for c in default_cdu_cases()]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
