"""이동지연 감도 — N-CSTR 수치 수렴 확인 (세션 5.8).

현재 모델은 온도를 **2노드**(공급 1 · 환수 1)로 푼다. 이 파일은 각 다리를 N/2
탱크 직렬로 나눈 **N-CSTR** 을 별도로 세워, N 을 늘려도 전이 지표가 더 움직이지
않는지 본다. **격자 수렴 시험과 같은 성격**이다.

**관측 판이다 — 본 모델을 고치지 않는다.** `model.py`·`plant.py`·`dynamics.py`
는 읽기·호출만 하고, 이 모듈은 **데이터셋 생성 경로에 들어가지 않는다**.
판정 기준은 `PROCEED.md` 「세션 5.8 … 판정 기준 선기재」에 **코드보다 먼저**
적었다.

**N 은 수치처리이지 물리 가정이 아니다**(사람이 정했다 · 5-1 빈칸 처리 순서 4번).
`assumptions.py` 에 넣지 않는다 — 세션 4 의 M 배분 민감도(#20), 세션 5.6 의 배치
자유도와 같은 취급이다.

**모델**::

    [공급 다리 N/2 탱크] → 랙(Q_rack 주입) → [환수 다리 N/2 탱크] → HX(Q_hx 제거) ↩

    공급 탱크 j : m·cp·dTc_j/dt = ṁ·cp·(Tc_{j-1} − Tc_j),  Tc_0 = T_ret − Q_hx/(ṁcp)
    환수 탱크 j : m·cp·dTh_j/dt = ṁ·cp·(Th_{j-1} − Th_j),  Th_0 = T_sup + Q_rack/(ṁcp)

    T_sup = Tc_마지막 · T_ret = Th_마지막 · m = (M/2)/(N/2)

랙과 열교환기는 **보유량 0 의 경계 열원**이다 — 5-1 이 콜드플레이트·HX 보유량을
M 에서 빼 두었으므로(#21) 그것이 현재 모델의 읽기이고 이 판은 바꾸지 않는다.
다리별 질량은 5-1 「M 의 노드 배분」대로 M/2 씩이고 **다리 안에서 균등 분배**한다
(추가 파라미터 0).

**N=2 는 `dynamics._derivative` 와 항등이다** — 대입하면
`ṁcp·(T_ret − T_sup) − Q_hx` · `ṁcp·(T_sup − T_ret) + Q_rack` 이 그대로 나온다.
그래서 N=2 는 대조 기준이지 새 계산이 아니고, `tests/` 가 그 항등을 고정한다.

**정상상태는 N 에 불변이다** — dT/dt=0 이면 같은 다리 안 탱크가 전부 같은 온도가
되어 N 이 사라진다. 기준 A 가 이것을 코드로 확인한다.

**모든 수치는 가정값 기반이며 실측이 아니다.**
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from cdu_simul.assumptions import (
    ASSUMPTION_TAG,
    HEAT_EXCHANGER,
    LEAK,
    LOAD_PROFILE,
    PIPING,
    SCENARIO,
)
from cdu_simul.dynamics import (
    INTEGRATION_ATOL,
    INTEGRATION_HORIZON_IN_TAU,
    INTEGRATION_RTOL,
    HoldupBound,
    holdup_bounds,
    storage_times_s,
)
from cdu_simul.fluid import coolant_cp_Jkg_K, coolant_density_kgm3
from cdu_simul.hydraulics import (
    HydraulicCase,
    apply_leak_to_rack,
    solve_flow_distribution,
)
from cdu_simul.hydraulics import default_cases as default_hydraulic_cases
from cdu_simul.model import (
    CduCase,
    hx_capacity_terms,
    property_temperature_from_state,
    solve_cdu_steady_state,
)

_M3_PER_LITRE: float = 1.0e-3
_W_PER_KW: float = 1.0e3

#: 총 노드 수 스윕. **2 는 현재 구조이고 대조 기준이다.**
#: 물리 가정이 아니라 수치 수렴 확인용 격자다(모듈 docstring · 판정 기준).
NODE_COUNTS: tuple[int, ...] = (2, 4, 8, 16, 32, 64)


@dataclass(frozen=True)
class LagCase:
    """N-CSTR 전이 케이스 1건. 케이스마다 새로 만들어 초기조건을 리셋한다."""

    label: str
    holdup: HoldupBound
    hydraulic: HydraulicCase
    k_multiplier: float
    T_secondary_supply_C: float
    ntu: float
    load_percent: float
    n_nodes: int

    def __post_init__(self) -> None:
        if self.n_nodes < 2 or self.n_nodes % 2 != 0:
            raise ValueError(
                f"총 노드 수는 2 이상 짝수여야 한다(다리당 N/2): {self.n_nodes}"
            )

    @property
    def nodes_per_leg(self) -> int:
        return self.n_nodes // 2

    @property
    def steady_case(self) -> CduCase:
        """누출 전 결합 정상상태 케이스 (물리 정의 재사용)."""
        return CduCase(
            hydraulic=self.hydraulic,
            T_secondary_supply_C=self.T_secondary_supply_C,
            ntu=self.ntu,
            load_percent=self.load_percent,
        )

    @property
    def hydraulic_after_leak(self) -> HydraulicCase:
        return apply_leak_to_rack(self.hydraulic, self.k_multiplier)


@dataclass(frozen=True)
class LagResult:
    """N-CSTR 적분 결과 1건. solver 플래그를 함께 싣는다(절대 규칙 5)."""

    case: LagCase
    t_s: np.ndarray
    T_supply_C: np.ndarray
    T_return_C: np.ndarray
    tau_theory_s: float
    t_end_s: float
    total_flow_initial_Lps: float
    total_flow_final_Lps: float
    leak_rack_flow_initial_Lps: float
    leak_rack_flow_final_Lps: float
    pump_head_initial_mAq: float
    pump_head_final_mAq: float
    rack_outlet_initial_C: float
    rack_outlet_final_C: float
    hydraulic_solver_converged: bool
    solver_success: bool
    solver_message: str

    @property
    def T_return_initial_C(self) -> float:
        return float(self.T_return_C[0])

    @property
    def T_return_final_C(self) -> float:
        return float(self.T_return_C[-1])

    def time_to_fraction_s(self, fraction: float) -> float | None:
        """T_return 이 순변화의 `fraction` 에 도달한 시각 [s] — **선형보간**.

        `dynamics.time_to_fraction_s` 는 저장격자에 **스냅**한 시각을 낸다. 이 판은
        N 별 차이를 재는 것이라 격자 해상도가 그 차이를 가릴 수 있다 — 실제로
        N=32·64 가 스냅 때문에 **같은 값**으로 나왔다. 그래서 브래킷 구간에서
        선형보간해 격자 아래 해상도를 얻는다.

        **관측 정의를 바꾼 것이지 물리를 바꾼 것이 아니다.** 저장격자
        (`storage_times_s`)는 N 에 의존하지 않으므로 N 간 비교는 그대로 공정하다.
        """
        T0, T_end = self.T_return_initial_C, self.T_return_final_C
        total = T_end - T0
        if total == 0.0:
            return None
        progress = (self.T_return_C - T0) / total
        reached = np.nonzero(progress >= fraction)[0]
        if reached.size == 0:
            return None
        i = int(reached[0])
        if i == 0:
            return float(self.t_s[0])
        p0, p1 = float(progress[i - 1]), float(progress[i])
        t0, t1 = float(self.t_s[i - 1]), float(self.t_s[i])
        if p1 == p0:
            return t1
        return t0 + (fraction - p0) * (t1 - t0) / (p1 - p0)

    def max_abs_deviation_K(self, other: LagResult) -> float:
        """다른 N 해와의 **파형 최대 편차** [K] — 격자에 의존하지 않는 수렴 지표.

        저장격자가 N 에 의존하지 않으므로(τ·t_end 가 N 무관) 두 궤적을 시각별로
        직접 뺄 수 있다. t63 하나보다 파형 전체를 보는 쪽이 기준 C 에 맞다.
        """
        return float(np.max(np.abs(self.T_return_C - other.T_return_C)))


def integrate_leak_step_n_cstr(
    case: LagCase, horizon_in_tau: float = INTEGRATION_HORIZON_IN_TAU
) -> LagResult:
    """정격 운전 중 t=0 에 누출을 계단으로 주입하고 N-CSTR 로 적분한다.

    자극 형태는 `dynamics.integrate_leak_step` 과 같다 — 부하는 그대로이고
    K값 배율만 t=0 에 바뀐다. 초기조건은 **누출 전 결합 정상상태**이고, 정상상태는
    N 에 불변이므로 전 탱크를 그 두 온도로 채운다(기준 A 가 확인한다).

    절대 규칙 4: 압력-유량은 매 시점 quasi-steady 로 다시 푼다.
    절대 규칙 5: `solve_ivp` `success` 와 수력 `fsolve` `ier` 를 둘 다 확인한다.
    """
    before = solve_cdu_steady_state(case.steady_case)
    if not before.solver_converged:
        raise RuntimeError(f"{case.label}: 누출 전 정상상태가 수렴하지 않았다")

    per_leg = case.nodes_per_leg
    mass_supply_kg = PIPING.holdup_supply_node_fraction * case.holdup.mass_kg
    mass_return_kg = PIPING.holdup_return_node_fraction * case.holdup.mass_kg
    tank_supply_kg = mass_supply_kg / per_leg
    tank_return_kg = mass_return_kg / per_leg

    hydraulic_after = case.hydraulic_after_leak
    Q_rack_W = case.steady_case.rack_load_kW * case.hydraulic.n_racks * _W_PER_KW

    tau_theory_s = case.holdup.mass_kg / before.thermal.m_dot_kgs
    t_end_s = horizon_in_tau * tau_theory_s

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        supply, ret = y[:per_leg], y[per_leg:]
        T_supply_C, T_return_C = float(supply[-1]), float(ret[-1])

        T_property_C = property_temperature_from_state(
            T_supply_C, T_return_C, "bulk_mean"
        )
        rho_kgm3 = coolant_density_kgm3(T_property_C)
        cp_Jkg_K = coolant_cp_Jkg_K(T_property_C)

        flow = solve_flow_distribution(hydraulic_after, T_property_C)
        m_dot_kgs = flow.total_flow_Lps * _M3_PER_LITRE * rho_kgm3
        C_W_K = m_dot_kgs * cp_Jkg_K

        effectiveness, C_min_W_K = hx_capacity_terms(
            C_W_K, case.ntu, case.T_secondary_supply_C, HEAT_EXCHANGER.secondary_flow_Lps
        )
        Q_hx_W = effectiveness * C_min_W_K * (T_return_C - case.T_secondary_supply_C)

        # 경계 열원 — 보유량 0. 랙은 환수 다리 입구를, HX 는 공급 다리 입구를 만든다.
        return_inlet_C = T_supply_C + Q_rack_W / C_W_K
        supply_inlet_C = T_return_C - Q_hx_W / C_W_K

        d_supply = (
            C_W_K
            * (np.concatenate(([supply_inlet_C], supply[:-1])) - supply)
            / (tank_supply_kg * cp_Jkg_K)
        )
        d_return = (
            C_W_K
            * (np.concatenate(([return_inlet_C], ret[:-1])) - ret)
            / (tank_return_kg * cp_Jkg_K)
        )
        return np.concatenate([d_supply, d_return])

    y0 = np.concatenate(
        [
            np.full(per_leg, before.thermal.T_supply_C),
            np.full(per_leg, before.thermal.T_return_C),
        ]
    )
    solution = solve_ivp(
        rhs,
        t_span=(0.0, t_end_s),
        y0=y0,
        method="RK45",
        rtol=INTEGRATION_RTOL,
        atol=INTEGRATION_ATOL,
        dense_output=True,
        t_eval=storage_times_s(t_end_s, tau_theory_s),
    )

    T_supply_C = solution.y[per_leg - 1]
    T_return_C = solution.y[-1]
    final_flow = solve_flow_distribution(
        hydraulic_after,
        property_temperature_from_state(
            float(T_supply_C[-1]), float(T_return_C[-1]), "bulk_mean"
        ),
    )
    index = LEAK.injection_rack_index
    rho_final = coolant_density_kgm3(
        property_temperature_from_state(
            float(T_supply_C[-1]), float(T_return_C[-1]), "bulk_mean"
        )
    )
    cp_final = coolant_cp_Jkg_K(
        property_temperature_from_state(
            float(T_supply_C[-1]), float(T_return_C[-1]), "bulk_mean"
        )
    )
    rack_load_W = case.steady_case.rack_load_kW * _W_PER_KW
    return LagResult(
        case=case,
        t_s=solution.t,
        T_supply_C=T_supply_C,
        T_return_C=T_return_C,
        tau_theory_s=tau_theory_s,
        t_end_s=t_end_s,
        total_flow_initial_Lps=before.flow.total_flow_Lps,
        total_flow_final_Lps=final_flow.total_flow_Lps,
        leak_rack_flow_initial_Lps=before.flow.rack_flows_Lps[index],
        leak_rack_flow_final_Lps=final_flow.rack_flows_Lps[index],
        pump_head_initial_mAq=before.flow.pump_head_mAq,
        pump_head_final_mAq=final_flow.pump_head_mAq,
        rack_outlet_initial_C=before.thermal.rack_return_temps_C[index],
        rack_outlet_final_C=(
            float(T_supply_C[-1])
            + rack_load_W
            / (
                final_flow.rack_flows_Lps[index]
                * _M3_PER_LITRE
                * rho_final
                * cp_final
            )
        ),
        hydraulic_solver_converged=(
            before.flow.solver_converged and final_flow.solver_converged
        ),
        solver_success=bool(solution.success),
        solver_message=str(solution.message).strip(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 케이스 축 — 새 숫자 0개 (5장·5-1 양 끝만 쓴다)
# ─────────────────────────────────────────────────────────────────────────────
_LOADS: tuple[float, ...] = (
    LOAD_PROFILE.idle_load_percent,
    LOAD_PROFILE.rated_load_percent,
)
#: 5장 누출 3수준. `k_multiplier_levels` 의 첫 항은 정상(1.0)이라 뺀다 —
#: 누출 스텝에서 배율 1.0 은 아무 일도 일어나지 않는 케이스다.
_LEAK_MULTIPLIERS: tuple[float, ...] = tuple(
    multiplier for _label, multiplier in LEAK.k_multiplier_levels[1:]
)


def _build(
    hydraulics: tuple[HydraulicCase, ...],
    ntus: tuple[float, ...],
    secondaries: tuple[float, ...],
    n_nodes: int,
) -> list[LagCase]:
    return [
        LagCase(
            label=(
                f"{hydraulic.label}/NTU={ntu:g}/T2nd={T_secondary_C:g}C"
                f"/load={load:g}%/{holdup.label}/k={multiplier:g}"
            ),
            holdup=holdup,
            hydraulic=hydraulic,
            k_multiplier=multiplier,
            T_secondary_supply_C=T_secondary_C,
            ntu=ntu,
            load_percent=load,
            n_nodes=n_nodes,
        )
        for hydraulic in hydraulics
        for ntu in ntus
        for T_secondary_C in secondaries
        for load in _LOADS
        for holdup in holdup_bounds()
        for multiplier in _LEAK_MULTIPLIERS
    ]


def convergence_cases(n_nodes: int) -> list[LagCase]:
    """1단 수렴 스윕 — 수력 **양 끝 두 모서리** × 부하 2 × M 2 × 누출 3 = 24.

    NTU·2차측 공급온도를 5장 **하한**에 고정해 줄였다. 줄인 이유는 시간이다
    (§6) — 전 조합 × N 6수준은 60분 정지선을 넘는다. **줄인 축은 2단(반례
    확인)에서 전수로 되살린다.**
    """
    corners = default_hydraulic_cases()
    return _build(
        (corners[0], corners[-1]),
        (HEAT_EXCHANGER.ntu.low,),
        (SCENARIO.T_secondary_supply_C.low,),
        n_nodes,
    )


def full_cases(n_nodes: int) -> list[LagCase]:
    """2단 반례 확인 — 5장·5-1 양 끝 **전 조합 384**.

    수력 8 × NTU 2 × 2차측 2 × 부하 2 × M 2 × 누출 3.
    """
    return _build(
        tuple(default_hydraulic_cases()),
        (HEAT_EXCHANGER.ntu.low, HEAT_EXCHANGER.ntu.high),
        (
            SCENARIO.T_secondary_supply_C.low,
            SCENARIO.T_secondary_supply_C.high,
        ),
        n_nodes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 신호 — 정상상태 사이의 순변화. A 가 성립하면 이 넷은 N 에 불변이어야 한다
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LeakSignal:
    """누출 전 정상상태 → 누출 후 정상상태의 순변화 (판정 기준 D)."""

    total_flow_Lps: float
    pump_head_mAq: float
    rack_outlet_C: float
    T_return_C: float
    #: 전이 도중 순변화와 **반대 부호**로 벗어난 최대폭 [K] (오버슈트/언더슈트).
    opposite_sign_excursion_K: float


def leak_signal(result: LagResult) -> LeakSignal:
    """신호 넷과 전이 경로의 역부호 이탈폭을 낸다 (순수 함수)."""
    net_K = result.T_return_final_C - result.T_return_initial_C
    deviation = result.T_return_C - result.T_return_initial_C
    # 순변화와 반대 방향으로 간 최대폭. 0 이면 경로가 부호를 뒤집지 않았다.
    opposite = -deviation if net_K >= 0.0 else deviation
    return LeakSignal(
        total_flow_Lps=result.total_flow_final_Lps - result.total_flow_initial_Lps,
        pump_head_mAq=result.pump_head_final_mAq - result.pump_head_initial_mAq,
        rack_outlet_C=result.rack_outlet_final_C - result.rack_outlet_initial_C,
        T_return_C=net_K,
        opposite_sign_excursion_K=max(0.0, float(np.max(opposite))),
    )


_SIGN_ZERO_TOL: float = 1.0e-12


def sign_of(value: float) -> str:
    if value > _SIGN_ZERO_TOL:
        return "+"
    if value < -_SIGN_ZERO_TOL:
        return "-"
    return "0"


def sign_summary(values: list[float]) -> str:
    signs = sorted({sign_of(v) for v in values})
    return signs[0] if len(signs) == 1 else "/".join(signs)


def note() -> str:
    return (
        f"{ASSUMPTION_TAG}\n"
        "세션 5.8 · 관측 판 — 게이트 아님 · 6장 기준 판정 아님.\n"
        "N 은 수치 수렴 확인용 격자이지 물리 가정이 아니다(새 숫자 0개).\n"
        "N 수렴은 「이동지연이 실제와 맞다」가 아니다 — 실측이 없다.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1단 — 수렴 스윕 (판정 기준 A·B·C·E)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class NodeMetrics:
    """한 케이스·한 N 의 전이 지표."""

    n_nodes: int
    tau_theory_s: float
    t63_s: float | None
    t95_s: float | None
    net_signal_K: float
    #: 직전 N(절반) 대비 파형 최대 편차 [K]. N=2 는 None.
    deviation_from_coarser_K: float | None
    solver_success: bool


@dataclass(frozen=True)
class ConvergenceRun:
    """케이스 1건의 N 스윕."""

    case_label: str
    holdup_label: str
    metrics: tuple[NodeMetrics, ...]

    @property
    def t63_percent_change_vs_two(self) -> tuple[float | None, ...]:
        base = self.metrics[0].t63_s
        if base in (None, 0.0):
            return tuple(None for _ in self.metrics)
        return tuple(
            None if m.t63_s is None else (m.t63_s - base) / base * 100.0
            for m in self.metrics
        )


def run_convergence_sweep() -> list[ConvergenceRun]:
    """1단 — 24 케이스 × N 6수준 = 144 적분.

    저장격자가 N 에 의존하지 않으므로(τ·t_end 가 N 무관) 궤적을 시각별로 직접
    빼서 파형 편차를 낸다.
    """
    runs: list[ConvergenceRun] = []
    templates = convergence_cases(NODE_COUNTS[0])
    for template in templates:
        results: list[LagResult] = []
        metrics: list[NodeMetrics] = []
        for n_nodes in NODE_COUNTS:
            case = LagCase(
                label=template.label,
                holdup=template.holdup,
                hydraulic=template.hydraulic,
                k_multiplier=template.k_multiplier,
                T_secondary_supply_C=template.T_secondary_supply_C,
                ntu=template.ntu,
                load_percent=template.load_percent,
                n_nodes=n_nodes,
            )
            result = integrate_leak_step_n_cstr(case)
            results.append(result)
            signal = leak_signal(result)
            metrics.append(
                NodeMetrics(
                    n_nodes=n_nodes,
                    tau_theory_s=result.tau_theory_s,
                    t63_s=result.time_to_fraction_s(0.63),
                    t95_s=result.time_to_fraction_s(0.95),
                    net_signal_K=signal.T_return_C,
                    deviation_from_coarser_K=(
                        None
                        if not results[:-1]
                        else result.max_abs_deviation_K(results[-2])
                    ),
                    solver_success=result.solver_success,
                )
            )
        runs.append(
            ConvergenceRun(
                case_label=template.label,
                holdup_label=template.holdup.label,
                metrics=tuple(metrics),
            )
        )
    return runs


def format_convergence_table(runs: list[ConvergenceRun]) -> str:
    """기준 B·C — N 별 t63 이동과 파형 편차 (순수 함수)."""
    lines = [
        "표 1. N 별 전이 지표 — 판정 기준 B·C",
        "  t63 변화율은 N=2(현재 구조) 대비. 편차는 직전 N 대비 파형 최대 |ΔT_return|.",
        "  편차%는 그 케이스의 순신호 크기 대비다.",
        "",
        f"{'N':>4}{'t63 변화율 [%]':>20}{'파형 편차 [K]':>18}{'편차 / 순신호 [%]':>20}",
        "-" * 78,
    ]
    for index, n_nodes in enumerate(NODE_COUNTS):
        changes = [
            value
            for run in runs
            if (value := run.t63_percent_change_vs_two[index]) is not None
        ]
        deviations = [
            value
            for run in runs
            if (value := run.metrics[index].deviation_from_coarser_K) is not None
        ]
        relatives = [
            value / abs(run.metrics[index].net_signal_K) * 100.0
            for run in runs
            if (value := run.metrics[index].deviation_from_coarser_K) is not None
            and run.metrics[index].net_signal_K != 0.0
        ]
        change_text = (
            f"{min(changes):+.3f} ~ {max(changes):+.3f}" if changes else "—"
        )
        deviation_text = (
            f"{min(deviations):.3e} ~ {max(deviations):.3e}" if deviations else "—"
        )
        relative_text = (
            f"{min(relatives):.4f} ~ {max(relatives):.4f}" if relatives else "—"
        )
        lines.append(
            f"{n_nodes:>4}{change_text:>20}{deviation_text:>18}{relative_text:>20}"
        )
    lines += [
        "-" * 78,
        f"케이스 {len(runs)}건 × N {len(NODE_COUNTS)}수준 = "
        f"{len(runs) * len(NODE_COUNTS)} 적분.",
        "편차가 N 을 늘려도 더 줄지 않으면 그 지점에서 수렴한 것이다.",
    ]
    return "\n".join(lines)


def format_m_consistency_table(runs: list[ConvergenceRun]) -> str:
    """기준 E — τ ∝ M 이 N 에 흔들리는가 (순수 함수)."""
    by_key: dict[tuple[str, int], dict[str, float]] = {}
    for run in runs:
        for index, n_nodes in enumerate(NODE_COUNTS):
            t63 = run.metrics[index].t63_s
            if t63 is None:
                continue
            key = (run.case_label.replace(run.holdup_label, ""), n_nodes)
            by_key.setdefault(key, {})[run.holdup_label] = t63

    lines = [
        "표 2. τ ∝ M 정합성 — 판정 기준 E",
        "  같은 케이스에서 M 상한/하한의 t63 비. M 비는 12.832779 다.",
        "",
        f"{'N':>4}{'t63 비 (M상한/M하한)':>28}{'표본':>8}",
        "-" * 78,
    ]
    for n_nodes in NODE_COUNTS:
        ratios = [
            max(pair.values()) / min(pair.values())
            for (_label, n), pair in by_key.items()
            if n == n_nodes and len(pair) == 2 and min(pair.values()) > 0.0
        ]
        text = f"{min(ratios):.6f} ~ {max(ratios):.6f}" if ratios else "—"
        lines.append(f"{n_nodes:>4}{text:>28}{len(ratios):>8}")
    lines += ["-" * 78]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 2단 — 반례 확인 (판정 기준 C·D 를 전 조합에서)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CounterexampleRun:
    """전 조합 반례 확인 결과."""

    n_coarse: int
    n_fine: int
    #: 신호 넷의 (거친 N, 고운 N) 쌍. D 의 부호 대조가 읽는다.
    signals: tuple[tuple[LeakSignal, LeakSignal], ...]
    #: 케이스별 파형 최대 편차 [K] 와 순신호 대비 비율 [%].
    deviations_K: tuple[float, ...]
    deviation_ratios_percent: tuple[float, ...]
    t63_ratios: tuple[float, ...]
    solver_failures: tuple[str, ...]

    @property
    def n_cases(self) -> int:
        return len(self.signals)


def run_counterexample_sweep(n_fine: int) -> CounterexampleRun:
    """2단 — 전 조합 384 를 N=2 와 `n_fine` 둘로 돌린다.

    **`n_fine` 은 「수렴한 N」이 아니다** — 1단이 N≤64 안에서 수렴을 찾지 못했다.
    60분 정지선 안에서 전 조합을 볼 수 있는 가장 고운 격자로 고른 것이고,
    그 사정을 결과에 함께 적는다(§6 — 줄인 것과 이유).
    """
    signals: list[tuple[LeakSignal, LeakSignal]] = []
    deviations: list[float] = []
    ratios: list[float] = []
    t63_ratios: list[float] = []
    failures: list[str] = []

    for template in full_cases(2):
        pair: list[LagResult] = []
        for n_nodes in (2, n_fine):
            case = LagCase(
                label=template.label,
                holdup=template.holdup,
                hydraulic=template.hydraulic,
                k_multiplier=template.k_multiplier,
                T_secondary_supply_C=template.T_secondary_supply_C,
                ntu=template.ntu,
                load_percent=template.load_percent,
                n_nodes=n_nodes,
            )
            result = integrate_leak_step_n_cstr(case)
            if not result.solver_success:
                failures.append(
                    f"{template.label} / N={n_nodes}: {result.solver_message}"
                )
            pair.append(result)

        coarse, fine = pair
        coarse_signal, fine_signal = leak_signal(coarse), leak_signal(fine)
        signals.append((coarse_signal, fine_signal))

        deviation = fine.max_abs_deviation_K(coarse)
        deviations.append(deviation)
        if coarse_signal.T_return_C != 0.0:
            ratios.append(deviation / abs(coarse_signal.T_return_C) * 100.0)
        coarse_t63, fine_t63 = coarse.time_to_fraction_s(0.63), fine.time_to_fraction_s(
            0.63
        )
        if coarse_t63 and fine_t63:
            t63_ratios.append(fine_t63 / coarse_t63)

    return CounterexampleRun(
        n_coarse=2,
        n_fine=n_fine,
        signals=tuple(signals),
        deviations_K=tuple(deviations),
        deviation_ratios_percent=tuple(ratios),
        t63_ratios=tuple(t63_ratios),
        solver_failures=tuple(failures),
    )


_SIGNAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("⑴ 총유량", "total_flow_Lps"),
    ("⑵ 펌프 양정", "pump_head_mAq"),
    ("⑶ 누출랙 출구온도", "rack_outlet_C"),
    ("⑷ CDU 환수온도", "T_return_C"),
)


def format_counterexample_table(run: CounterexampleRun) -> str:
    """기준 D — 누출 신호의 부호가 N 에 뒤집히는가 (순수 함수)."""
    lines = [
        f"표 3. 누출 신호의 부호 — 판정 기준 D (N={run.n_coarse} 대 N={run.n_fine})",
        "  정상상태 사이의 순변화다. 기준 A 가 성립하면 N 에 불변이어야 한다.",
        "",
        f"{'양':<22}{f'N={run.n_coarse} 부호':>12}{f'N={run.n_fine} 부호':>12}"
        f"{'최대 |차이|':>18}",
        "-" * 78,
    ]
    for title, field in _SIGNAL_FIELDS:
        coarse = [getattr(c, field) for c, _f in run.signals]
        fine = [getattr(f, field) for _c, f in run.signals]
        gap = max(abs(a - b) for a, b in zip(coarse, fine, strict=True))
        lines.append(
            f"{title:<22}{sign_summary(coarse):>12}{sign_summary(fine):>12}{gap:>18.3e}"
        )
    excursions = [f.opposite_sign_excursion_K for _c, f in run.signals]
    lines += [
        "-" * 78,
        f"전 조합 {run.n_cases}건 × N 2수준 = {run.n_cases * 2} 적분.",
        f"전이 경로가 순변화와 **반대 부호**로 벗어난 최대폭: "
        f"{max(excursions):.3e} K (N={run.n_fine})",
        "  0 이면 경로가 부호를 뒤집지 않았다는 뜻이다.",
        "",
        f"파형 편차 (N={run.n_coarse} 대 N={run.n_fine}):",
        f"  절대  {min(run.deviations_K):.3e} ~ {max(run.deviations_K):.3e} K",
        f"  순신호 대비  {min(run.deviation_ratios_percent):.2f} ~ "
        f"{max(run.deviation_ratios_percent):.2f} %",
        f"  t63 비  {min(run.t63_ratios):.4f} ~ {max(run.t63_ratios):.4f}",
    ]
    return "\n".join(lines)
