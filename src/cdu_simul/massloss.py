"""「샘」(질량손실) 수력 모사 — 부호 확인 전용 (세션 5.6).

**절대 규칙 8 의 예외가 아니다** [세션 7.32]. 이 파일이 쓰일 때(세션 5.6)의 규칙 8
은 이상 상태를 K값 근사 하나로만 정하고 있었고 그래서 사람이 승인한 예외였다
(PROCEED.md 「사람이 정한 것 — 세션 5.5-D 마무리」⑶). **세션 7.26 이 규칙 8 을
「막힘」·「샘」 둘로 고쳐 그 예외 조건 셋이 전부 무효가 됐다** — 「샘」은 이제
규칙 8 이 정한 두 이상 상태 중 하나이고, 이 파일은 그것을 푼다. 남은 범위는 하나다:

- **이 모듈은 데이터셋 생성 경로에 들어가지 않는다** — 데이터셋 쪽 어느 모듈도 이
  파일을 import 하지 않고, 이 파일이 기존 모듈을 **읽기만** 한다. 읽는 쪽은
  같은 관측 판의 `massloss_thermal.py` 하나뿐이다.
- 목적은 하나다: 데이터셋의 「막힘」 신호(K값 증가)가 **「샘」과 같은 부호를
  가리키는가**.

**수력 한정이다. 열·온도를 보지 않는다.** 이유는 시간이 아니라 절대 규칙 1이다 —
빠져나간 유체를 보충수로 채우면 **보충수 온도**라는 5장에 없는 숫자가 필요해진다.
물성은 정격 운전점의 1차측 벌크평균(37℃)에 고정하고, 온도가 움직이지 않으므로
5-1 「수력 계산의 물성 평가 온도」 규약과 어긋나지 않는다.
**6장 feasibility 기준(energy balance · T_return 방향성 · 수렴시간 · 극단 케이스)을
하나도 판정하지 않는다.**

**모델 (5-1 계통 구조 위에 누출 노드 하나를 더한 것)**::

    펌프 ─[잔여저항 (1-g)·C]─ 헤더 ─┬ 랙0 분기+밸브 ┐
                                     ├ ...            ├─ 누출 노드 ─[잔여저항 g·C]─ 펌프
                                     └ 랙7 분기+밸브 ┘        │
                                                              └→ Q_massloss (계통 밖)

랙 i 마다 한 식::

    H_pump(Q_pump) - ΔP_res(Q_sup, (1-g)C) - ΔP_res(Q_ret, gC)
                   - ΔP_분기,i(Q_i) - ΔP_밸브,i(Q_i) = 0

    Q_sup = ΣQ_i (헤더 공급유량)   ·   Q_ret = Q_sup - Q_massloss (환수유량)

**누출 크기 Q_massloss 은 주어진 값으로 강제한다.** 누출 오리피스를 모델링하면
유출계수와 누출점 절대압(계통 기준압)이라는 5장에 없는 숫자 둘이 필요해진다.
크기를 강제하면 그 둘이 필요 없고, 부호는 크기와 무관하게 정해진다.

**보충수를 모델링하지 않는다.** 계통 재고가 줄어드는 것은 시간축 문제이고 이 판
범위 밖이다 — 여기서는 매 시점의 quasi-steady 수력 스냅숏만 본다.

**5장·5-1 이 정하지 않는 구조 자유도 둘 — 값을 고르지 않고 전수로 돌린다**
(PROCEED.md 세션 5.6 판정 기준 「구조 자유도」):

- `residual_return_share` (g): 잔여저항 중 **누출점 하류(환수측)** 몫. 5-1 이
  잔여저항을 집중저항 하나로 두고 **분해를 금지**하므로(미해결 #24) 5장에도
  5-1 에도 없다. g=0 이면 누출점 하류에 저항이 없어 질량손실이 수력적으로
  **아무 흔적도 남기지 않는다**.
- `pump_sees_supply_flow`: 줄어드는 재고가 펌프 **흡입** 쪽에서 채워지면 펌프는
  공급유량 Q_sup 을, **토출** 쪽이면 환수유량 Q_ret 를 본다.

둘 다 분수·위상의 **정의상 양 끝**이므로 새 가정치가 아니다. 세션 4 가 M 노드
배분 민감도(미해결 #20)를 시험 코드 안에서만 돌리고 `assumptions.py` 에 넣지
않은 것과 같은 취급이다 — **이 파일의 g 도 `assumptions.py` 에 넣지 않는다.**

**모든 수치는 가정값 기반이며 실측이 아니다.**
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve

from cdu_simul.assumptions import ASSUMPTION_TAG, LEAK, VALVE
from cdu_simul.hydraulics import (
    FlowDistributionResult,
    HydraulicCase,
    apply_leak_to_rack,
    branch_dp_mAq,
    default_cases,
    pump_head_mAq,
    rated_property_temperature_C,
    residual_dp_mAq,
    residual_resistance_coeff_mAq_per_Lps2,
    solve_flow_distribution,
    valve_dp_mAq,
)

# ─────────────────────────────────────────────────────────────────────────────
# 「막힘」 쪽에서 빌려 쓰는 이름 [세션 7.32 · 대응표 `docs/leak-naming-map.md`]
#
# `apply_leak_to_rack` · `LEAK.injection_rack_index` · `LEAK.k_multiplier_levels`
# 는 **「막힘」의 이름**이고 이 파일은 그것을 그대로 빌려 쓴다. 끊지 않는다 —
# ⑴ 이 파일이 하는 일 자체가 「막힘」 해와 「샘」 해를 **같은 조건에서** 나란히
# 놓는 것이라 「막힘」 경로를 같은 코드로 돌아야 하고(세션 4 C2 와 같은 취지),
# ⑵ 끊으면 동작이 바뀐다.
#
# **문서 쪽 주입 지점은 이미 기구별로 갈렸다**(세션 7.27) — 5-1 「「막힘」 주입
# 지점」과 「「샘」 주입 지점」이다. 랙 번호가 같을 뿐 성립 이유가 다르다: 「막힘」은
# 8랙 대칭이라 랙 번호가 결과에 영향하지 않고, 「샘」은 **애초에 랙에 국소화되지
# 않기** 때문이다(랙 간 비대칭 2.220e-16 L/s · 세션 5.6).
# ─────────────────────────────────────────────────────────────────────────────

#: 부호를 "0" 으로 읽는 절대 임계. 세션 3-A/3-B 가 관측한 압력평형 잔차(~1e-15
#: mAq)·유량 균등성 잔차(상대 1e-9)보다 크고, 이 판이 내는 신호(1e-5 이상)보다
#: 네 자리 이상 작다. 통과시키려고 고른 값이 아니다.
_SIGN_ZERO_TOL: float = 1.0e-12


@dataclass(frozen=True)
class MassLossTopology:
    """「샘」(질량손실)의 계통 배치 — 5장·5-1 이 정하지 않는 구조 자유도 둘."""

    label: str
    residual_return_share: float
    pump_sees_supply_flow: bool


def massloss_topologies() -> tuple[MassLossTopology, ...]:
    """구조 자유도 전수 — g 양 끝과 중점 × 펌프 유량 위치 양쪽.

    g 의 중점 0.5 를 넣는 것은 **값을 고르는 것이 아니다** — 양 끝 사이에서 부호가
    갈리는 지점이 있는지 보려는 것이다(기준 B 반례 확인).
    """
    return tuple(
        MassLossTopology(
            label=f"g={share:.1f}/펌프={'공급' if sees_supply else '환수'}",
            residual_return_share=share,
            pump_sees_supply_flow=sees_supply,
        )
        for share in (0.0, 0.5, 1.0)
        for sees_supply in (True, False)
    )


@dataclass(frozen=True)
class MassLossResult:
    """질량손실 해. solver 성공 여부를 반드시 싣는다(절대 규칙 5)."""

    case: HydraulicCase
    topology: MassLossTopology
    massloss_flow_Lps: float
    rack_flows_Lps: tuple[float, ...]
    pump_head_mAq: float
    max_abs_equation_residual_mAq: float
    solver_ier: int
    solver_message: str
    solver_converged: bool

    @property
    def supply_flow_Lps(self) -> float:
        """헤더 공급유량 ΣQ_i — 랙(콜드플레이트)이 실제로 받는 총유량."""
        return float(sum(self.rack_flows_Lps))

    @property
    def pump_flow_Lps(self) -> float:
        """펌프를 통과하는 유량. 배치에 따라 Q_sup 또는 Q_ret 다."""
        if self.topology.pump_sees_supply_flow:
            return self.supply_flow_Lps
        return self.supply_flow_Lps - self.massloss_flow_Lps


def solve_massloss(
    case: HydraulicCase,
    massloss_flow_Lps: float,
    topology: MassLossTopology,
    T_property_C: float,
) -> MassLossResult:
    """누출 유량을 강제한 상태로 헤더 압력평형을 `fsolve` 로 푼다.

    `massloss_flow_Lps = 0` 이면 식이 K 근사 정상 케이스와 **항등적으로 같다**
    (g·C·Q² + (1-g)·C·Q² = C·Q²) — 격리 확인(기준 C)이 이것을 쓴다.

    절대 규칙 5: `ier` 를 확인해 결과에 싣고, 실패하면 조용히 넘어가지 않는다.
    """
    K_per_rack = case.rack_branch_K
    residual_coeff = residual_resistance_coeff_mAq_per_Lps2(case)
    g = topology.residual_return_share

    def equations(Q_racks_Lps: np.ndarray) -> np.ndarray:
        Q_supply_Lps = float(np.sum(Q_racks_Lps))
        Q_return_Lps = Q_supply_Lps - massloss_flow_Lps
        Q_pump_Lps = (
            Q_supply_Lps if topology.pump_sees_supply_flow else Q_return_Lps
        )
        available_mAq = (
            pump_head_mAq(Q_pump_Lps, case.pump)
            - residual_dp_mAq(Q_supply_Lps, (1.0 - g) * residual_coeff, T_property_C)
            - residual_dp_mAq(Q_return_Lps, g * residual_coeff, T_property_C)
        )
        return np.array(
            [
                available_mAq
                - branch_dp_mAq(float(Q_i), K_i, T_property_C)
                - valve_dp_mAq(
                    float(Q_i),
                    case.valve_Kv_max_m3h,
                    case.opening_fraction,
                    T_property_C,
                )
                for Q_i, K_i in zip(Q_racks_Lps, K_per_rack, strict=True)
            ]
        )

    # 초기값은 K 근사 경로와 같다 — 두 경로의 차이가 누출 때문이지 출발점 때문이
    # 아니게 하려는 것이다(세션 4 C2 와 같은 취지).
    guess = np.full(case.n_racks, VALVE.rated_flow_per_rack_Lps)
    solution, _info, ier, message = fsolve(equations, guess, full_output=True)
    converged = ier == 1
    if not converged:
        raise RuntimeError(
            f"fsolve 가 수렴하지 않았다 (ier={ier}): {case.label} / "
            f"{topology.label} / Q_massloss={massloss_flow_Lps:g} L/s — "
            f"{str(message).strip()}"
        )

    rack_flows = tuple(float(q) for q in solution)
    Q_supply = float(sum(rack_flows))
    Q_pump = (
        Q_supply if topology.pump_sees_supply_flow else Q_supply - massloss_flow_Lps
    )
    return MassLossResult(
        case=case,
        topology=topology,
        massloss_flow_Lps=massloss_flow_Lps,
        rack_flows_Lps=rack_flows,
        pump_head_mAq=pump_head_mAq(Q_pump, case.pump),
        max_abs_equation_residual_mAq=float(np.max(np.abs(equations(solution)))),
        solver_ier=int(ier),
        solver_message=str(message).strip(),
        solver_converged=converged,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 누출 크기 — 새 값을 만들지 않는다. K 근사가 만든 감소량을 상한으로 쓴다
# ─────────────────────────────────────────────────────────────────────────────
def k_approx_results(
    case: HydraulicCase, T_property_C: float
) -> list[FlowDistributionResult]:
    """K 근사: 정상 + 5장 누출 3수준. 정상도 같은 경로를 탄다(세션 4 C2)."""
    return [
        solve_flow_distribution(apply_leak_to_rack(case, multiplier), T_property_C)
        for _label, multiplier in LEAK.k_multiplier_levels
    ]


def massloss_flow_bound_Lps(k_results: list[FlowDistributionResult]) -> float:
    """스윕 상한 [L/s] — K 근사 +50% 가 만든 **누출랙 통과유량 감소량**.

    [유도: 5-1 「「샘」(질량손실) 크기 수준」 — 「샘」에는 독립한 수준값이 없고
    「막힘」 3수준(5장 「누출 시나리오(「막힘」)」)의 해에서 역산한다. 읽는 랙은
    「막힘」 해의 랙이므로 랙 번호는 5-1 「「막힘」 주입 지점」의 것이다.
    새 숫자를 만들지 않는다]
    """
    i = LEAK.injection_rack_index
    return k_results[0].rack_flows_Lps[i] - k_results[-1].rack_flows_Lps[i]


#: 스윕 지점 — 0(정상)부터 상한까지. 지점 개수는 부호 판정에 영향하지 않는다.
SWEEP_FRACTIONS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 부호 대조
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SignalDeltas:
    """정상(누출 0) 대비 네 양의 변화. 두 방식이 같은 형태로 담긴다."""

    supply_flow_change_Lps: float
    pump_flow_change_Lps: float
    pump_head_change_mAq: float
    injection_rack_flow_change_Lps: float
    other_rack_flow_change_Lps: float


def _other_rack_index(rack_index: int) -> int:
    return 1 if rack_index == 0 else 0


def k_approx_deltas(
    k_results: list[FlowDistributionResult], level_index: int, rack_index: int
) -> SignalDeltas:
    """K 근사의 정상 대비 변화. 펌프 유량 = 총유량이다(계통 밖으로 나가지 않는다)."""
    normal, perturbed = k_results[0], k_results[level_index]
    other = _other_rack_index(rack_index)
    total_change = perturbed.total_flow_Lps - normal.total_flow_Lps
    return SignalDeltas(
        supply_flow_change_Lps=total_change,
        pump_flow_change_Lps=total_change,
        pump_head_change_mAq=perturbed.pump_head_mAq - normal.pump_head_mAq,
        injection_rack_flow_change_Lps=(
            perturbed.rack_flows_Lps[rack_index] - normal.rack_flows_Lps[rack_index]
        ),
        other_rack_flow_change_Lps=(
            perturbed.rack_flows_Lps[other] - normal.rack_flows_Lps[other]
        ),
    )


def massloss_deltas(
    normal: MassLossResult, perturbed: MassLossResult, rack_index: int
) -> SignalDeltas:
    """질량손실의 정상 대비 변화. 기준은 **같은 배치의 Q_massloss=0 해**다."""
    other = _other_rack_index(rack_index)
    return SignalDeltas(
        supply_flow_change_Lps=perturbed.supply_flow_Lps - normal.supply_flow_Lps,
        pump_flow_change_Lps=perturbed.pump_flow_Lps - normal.pump_flow_Lps,
        pump_head_change_mAq=perturbed.pump_head_mAq - normal.pump_head_mAq,
        injection_rack_flow_change_Lps=(
            perturbed.rack_flows_Lps[rack_index] - normal.rack_flows_Lps[rack_index]
        ),
        other_rack_flow_change_Lps=(
            perturbed.rack_flows_Lps[other] - normal.rack_flows_Lps[other]
        ),
    )


_QUANTITIES: tuple[tuple[str, str], ...] = (
    ("⑴ 랙 통과 총유량", "supply_flow_change_Lps"),
    ("⑴' 펌프 통과유량", "pump_flow_change_Lps"),
    ("⑵ 펌프 양정", "pump_head_change_mAq"),
    ("⑶ 누출랙 통과유량", "injection_rack_flow_change_Lps"),
    ("⑷ 타 7랙 유량", "other_rack_flow_change_Lps"),
)


@dataclass(frozen=True)
class MismatchRow:
    """공급·환수 유량 불일치 한 케이스 — **세션 5.6-B 출력용**.

    **계산이 아니라 재출력이다.** 세션 5.6 이 이미 푼 해에서 값을 읽기만 하며,
    `solve_massloss` 의 식·solver 설정·스윕 범위·배치 정의를 하나도 바꾸지 않는다.

    `return_flow_Lps` 는 `solve_massloss` 안의 `Q_return_Lps = Q_supply_Lps -
    massloss_flow_Lps` 를 **그대로 되읽은 것**이다 — 따라서 Δ = Q_massloss 은 물리 검산이
    아니라 **코드가 정의대로 계산했는지 보는 항등식**이다(세션 5.6-B 판정 기준
    「먼저 적어 두는 것」).
    """

    path_label: str
    case_label: str
    size_index: int
    supply_flow_Lps: float
    return_flow_Lps: float
    massloss_flow_Lps: float

    @property
    def mismatch_Lps(self) -> float:
        """Δ = 공급유량 − 환수유량."""
        return self.supply_flow_Lps - self.return_flow_Lps

    @property
    def mismatch_percent(self) -> float:
        """Δ 의 공급유량 대비 상대값 [%]."""
        return self.mismatch_Lps / self.supply_flow_Lps * 100.0

    @property
    def identity_residual_Lps(self) -> float:
        """기준 A — Δ 와 누출 유량의 차. 항등식의 부동소수점 잔차다."""
        return self.mismatch_Lps - self.massloss_flow_Lps


def sign_summary(values: list[float]) -> str:
    """부호 요약 — 전 조합에서 일정하면 `+`/`-`/`0`, 갈리면 `갈림`."""
    has_pos = any(v > _SIGN_ZERO_TOL for v in values)
    has_neg = any(v < -_SIGN_ZERO_TOL for v in values)
    if has_pos and has_neg:
        return "갈림"
    if has_pos:
        return "+"
    if has_neg:
        return "-"
    return "0"


def span(values: list[float]) -> str:
    return f"{min(values):+.3e} ~ {max(values):+.3e}"


def _mismatch_row(result: MassLossResult, size_index: int) -> MismatchRow:
    """푼 해에서 불일치 행을 읽어낸다 (순수 함수 · 세션 5.6-B).

    환수유량은 `solve_massloss` 의 정의 `Q_ret = Q_sup - Q_massloss` 를 그대로 쓴다.
    """
    return MismatchRow(
        path_label=result.topology.label,
        case_label=result.case.label,
        size_index=size_index,
        supply_flow_Lps=result.supply_flow_Lps,
        return_flow_Lps=result.supply_flow_Lps - result.massloss_flow_Lps,
        massloss_flow_Lps=result.massloss_flow_Lps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 실행 · 결과표
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ComparisonRun:
    """8조합 × (K 근사 3수준 · 질량손실 배치 6 × 크기 4) 전수 결과."""

    T_property_C: float
    rack_index: int
    k_deltas: dict[str, list[SignalDeltas]]
    massloss_deltas: dict[str, list[SignalDeltas]]
    #: 세션 5.6-B 재출력용. 계산에 관여하지 않는다 — 푼 해를 읽어 담을 뿐이다.
    mismatch_rows: list[MismatchRow]
    massloss_bounds_Lps: list[float]
    isolation_max_abs_diff_Lps: float
    massloss_max_rack_asymmetry_Lps: float
    max_abs_equation_residual_mAq: float
    n_solves: int
    n_solver_failures: int


def run_comparison(rack_index: int = LEAK.injection_rack_index) -> ComparisonRun:
    """전수 실행. 실패는 예외로 드러난다(절대 규칙 5) — 조용히 넘기지 않는다."""
    T_C = rated_property_temperature_C()
    cases = default_cases()
    levels = LEAK.k_multiplier_levels
    topologies = massloss_topologies()

    k_deltas: dict[str, list[SignalDeltas]] = {label: [] for label, _ in levels[1:]}
    ml_deltas: dict[str, list[SignalDeltas]] = {t.label: [] for t in topologies}
    mismatch_rows: list[MismatchRow] = []
    bounds: list[float] = []
    isolation_diff = 0.0
    asymmetry = 0.0
    max_residual = 0.0
    n_solves = 0

    for case in cases:
        k_results = k_approx_results(case, T_C)
        n_solves += len(k_results)
        max_residual = max(
            max_residual, *(r.max_abs_equation_residual_mAq for r in k_results)
        )
        for index, (label, _multiplier) in enumerate(levels[1:], start=1):
            k_deltas[label].append(k_approx_deltas(k_results, index, rack_index))
        # K 근사에는 계통 밖으로 나가는 경로가 없다 — 환수유량 = 총유량이다.
        mismatch_rows += [
            MismatchRow(
                path_label="K 근사",
                case_label=case.label,
                size_index=index,
                supply_flow_Lps=result.total_flow_Lps,
                return_flow_Lps=result.total_flow_Lps,
                massloss_flow_Lps=0.0,
            )
            for index, result in enumerate(k_results)
        ]

        bound = massloss_flow_bound_Lps(k_results)
        bounds.append(bound)

        for topology in topologies:
            normal = solve_massloss(case, 0.0, topology, T_C)
            n_solves += 1
            max_residual = max(max_residual, normal.max_abs_equation_residual_mAq)
            # 기준 C 격리 확인 — Q_massloss=0 해가 K 근사 정상 해와 같아야 한다.
            isolation_diff = max(
                isolation_diff,
                max(
                    abs(a - b)
                    for a, b in zip(
                        normal.rack_flows_Lps,
                        k_results[0].rack_flows_Lps,
                        strict=True,
                    )
                ),
            )
            mismatch_rows.append(_mismatch_row(normal, 0))
            for size_index, fraction in enumerate(SWEEP_FRACTIONS[1:], start=1):
                perturbed = solve_massloss(case, fraction * bound, topology, T_C)
                n_solves += 1
                max_residual = max(
                    max_residual, perturbed.max_abs_equation_residual_mAq
                )
                # 5-1 이 헤더를 저항 0 의 공통 노드로 두므로 랙 출구 누출은
                # 랙에 국소화될 수 없다 — 그 예상을 값으로 확인한다.
                asymmetry = max(
                    asymmetry,
                    max(perturbed.rack_flows_Lps) - min(perturbed.rack_flows_Lps),
                )
                ml_deltas[topology.label].append(
                    massloss_deltas(normal, perturbed, rack_index)
                )
                mismatch_rows.append(_mismatch_row(perturbed, size_index))

    return ComparisonRun(
        T_property_C=T_C,
        rack_index=rack_index,
        k_deltas=k_deltas,
        massloss_deltas=ml_deltas,
        mismatch_rows=mismatch_rows,
        massloss_bounds_Lps=bounds,
        isolation_max_abs_diff_Lps=isolation_diff,
        massloss_max_rack_asymmetry_Lps=asymmetry,
        max_abs_equation_residual_mAq=max_residual,
        n_solves=n_solves,
        # 수렴 실패는 `solve_massloss`·`solve_flow_distribution` 이 예외로 던지므로
        # 여기까지 오면 0이다. 세는 자리를 남겨 두어 결과표가 그 사실을 적게 한다.
        n_solver_failures=0,
    )


def format_comparison_table(run: ComparisonRun) -> str:
    """부호 대조 표 (순수 함수).

    절대 규칙 11: 산출물에 "가정값 기반 — 실측 아님" 표시를 반드시 넣는다.
    """
    lines = [
        "세션 5.6 · 「막힘」(K값 증가) 대 「샘」(질량손실) — 부호 대조 (수력 한정)",
        "※ " + ASSUMPTION_TAG,
        "※ **절대 규칙 8 이 정한 두 이상 상태다 — 예외가 아니다**(세션 7.32).",
        "   「샘」 모사는 이 파일에 있고, 데이터셋 생성 경로는 「막힘」뿐이다.",
        "※ 6장 feasibility 기준(energy balance · T_return 방향성 · 수렴시간 ·",
        "   극단 케이스)을 **하나도 판정하지 않는다** — 열모델이 이 판에 없다.",
        f"※ 물성 고정 온도 {run.T_property_C:.1f} ℃ (5장 1차측 벌크평균 · 5-1 규약)"
        " — 이 판은 온도를 풀지 않는다.",
        "",
        "[표 1] K 근사 — 정상 대비 변화 (8조합 범위)",
    ]
    header = (
        f"{'양':<20}"
        + "".join(f"{label:>26}" for label, _ in LEAK.k_multiplier_levels[1:])
        + f"{'부호':>8}"
    )
    lines += [header, "-" * len(header)]
    for name, field in _QUANTITIES:
        row = f"{name:<20}"
        all_values: list[float] = []
        for label, _ in LEAK.k_multiplier_levels[1:]:
            values = [getattr(d, field) for d in run.k_deltas[label]]
            all_values += values
            row += f"{span(values):>26}"
        lines.append(row + f"{sign_summary(all_values):>8}")

    lines += [
        "-" * len(header),
        "  단위: 유량 [L/s] · 양정 [mAq]. ⑴ 과 ⑴' 은 K 근사에서 같은 값이다",
        "  (계통 밖으로 나가는 유량이 없다).",
        "",
        "[표 2] 질량손실 — 정상(Q_massloss=0) 대비 변화 (8조합 × 크기 4수준 범위)",
        "  누출 크기는 조합마다 0 부터 **K 근사 +50% 가 만든 누출랙 유량 감소량**"
        " 까지 스윕했다.",
        f"  상한 범위: {min(run.massloss_bounds_Lps):.6f} ~ "
        f"{max(run.massloss_bounds_Lps):.6f} L/s (새 숫자가 아니다 — K 근사의 산출값).",
        "",
    ]
    header2 = f"{'배치':<24}" + "".join(f"{name:>22}" for name, _ in _QUANTITIES)
    lines += [header2, "-" * len(header2)]
    for topology in massloss_topologies():
        deltas = run.massloss_deltas[topology.label]
        row = f"{topology.label:<24}"
        for _name, field in _QUANTITIES:
            row += f"{sign_summary([getattr(d, field) for d in deltas]):>22}"
        lines.append(row)
    lines += ["-" * len(header2), ""]

    lines.append("[표 3] 질량손실 — 배치별 변화 범위 (배치 전수)")
    header3 = f"{'배치':<24}" + "".join(f"{name:>26}" for name, _ in _QUANTITIES)
    lines += [header3, "-" * len(header3)]
    for topology in massloss_topologies():
        deltas = run.massloss_deltas[topology.label]
        row = f"{topology.label:<24}"
        for _name, field in _QUANTITIES:
            row += f"{span([getattr(d, field) for d in deltas]):>26}"
        lines.append(row)

    lines += [
        "-" * len(header3),
        "",
        f"격리 확인(기준 C): Q_massloss=0 의 질량손실 해와 K 근사 정상 해의 랙 유량"
        f" 최대 차 = {run.isolation_max_abs_diff_Lps:.3e} L/s",
        f"랙 간 비대칭(질량손실 전 조합): 최대 "
        f"{run.massloss_max_rack_asymmetry_Lps:.3e} L/s — 5-1 이 헤더를 저항 0 의",
        "  공통 노드로 두므로 랙 출구 누출은 **랙에 국소화되지 않는다**. ⑶ 과 ⑷ 가",
        "  같이 움직이는 것이 그 결과다(K 근사는 정반대로 갈라진다).",
        f"solver(기준 D): {run.n_solves} 회 전부 `ier=1` · 실패 "
        f"{run.n_solver_failures} 건 · 압력평형 최대 잔차 "
        f"{run.max_abs_equation_residual_mAq:.3e} mAq",
        "",
        "읽는 법:",
        "  · `갈림` = 조합·크기·수준에 따라 부호가 뒤집힌다(기준 B 반례 확인).",
        f"  · `0` = 변화가 {_SIGN_ZERO_TOL:.0e} 미만이다.",
        "  · 배치 g 는 잔여저항 중 **누출점 하류(환수측)** 몫이며 5장·5-1 에 없다.",
        "    값을 고르지 않고 정의상 양 끝을 포함해 전수로 돌린 것이다.",
        "",
        "※ " + ASSUMPTION_TAG,
        "※ 이 표가 재지 **않은** 것: 어느 모사가 실제 누출에 가까운가(실측 없음) ·",
        "   신호가 계측 가능한 크기인가(계측기 사양 없음) · 열·온도 거동 전부 ·",
        "   계통 재고가 줄어드는 것(보충수를 모델링하지 않았다 — 시간축 문제다).",
    ]
    return "\n".join(lines)


def format_mismatch_table(run: ComparisonRun) -> str:
    """[표 4] 공급·환수 유량 불일치 (세션 5.6-B · 순수 함수).

    **새로 푼 것이 없다** — 세션 5.6 의 272 회 해를 다른 양으로 읽었을 뿐이다.
    """
    rows = run.mismatch_rows
    k_rows = [r for r in rows if r.path_label == "K 근사"]
    lines = [
        "[표 4] 공급·환수 유량 불일치 Δ = Q_sup − Q_ret (세션 5.6-B · 재출력)",
        "  Δ 는 열교환기 1차측 입구가 펌프 공급보다 얼마나 모자라는가다.",
        "  K 근사에는 계통 밖으로 나가는 경로가 없어 이 양이 원리적으로 0 이다.",
        "",
    ]
    header = (
        f"{'경로 · 배치':<24}{'Δ [L/s]':>26}{'Δ/Q_sup [%]':>26}"
        f"{'항등식 잔차 [L/s]':>22}{'단조':>8}"
    )
    lines += [header, "-" * len(header)]

    def monotone(path_label: str) -> str:
        """같은 조합 안에서 누출 크기 순으로 Δ 가 엄격히 증가하는가.

        K 근사에는 누출 크기 축 자체가 없고(축은 K 배율이다) Δ 가 항상 0 이므로
        기준 B 의 대상이 아니다 — `해당없음` 으로 적는다.
        """
        if path_label == "K 근사":
            return "해당없음"
        by_case: dict[str, list[tuple[int, float]]] = {}
        for r in rows:
            if r.path_label == path_label:
                by_case.setdefault(r.case_label, []).append(
                    (r.size_index, r.mismatch_Lps)
                )
        for series in by_case.values():
            values = [v for _i, v in sorted(series)]
            if any(b <= a for a, b in zip(values, values[1:], strict=False)):
                return "아니오"
        return "예"

    for path_label in ["K 근사"] + [t.label for t in massloss_topologies()]:
        group = k_rows if path_label == "K 근사" else [
            r for r in rows if r.path_label == path_label
        ]
        lines.append(
            f"{path_label:<24}"
            f"{span([r.mismatch_Lps for r in group]):>26}"
            f"{span([r.mismatch_percent for r in group]):>26}"
            f"{max(abs(r.identity_residual_Lps) for r in group):>22.3e}"
            f"{monotone(path_label):>8}"
        )
    lines += ["-" * len(header), ""]

    ml_identity_max = max(
        abs(r.identity_residual_Lps) for r in rows if r.path_label != "K 근사"
    )
    degenerate = [r for r in rows if r.path_label == "g=0.0/펌프=공급"]
    nonzero = [r for r in degenerate if r.size_index > 0]
    lines += [
        "기준 B — 단조: 질량손실 전 배치에서 Δ 가 누출 크기 4수준에 대해 엄격히",
        "  증가한다. K 근사는 누출 크기 축이 없고 Δ ≡ 0 이라 대상이 아니다.",
        "",
        "기준 A — 검산:",
        f"  · 질량손실 Δ 와 누출 유량의 차 최대 {ml_identity_max:.3e} L/s",
        f"  · K 근사 Δ 최대 {max(abs(r.mismatch_Lps) for r in k_rows):.3e} L/s",
        "  · **이 둘은 구성상 항등식이다** — `solve_massloss` 가 환수유량을",
        "    Q_ret = Q_sup − Q_massloss 으로 정의하고, K 근사에는 유출 경로가 없다.",
        "    통과의 뜻은 「코드가 정의대로 계산한다」까지이고 **질량보존의 물리적",
        "    검증이 아니다**(세션 5.6-B 판정 기준 「먼저 적어 두는 것」).",
        "",
        "기준 C — 배치 불변: 같은 (조합 · 누출 크기)에서 배치 6 에 걸친 Δ 의 최대 편차 "
        f"{_mismatch_spread_across_topologies(rows):.3e} L/s",
        f"  · `g=0.0/펌프=공급` 배치의 Δ (누출 크기 4수준): "
        f"{span([r.mismatch_Lps for r in nonzero])} L/s",
        "    — 세션 5.6 관측 ④ 에서 다섯 양이 전부 정확히 0 이었던 배치다.",
        "",
        "※ " + ASSUMPTION_TAG,
        "※ Δ 가 크게 나와도 **계측 가능하다는 뜻이 아니다**(계측기 사양 없음) ·",
        "   **실제 누출에 가깝다는 뜻도 아니다**(실측 없음).",
        "※ 세션 5.6 이 낸 네 양과 크기를 비교하지 않았다 — 크기 비교는 계측기",
        "   사양이 있어야 뜻이 생긴다.",
    ]
    return "\n".join(lines)


def _mismatch_spread_across_topologies(rows: list[MismatchRow]) -> float:
    """같은 (조합 · 누출 크기)에서 배치별 Δ 가 얼마나 벌어지는가 [L/s]."""
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        if row.path_label == "K 근사":
            continue
        grouped.setdefault((row.case_label, row.size_index), []).append(
            row.mismatch_Lps
        )
    return max(max(v) - min(v) for v in grouped.values())


def main() -> int:
    import sys

    if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔 기본 인코딩 대비
        sys.stdout.reconfigure(encoding="utf-8")
    run = run_comparison()
    print(format_comparison_table(run))
    print()
    print(format_mismatch_table(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
