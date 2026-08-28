"""누출 시나리오 실행 (세션 4) — 정상 대비 신호 패턴.

범위 — 4장 「단계적 확장 전략」 3단계다.

    정상 · 누출 +5/+20/+50% → 랙 유량 · 랙 출구온도 · 펌프 운전점의 변화

**누출은 배관저항(K값) 변화로만 근사한다**(절대 규칙 8 · 5장 누출 시나리오).
질량 손실을 직접 모델링하지 않는다 — 모델에 그런 경로가 없다. 따라서 여기서
나오는 것은 *누출의 물리*가 아니라 **5장이 정의한 누출 대용(proxy)의 거동**이다.

**정상(배율 1.0)이 같은 코드 경로로 돈다.** `LEAK.k_multiplier_levels` 가 정상을
첫 수준으로 담고 있고, 정상도 `apply_leak_to_rack` 를 통과한다 — 정상과 누출이
다른 경로를 타면 결과 차이가 누출 때문인지 경로 때문인지 갈라낼 수 없다.

**주입 지점은 랙 1개**다 [시나리오 정의: 5-1 「누출 주입 지점」 · 세션 4].
8랙이 전부 동일하므로 랙 번호는 결과에 영향하지 않는다. 전 랙 동시 누출은 5장에
근거가 없어 돌리지 않는다.

**펌프는 정속이다** [규약: 5-1 「펌프 운전 방식」 · 세션 4]. RPM 을 모델링하지
않는다 — 5장이 특성곡선 하나만 주고 제어 로직·목표값을 말하지 않기 때문이다.
신호는 사라지지 않는다: K 증가 → 총유량 감소 → 운전점이 곡선을 따라 이동해
**양정이 오른다.**

**모든 수치는 가정값 기반이며 실측이 아니다.**
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from cdu_simul.assumptions import ASSUMPTION_TAG, LEAK, SESSION_4_CAVEAT
from cdu_simul.hydraulics import apply_leak_to_rack
from cdu_simul.model import (
    CduCase,
    CduSteadyStateResult,
    default_cdu_cases,
    energy_balance_residual_percent,
    solve_cdu_steady_state,
)


@dataclass(frozen=True)
class LeakLevel:
    """누출 수준 하나. 정상(배율 1.0)도 이 형태로 담긴다."""

    label: str
    k_multiplier: float

    @property
    def is_normal(self) -> bool:
        return self.k_multiplier == 1.0


def leak_levels() -> tuple[LeakLevel, ...]:
    """정상 + 5장 누출 3수준. `assumptions.py` 에서만 읽는다(절대 규칙 2)."""
    return tuple(
        LeakLevel(label=label, k_multiplier=multiplier)
        for label, multiplier in LEAK.k_multiplier_levels
    )


def leak_case(
    case: CduCase,
    level: LeakLevel,
    rack_index: int = LEAK.injection_rack_index,
) -> CduCase:
    """CduCase 에 누출 수준을 적용한다 — 수력 케이스만 바뀌고 열 조건은 그대로다."""
    return replace(
        case,
        hydraulic=apply_leak_to_rack(case.hydraulic, level.k_multiplier, rack_index),
    )


@dataclass(frozen=True)
class LeakSteadySignal:
    """정상 대비 누출 1수준의 정상상태 신호 (조합 1개분).

    모든 변화량은 **같은 조합의 정상 해를 기준**으로 한 차이다 — 조합이 다르면
    절대값이 다르므로 조합 안에서만 비교한다.
    """

    case_label: str
    level: LeakLevel
    normal: CduSteadyStateResult
    leaked: CduSteadyStateResult
    rack_index: int

    @property
    def leak_rack_flow_change_percent(self) -> float:
        """누출 랙 유량 변화 [%] — 기대 부호: 음(감소)."""
        before = self.normal.flow.rack_flows_Lps[self.rack_index]
        after = self.leaked.flow.rack_flows_Lps[self.rack_index]
        return (after - before) / before * 100.0

    @property
    def other_rack_flow_change_percent(self) -> float:
        """비누출 랙 유량 변화 [%] — 기대 부호: 양(증가).

        나머지 7랙은 전부 같은 조건이므로 첫 비누출 랙 하나로 대표한다.
        """
        other = 1 if self.rack_index == 0 else 0
        before = self.normal.flow.rack_flows_Lps[other]
        after = self.leaked.flow.rack_flows_Lps[other]
        return (after - before) / before * 100.0

    @property
    def total_flow_change_percent(self) -> float:
        """총유량 변화 [%] — 기대 부호: 음(감소)."""
        before = self.normal.flow.total_flow_Lps
        after = self.leaked.flow.total_flow_Lps
        return (after - before) / before * 100.0

    @property
    def leak_rack_outlet_change_C(self) -> float:
        """누출 랙 출구온도 변화 [K] — 기대 부호: 양(상승)."""
        before = self.normal.thermal.rack_return_temps_C[self.rack_index]
        after = self.leaked.thermal.rack_return_temps_C[self.rack_index]
        return after - before

    @property
    def header_return_change_C(self) -> float:
        """CDU 합류 T_return 변화 [K]."""
        return self.leaked.thermal.T_return_C - self.normal.thermal.T_return_C

    @property
    def pump_head_change_mAq(self) -> float:
        """펌프 양정 변화 [mAq] — 기대 부호: 양(정속 곡선을 따라 좌상 이동)."""
        return self.leaked.flow.pump_head_mAq - self.normal.flow.pump_head_mAq

    @property
    def energy_balance_residual_percent(self) -> float:
        """누출 상태의 energy balance 잔차 [%] — 1-B 게이트가 누출에서도 서는가."""
        return energy_balance_residual_percent(self.leaked.thermal)

    @property
    def solvers_converged(self) -> bool:
        return self.normal.solver_converged and self.leaked.solver_converged


def steady_signals(
    case: CduCase, rack_index: int = LEAK.injection_rack_index
) -> list[LeakSteadySignal]:
    """조합 하나에 대해 누출 3수준의 정상상태 신호를 낸다.

    정상 해도 `leak_case(..., 배율 1.0)` 를 통과시켜 얻는다 — 같은 코드 경로다.
    """
    levels = leak_levels()
    normal_level = levels[0]
    if not normal_level.is_normal:
        raise ValueError("첫 수준이 정상(배율 1.0)이 아니다")
    normal = solve_cdu_steady_state(leak_case(case, normal_level, rack_index))
    return [
        LeakSteadySignal(
            case_label=case.label,
            level=level,
            normal=normal,
            leaked=solve_cdu_steady_state(leak_case(case, level, rack_index)),
            rack_index=rack_index,
        )
        for level in levels[1:]
    ]


def all_steady_signals(
    rack_index: int = LEAK.injection_rack_index,
) -> list[LeakSteadySignal]:
    """5장·5-1 범위 양 끝 32조합 × 누출 3수준."""
    return [
        signal
        for case in default_cdu_cases()
        for signal in steady_signals(case, rack_index)
    ]


def format_steady_signal_table(signals: list[LeakSteadySignal]) -> str:
    """누출 3수준 신호를 조합 전체의 **범위(최소~최대)** 로 요약한다 (순수 함수).

    32조합 × 3수준을 행마다 다 적으면 96행이 되어 읽히지 않는다. 조합 간 차이는
    5장 범위 양 끝에서 오는 것이고, 게이트가 보는 것은 **부호 일관성과 수준 간
    단조**이므로 수준별 범위가 그것을 그대로 보여 준다.

    절대 규칙 11: 산출물에 "가정값 기반 — 실측 아님" 표시를 반드시 넣는다.
    """
    levels = leak_levels()[1:]
    header = (
        f"{'누출 수준':<12}{'누출랙 유량':>22}{'타랙 유량':>22}"
        f"{'총유량':>22}{'누출랙 출구온도':>24}{'펌프 양정':>22}{'balance':>20}"
    )
    units = (
        f"{'':<12}{'[%]':>22}{'[%]':>22}{'[%]':>22}{'[K]':>24}"
        f"{'[mAq]':>22}{'[%]':>20}"
    )
    lines = [
        "세션 4 · 누출 정상상태 신호 (32조합 범위 요약 · 랙 "
        f"{signals[0].rack_index} 주입)",
        "※ " + ASSUMPTION_TAG,
        "",
        header,
        units,
        "-" * len(header),
    ]

    def span(values: list[float], digits: int) -> str:
        return f"{min(values):+.{digits}f} ~ {max(values):+.{digits}f}"

    for level in levels:
        rows = [s for s in signals if s.level.label == level.label]
        lines.append(
            f"{level.label:<12}"
            f"{span([r.leak_rack_flow_change_percent for r in rows], 4):>22}"
            f"{span([r.other_rack_flow_change_percent for r in rows], 4):>22}"
            f"{span([r.total_flow_change_percent for r in rows], 4):>22}"
            f"{span([r.leak_rack_outlet_change_C for r in rows], 5):>24}"
            f"{span([r.pump_head_change_mAq for r in rows], 5):>22}"
            f"{span([r.energy_balance_residual_percent for r in rows], 5):>20}"
        )
    lines += [
        "-" * len(header),
        "",
        "변화량은 전부 **같은 조합의 정상 해 대비**다 (조합 안에서만 비교한다).",
        "타랙 유량 = 누출이 걸리지 않은 랙 하나 (나머지 7랙은 전부 동일 조건).",
        "펌프 양정 = 정속 곡선 위 운전점 이동 (5-1 「펌프 운전 방식」).",
        f"balance = 누출 상태의 energy balance 잔차 (6장 기준 <0.1%). "
        f"최대 |잔차| = "
        f"{max(abs(s.energy_balance_residual_percent) for s in signals):.5f} %",
        "",
        "※ " + ASSUMPTION_TAG,
        SESSION_4_CAVEAT,
    ]
    return "\n".join(lines)


def main() -> int:
    import sys

    if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔 기본 인코딩 대비
        sys.stdout.reconfigure(encoding="utf-8")
    print(format_steady_signal_table(all_steady_signals()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
