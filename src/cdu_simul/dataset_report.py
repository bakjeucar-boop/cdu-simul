"""데이터셋 요약 리포트 — 파일럿 종료 판단의 재료 (세션 5.5-B).

**판정하지 않는다.** `project-overview.md` 「파일럿 종료 조건」이 "판정 주체: 사람.
웹 대화창도 Claude Code도 판정하지 않는다" 라고 적었다. 이 모듈은 그 절이 **볼 것**
으로 적은 넷에 각각 답하는 재료를 모아 놓을 뿐이다.

CSV 자체는 사람이 읽을 물건이 아니다 — 그래서 읽을 수 있는 요약을 따로 낸다.

**모든 수치는 가정값 기반이며 실측이 아니다.**
"""

from __future__ import annotations

from cdu_simul.assumptions import ASSUMPTION_TAG
from cdu_simul.dataset import enumerate_specs
from cdu_simul.leak import all_steady_signals, leak_levels
from cdu_simul.model import (
    default_cdu_cases,
    energy_balance_residual_percent,
    solve_cdu_steady_state,
)

#: `project-overview.md` 「파일럿 종료 조건」이 볼 것으로 적은 넷.
PILOT_QUESTIONS: tuple[str, ...] = (
    "구성요소가 각각 독립된 모델로 서 있고, 물렸을 때 전체 거동이 나오는가",
    "6장 네 기준의 판정 결과 — 각 기준이 실제로 무엇을 쟀는지의 단서와 함께",
    "시나리오 간 신호가 구분 가능한 형태로 나오는가",
    "가정치를 바꿨을 때 결과가 예상대로 움직이는가 (5장 범위 양 끝 대조)",
)


def _component_map() -> list[tuple[str, str, str]]:
    """(구성요소, 맡은 모듈·함수, 무엇을 푸는가)."""
    return [
        (
            "펌프",
            "hydraulics.pump_head_mAq",
            "H = H0 - aQ - bQ² 정속 곡선. RPM 은 모델에 없다(5-1 「펌프 운전 방식」)",
        ),
        (
            "밸브",
            "hydraulics.valve_dp_mAq · valve_Kv_max_m3h_from_rated_dP",
            "선형 개도 특성 · Kv 는 5장 조건에서 역산해 고정",
        ),
        (
            "배관 (랙 분기)",
            "hydraulics.branch_dp_mAq · branch_K_from_rated_dP",
            "ΔP = K·ρv²/2 · K 는 5장 ΔP 에서 역산",
        ),
        (
            "계통 잔여저항",
            "hydraulics.residual_dp_mAq",
            "HX 1차측+CDU 내부+헤더를 집중저항 하나로 (총양정의 60~83% · 미해결 #24)",
        ),
        (
            "랙 (발열)",
            "model.SteadyStateCase.rack_loads_kW · rack_return_temps_C",
            "랙별 현열 상승 후 환수 헤더에서 유량가중 혼합",
        ),
        (
            "열교환기",
            "model.hx_capacity_terms · hx_effectiveness_counterflow",
            "대향류 ε-NTU · Cr 은 양측 물성에서 유도(세션 5-B)",
        ),
        (
            "계통 열용량",
            "dynamics.holdup_bounds · system_coolant_mass_kg",
            "배관 보유량만 (HX·CDU 내부·콜드플레이트 누락 · 미해결 #21·#31)",
        ),
        (
            "공유 2차측",
            "plant.solve_plant_steady_state · PLANT.secondary_shares_Lps",
            "총유량 고정 · 1차측 유량 비례 배분. 온도는 고정 경계조건",
        ),
        (
            "누출",
            "hydraulics.apply_leak_to_rack · leak.py",
            "배관 K값 증가 근사(5장 정의). 질량 손실은 모델에 없다",
        ),
    ]


def _gate_results() -> list[tuple[str, str, str]]:
    """(6장/세션 게이트, 판정, **실제로 잰 것** — 과대해석 금지 단서)."""
    return [
        (
            "6장 ① energy balance <0.1%",
            "통과 (32조합 최대 |잔차| 0.00506%)",
            "잰 것은 **상수 cp 선형화가 CoolProp 엔탈피 차와 어긋나지 않는가** 하나다"
            "(세션 1-B ※). HX duty 항등식은 구조상 0 이라 게이트에 쓰지 않았다. "
            "통과의 뜻은 「무너지지 않는다」까지이고 「정확하다」가 아니다 — 실측이 없다",
        ),
        (
            "6장 ② T_return 방향성",
            "통과 (8랙 16조합)",
            "부하 스텝의 **부호** 하나다(세션 2 ※2). 크기·시간은 판정하지 않았다",
        ),
        (
            "6장 ③ 극단 케이스 비발산",
            "통과 (32조합 × 부하 0/100% + 동적)",
            "부하 0 에서 1차측이 2차측 온도로 정확히 수렴함을 **사전에 적고** 대조했다"
            "(세션 3-B). 부하 0 은 5장 범위 밖이지만 6장이 발산 검사용으로 명시한 값이다",
        ),
        (
            "6장 ④ 수렴시간",
            "**판정하지 않았다**",
            "M 에 HX·CDU 내부·콜드플레이트 보유량이 빠져 있고(미해결 #21), 8랙에서 "
            "M 을 어떻게 읽어야 하는지도 5장·5-1 에 없다(미해결 #31). "
            "τ·t63·t95 의 **절대값을 해석하지 않는다**",
        ),
        (
            "세션 3 유량분배 수렴",
            "통과 (열결합 32조합 전수)",
            "수렴은 **해가 있다**는 뜻이지 값이 타당하다는 뜻이 아니다 — "
            "잔여저항이 총양정의 60~83% 라 유량분배는 그 배정 방식이 지배한다(#24)",
        ),
        (
            "세션 4 누출 신호 식별",
            "통과 (기준 A 부호 · B 단조 · C 잡음 대비)",
            "「식별 가능」은 **모델 안에서 잡음 위에 있다**는 뜻이지 실측 가능하다는 "
            "뜻이 아니다. 특히 펌프 양정 신호는 양정 대비 상대 5e-5 로 매우 약하다",
        ),
        (
            "세션 5 CDU 간 연동 수렴",
            "통과 (96 케이스)",
            "연동은 **유량 경로 하나**뿐이다 — 2차측 공급온도가 고정이라 열 경로 "
            "결합이 모델에 없다(5-1 한계). 연동이 작더라도 「상호작용이 작다」로 "
            "읽지 않는다",
        ),
        (
            "세션 5.5 상태 이월 없음",
            "통과 (A 순서 무관 · B 단독 재현 · C 정상 복귀 · D 전 행 표기)",
            "표본 검사다(전량 1,792 중 stride 61). 표본이 축을 다 덮는지도 테스트가 본다",
        ),
    ]


def _signal_separability() -> list[tuple[str, str, str, str]]:
    """(누출 수준, 누출랙 유량, 누출랙 출구온도, 펌프 양정) — 32조합 범위."""
    signals = all_steady_signals()
    rows = []
    for level in leak_levels()[1:]:
        subset = [s for s in signals if s.level.label == level.label]
        flows = [s.leak_rack_flow_change_percent for s in subset]
        temps = [s.leak_rack_outlet_change_C for s in subset]
        heads = [s.pump_head_change_mAq for s in subset]
        rows.append(
            (
                level.label,
                f"{min(flows):+.4f} ~ {max(flows):+.4f} %",
                f"{min(temps):+.5f} ~ {max(temps):+.5f} K",
                f"{min(heads):+.5f} ~ {max(heads):+.5f} mAq",
            )
        )
    return rows


def _range_sensitivity() -> list[tuple[str, str, str]]:
    """5장 범위 양 끝을 바꿨을 때 결과가 어디로 움직이는가 (예상 대비)."""
    results = {case.label: solve_cdu_steady_state(case) for case in default_cdu_cases()}
    returns = {label: r.thermal.T_return_C for label, r in results.items()}
    residuals = [
        abs(energy_balance_residual_percent(r.thermal)) for r in results.values()
    ]

    def span(predicate) -> str:  # type: ignore[no-untyped-def]
        values = [v for label, v in returns.items() if predicate(label)]
        return f"{min(values):.4f} ~ {max(values):.4f} ℃"

    return [
        (
            "2차측 공급온도 27 → 30 ℃",
            "환수온도가 **같이 오른다**(열침 온도가 오르므로)",
            f"27℃: {span(lambda x: 'T2nd=27' in x)}"
            f" → 30℃: {span(lambda x: 'T2nd=30' in x)}",
        ),
        (
            "NTU 2 → 3",
            "열교환이 좋아져 환수온도가 **내려간다**",
            f"NTU2: {span(lambda x: 'NTU=2' in x)}"
            f" → NTU3: {span(lambda x: 'NTU=3' in x)}",
        ),
        (
            "펌프 양정 20 → 30 mAq",
            "유량이 조금 늘어 ΔT 가 **조금 줄어든다**",
            f"H20: {span(lambda x: 'H22.4' in x)}"
            f" → H30: {span(lambda x: 'H33.6' in x)}",
        ),
        (
            "energy balance (전 조합)",
            "범위 어디서나 보존법칙이 성립해야 한다",
            f"최대 |잔차| {max(residuals):.5f} % (기준 <0.1%)",
        ),
    ]


#: 열린 한계 — (항목, 데이터셋의 **어느 열**에 영향하는가).
OPEN_LIMITS: tuple[tuple[str, str], ...] = (
    (
        "#21 M 결손 — HX·CDU 내부·콜드플레이트 보유량이 M 에 없다",
        "`t_s` · `holdup_mass_kg` · 전이 행의 `T_supply_C`·`T_return_C` 시간축 전체. "
        "정상상태 행은 영향 없다",
    ),
    (
        "#24 잔여저항이 유량분배를 지배 (총양정의 60~83%)",
        "`rack{i}_flow_Lps` · `total_flow_Lps` · `pump_head_mAq` — 유량 관련 열 전부. "
        "랙 간 상대 비교는 덜 영향받고 절대값이 크게 영향받는다",
    ),
    (
        "#31 M 의 8랙 해석 부재 — M 은 랙 1개 회로분이다",
        "`t_s` · `holdup_mass_kg`. **전이 시간 규모의 절대값을 해석할 수 없다**",
    ),
    (
        "#32 밸브 선형 개도 특성 미검증 (개도 80% 고정으로만 돌렸다)",
        "현재 데이터셋에는 **영향 없다** — 개도가 축에 없다. 개도 제어가 들어오면 "
        "`rack{i}_flow_Lps` 가 특성에 따라 갈린다",
    ),
    (
        "K값 근사의 부호 문제 — 누출은 저항 증가이지 계통 밖 유출이 아니다",
        "`total_flow_Lps` · `pump_head_mAq` — **부호가 실제와 반대일 수 있다**. "
        "`rack{i}_flow_Lps`·`rack{i}_outlet_C` 는 이 불확실성에 해당하지 않는다",
    ),
    (
        "고정 NTU 가정 — 유량이 크게 변하면 UA 가 따라 변해야 한다",
        "`ntu` 열이 5장 값으로 고정이다. 누출로 유량이 −9.7% 까지 움직이는 행에서 "
        "`T_supply_C`·`T_return_C`·`hx_duty_kW` 가 그만큼 어긋난다",
    ),
    (
        "2차측 온도 고정 — CDU 간 열 경로 결합이 없다",
        "`T_secondary_supply_C` 가 상수다. `secondary_share_Lps` 를 통한 유량 경로 "
        "연동만 데이터에 있고, 한 CDU 의 방열이 다른 CDU 를 데우는 경로는 없다",
    ),
)


def format_report() -> str:
    """파일럿 종료 판단용 요약 (순수 함수). **판정하지 않는다.**"""
    specs = enumerate_specs()
    steady = sum(1 for s in specs if s.regime == "steady")
    lines = [
        "=" * 78,
        "CDU 파일럿 데이터셋 요약 — 파일럿 종료 판단의 재료",
        "=" * 78,
        "※ " + ASSUMPTION_TAG,
        "※ **이 문서는 판정하지 않는다.** 판정 주체는 사람이다",
        "   (`project-overview.md` 「파일럿 종료 조건」).",
        "",
        f"데이터셋: 시나리오 {len(specs):,}개 "
        f"(정상상태 {steady:,} · 전이 {len(specs) - steady:,})",
        "",
        "─" * 78,
        f"① {PILOT_QUESTIONS[0]}",
        "─" * 78,
    ]
    for component, module, what in _component_map():
        lines.append(f"  {component:<14} {module}")
        lines.append(f"  {'':<14} └ {what}")
    lines += [
        "",
        "  → 아홉 구성요소가 각각 독립 함수·모듈로 서 있고, `plant` → `model` →",
        "     `hydraulics` 3중 구조로 물려 전체 거동을 낸다. 하이브리드(절대 규칙 4)",
        "     대로 압력-유량은 대수, 온도는 시간적분이다.",
        "",
        "─" * 78,
        f"② {PILOT_QUESTIONS[1]}",
        "─" * 78,
    ]
    for gate, verdict, caveat in _gate_results():
        lines.append(f"  {gate}")
        lines.append(f"    판정: {verdict}")
        lines.append(f"    실제로 잰 것: {caveat}")
        lines.append("")
    lines += [
        "─" * 78,
        f"③ {PILOT_QUESTIONS[2]}",
        "─" * 78,
        f"  {'누출 수준':<12}{'누출랙 유량':>24}{'누출랙 출구온도':>26}{'펌프 양정':>24}",
    ]
    for level, flow, temp, head in _signal_separability():
        lines.append(f"  {level:<12}{flow:>24}{temp:>26}{head:>24}")
    lines += [
        "",
        "  → 세 신호 모두 정상 대비 부호가 일관되고 수준 간 단조다(세션 4 게이트).",
        "     **다만 총유량 신호는 +50% 누출에서도 −0.16~0.37% 다** — 총유량만",
        "     계측하는 설비라면 대규모 누출도 놓칠 수 있다. 랙 단위 열이 필요하다.",
        "",
        "─" * 78,
        f"④ {PILOT_QUESTIONS[3]}",
        "─" * 78,
    ]
    for axis, expectation, observed in _range_sensitivity():
        lines.append(f"  {axis}")
        lines.append(f"    예상: {expectation}")
        lines.append(f"    관측: {observed}")
    lines += [
        "",
        "─" * 78,
        "열린 한계 — 각각 데이터셋의 어느 열에 영향하는가",
        "─" * 78,
    ]
    for limit, columns in OPEN_LIMITS:
        lines.append(f"  · {limit}")
        lines.append(f"      영향 열: {columns}")
    lines += [
        "",
        "=" * 78,
        "※ " + ASSUMPTION_TAG,
        "※ 세션 5-B 에서 Cr 을 선언값에서 유도값으로 바꿨다 — 세션 5-B 이전 로그의",
        "   수치를 이 데이터셋과 나란히 읽을 수 없다.",
        "=" * 78,
    ]
    return "\n".join(lines)


def main() -> int:
    import sys

    if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔 기본 인코딩 대비
        sys.stdout.reconfigure(encoding="utf-8")
    print(format_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
