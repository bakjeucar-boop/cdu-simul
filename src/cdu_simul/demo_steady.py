"""시연용 정상상태 실행 → JSON (세션 7.0).

**파일럿 이후의 판이다.** 파일럿은 2026-08-31 에 종료 판정이 났고
(`project-overview.md` 「판정 결과」 — 「전환 O, 조건부」), 이 모듈은 판정 8 이
가리킨 「시연 데이터 준비」다. **게이트가 없고 feasibility 를 판정하지 않는다.**

**데이터셋(52열본)과 별개다.** 이 모듈은 `dataset.generate` 를 호출하지 않고 CSV
를 건드리지 않는다. 데이터셋은 5장 부하 양 끝 2수준뿐이라 「부하가 점점 오른다」를
보여주지 못한다 — 그래서 시연 전용으로 부하를 촘촘히 훑는다(세션 7.0 §3⑴).

**정상상태만 돌린다. 전이를 돌리지 않는다.** 전이 파형이 노드 수에 수렴하지
않기 때문이다(미해결 #40 · 세션 5.8 C — 2노드와 16노드가 순신호 대비
31.72~57.96% 차이 · N≤64 미수렴). 그래서 **케이스 배열은 시간축이 아니다** —
화면에서 케이스가 바뀌는 것은 **케이스 전환의 표시**이지 시간 경과가 아니다.

**「누출」을 단독으로 쓰지 않는다.** 필드·라벨은 `resistance_increase` 다
(랙 배관 K값 증가 = 이상 상태 「막힘」 · 절대 규칙 8). 파일럿 종료 판정 5-b 가
**X** 이기 때문이다 — 여기 실린 구분은 「막힘」이고 **「샘」의 방향으로 읽으면
안 된다**(세션 5.6 수력 · 세션 5.7-D 열에서 부호가 정반대로 나왔다).
`leak_model` 표기를 함께 싣는다.

**표시는 HTML 이 맡는다.** 이 모듈은 완성된 수치만 낸다 — 화면이 계산하지
않게 한다. Python 의존성을 늘리지 않았다(표준 라이브러리 `json` 뿐).

**모든 수치는 가정값 기반이며 실측이 아니다.**
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from cdu_simul.assumptions import (
    ASSUMPTION_TAG,
    HEAT_EXCHANGER,
    LOAD_PROFILE,
    PIPING,
    PLANT,
    PUMP,
    SCENARIO,
    VALVE,
)
from cdu_simul.dataset import LEAK_MODEL_K_APPROX
from cdu_simul.fluid import coolant_density_kgm3
from cdu_simul.hydraulics import bulk_mean_temperature_C
from cdu_simul.hydraulics import default_cases as default_hydraulic_cases
from cdu_simul.leak import LeakLevel, leak_case, leak_levels
from cdu_simul.model import CduCase, CduSteadyStateResult, solve_cdu_steady_state
from cdu_simul.plant import PlantCase, solve_plant_steady_state

#: 저장소 루트의 `demo/` — 절대경로를 박지 않는다(절대 규칙 15).
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "demo"
OUTPUT_FILENAME = "demo_steady.json"

DEMO_VERSION = "demo-7.2-B"

#: L/s × kg/m³ → kg/h 환산계수. 1 L = 1e-3 m³ · 1 h = 3600 s 이므로 3.6 이다.
#: **단위 환산이며 가정치가 아니다**(절대 규칙 9 — 변환 지점을 한 곳에 둔다).
LPS_TO_KGPH_PER_DENSITY: float = 3.6

#: 랙별 등가길이 배분을 **켠다**(세션 7.2 · 5-1 「랙별 등가길이 배분」).
#: 모델 기본값은 균등이고 이 시연만 명시적으로 켠다 — 기존 시험·데이터셋·게이트
#: 기록은 균등 그대로다.
USE_RACK_LENGTH_DISTRIBUTION = True

#: 저항을 거는 랙 = **가장 먼 랙**(마지막 인덱스 · 세션 7.2-B · 5-1 「누출 주입
#: 지점」). 랙별 배분을 켜면 랙이 대칭이 아니므로 랙 번호가 결과에 영향한다 —
#: 가장 가까운 랙(K 배수 0.8)에 +50% 를 걸면 0.8 × 1.5 = 1.2 로 가장 먼 랙과
#: 정확히 겹친다(미해결 #45). 가장 먼 랙에 걸면 1.2 × 1.5 = 1.8 이라 어느 랙과도
#: 겹치지 않는다. **랙 개수에서 유도하며 새 숫자가 아니다.** 균등 경로(52열본
#: 데이터셋)는 랙 대칭이 성립하므로 `LEAK.injection_rack_index` 그대로 둔다.
RESISTANCE_RACK_INDEX = SCENARIO.racks_per_cdu - 1

CONFIG_SINGLE = "single"
CONFIG_DUAL_ASYMMETRIC = "dual_asymmetric"

#: 부하 축 표본 간격 [%].
#:
#: **★ 5장에 없는 숫자다.** 5장은 「유휴 20% ~ 정격 100%」를 **범위로** 주고
#: 그 안의 점을 지정하지 않는다. 따라서 `assumptions.py` 에 넣지 않고 5-1 에도
#: 적지 않는다 — 물리 가정이 아니라 **시연용 표본 추출**이기 때문이다
#: (세션 7.0 §6). 데이터셋(52열본)은 여전히 양 끝 2수준 그대로이며 둘을 섞어
#: 읽지 않는다.
LOAD_STEP_PERCENT = 5.0

#: 다중 CDU 에서 **고정하는 쪽**의 부하 [%]. 5장 정격이므로 새 숫자가 아니다.
#: 한쪽만 올리고 다른 쪽을 고정해야 공유 2차측 배분의 연동이 드러난다.
FIXED_LOAD_PERCENT = LOAD_PROFILE.rated_load_percent

#: 저항 증가를 거는 CDU. 5-1 「「막힘」 주입 지점」이 랙 1개로 정하고 있고,
#: 데이터셋도 CDU 0 에만 건다(`dataset.LEAK_CDU_INDEX`).
RESISTANCE_CDU_INDEX = 0


# ─────────────────────────────────────────────────────────────────────────────
# 축 — 부하만 훑고 나머지는 한 조합에 고정한다
# ─────────────────────────────────────────────────────────────────────────────
def load_points_percent() -> tuple[float, ...]:
    """부하 축 [%] — 5장 양 끝을 포함해 `LOAD_STEP_PERCENT` 간격으로 훑는다.

    양 끝(유휴 20 · 정격 100)이 반드시 들어가야 한다(세션 7.0 §6). 간격이 범위를
    정수로 나누지 못하면 그 조건이 깨지므로 여기서 막는다.
    """
    low = LOAD_PROFILE.idle_load_percent
    high = LOAD_PROFILE.rated_load_percent
    steps = round((high - low) / LOAD_STEP_PERCENT)
    if abs(low + steps * LOAD_STEP_PERCENT - high) > 1.0e-9:
        raise ValueError(
            f"간격 {LOAD_STEP_PERCENT}% 가 {low}~{high}% 를 나누지 못한다 — "
            "5장 양 끝이 표본에서 빠진다"
        )
    return tuple(low + LOAD_STEP_PERCENT * i for i in range(steps + 1))


def fixed_case_template() -> CduCase:
    """부하 말고 다른 축을 고정한 CDU 케이스 (부하는 케이스마다 덮어쓴다).

    **5장·5-1 범위 축의 하단 조합**이다 — 양정 20 mAq · 분기 ΔP 2 mAq ·
    밸브 ΔP 3 mAq · NTU 2 · 2차측 공급 27℃. 세션 5 의 대표 표
    (`plant.format_plant_table`)가 이미 「수력·NTU·2차측을 범위 하단으로 고정」한
    3케이스를 내고 있으므로, **같은 조합을 골라야 기존 기록과 나란히 읽힌다.**
    시연은 전수가 필요 없다(세션 7.0 §6).

    `default_hydraulic_cases()[0]` 이 그 하단 조합인지 여기서 확인한다 —
    순서가 바뀌면 조용히 다른 조합을 고르게 되기 때문이다.
    """
    hydraulic = default_hydraulic_cases()[0]
    if hydraulic.pump != PUMP.curve_coefficients_at_head_low:
        raise ValueError(
            f"수력 케이스 0 ({hydraulic.label}) 이 양정 하단 조합이 아니다"
        )
    if USE_RACK_LENGTH_DISTRIBUTION:
        hydraulic = replace(
            hydraulic,
            branch_K_multipliers=PIPING.rack_branch_K_multipliers(hydraulic.n_racks),
        )
    return CduCase(
        hydraulic=hydraulic,
        T_secondary_supply_C=SCENARIO.T_secondary_supply_C.low,
        ntu=HEAT_EXCHANGER.ntu.low,
    )


@dataclass(frozen=True)
class ResistanceLevel:
    """랙 배관 저항 증가 1수준. 기준(증가 없음)도 같은 형태로 담긴다.

    **「누출」이라 부르지 않는다**(세션 7.0 §6). 값 자체는 5장 누출 3수준
    (+5/+20/+50%)이고 `leak.leak_levels()` 에서 그대로 읽는다 — 새 숫자가 아니다.
    """

    increase_percent: float
    k_multiplier: float

    @property
    def label(self) -> str:
        if self.k_multiplier == 1.0:
            return "기준 (저항 증가 없음)"
        return f"랙 저항 +{self.increase_percent:g}%"


def resistance_levels() -> tuple[ResistanceLevel, ...]:
    """기준 + 5장 3수준. `assumptions.py` 에서만 읽는다(절대 규칙 2)."""
    return tuple(
        ResistanceLevel(
            increase_percent=round((level.k_multiplier - 1.0) * 100.0, 6),
            k_multiplier=level.k_multiplier,
        )
        for level in leak_levels()
    )


def _leak_level_for(level: ResistanceLevel) -> LeakLevel:
    """`ResistanceLevel` → `leak.LeakLevel`. 배율 1.0 도 같은 경로를 통과한다."""
    return next(lv for lv in leak_levels() if lv.k_multiplier == level.k_multiplier)


# ─────────────────────────────────────────────────────────────────────────────
# 표기 — 케이스마다 반복해서 싣는다 (5-1 스키마 규약과 같은 취지)
# ─────────────────────────────────────────────────────────────────────────────
CASE_CAVEATS: dict[str, str] = {
    "assumption_tag": ASSUMPTION_TAG,
    "steady_only": (
        "정상상태 해다. 케이스 배열은 **시간축이 아니다** — 화면에서 케이스가 "
        "바뀌는 것은 케이스 전환의 표시이지 시간 경과가 아니다. 전이는 돌리지 "
        "않았다: 전이 파형이 노드 수에 수렴하지 않는다(미해결 #40 · 세션 5.8)"
    ),
    "load_sampling": (
        "부하 값은 **시연용 범위 샘플링**이다 — 5장은 「유휴 20% ~ 정격 100%」를 "
        "범위로만 주고 그 안의 점을 지정하지 않는다. 그래서 assumptions.py 와 "
        "5-1 에 넣지 않았다. 데이터셋(52열본)은 여전히 양 끝 2수준이므로 둘을 "
        "섞어 읽지 않는다"
    ),
    "resistance_proxy": (
        "resistance_increase_percent 는 **랙 배관 K값 증가**이며 이상 상태 "
        "「막힘」이다(절대 규칙 8). **「샘」(질량손실)의 대용이 아니라 독립된 "
        "이상 상태**다 — 파일럿 종료 판정 5-b 가 X 인 것이 이것이다: 여기 실린 "
        "구분을 「샘」의 방향으로 읽으면 안 된다(세션 5.6 수력 · 세션 5.7-D "
        "열에서 부호가 정반대로 나왔다)"
    ),
    "total_flow_caveat": (
        "총유량만 보면 +50% 도 −0.16~0.37% 라 놓칠 수 있다(세션 4 관측 ③). "
        "랙별 유량·랙별 출구온도와 함께 읽는다. "
        "**이 수치도 처방도 「막힘」 것이다** — 이 화면은 「막힘」만 싣는다"
        "(resistance_proxy 표기). 「샘」은 총유량이 **반대 방향(증가)** 으로 "
        "움직이고 랙에 국소화되지 않아 랙별 열이 어느 랙인지 가리키지 못한다"
        "(미해결 #36 · 절대 규칙 8) — 이 문언을 「샘」 쪽으로 옮겨 읽지 않는다"
    ),
    "fixed_axes": (
        "부하 말고 다른 5장·5-1 범위 축은 한 조합에 고정했다 — meta.fixed_axes 를 "
        "본다. 다른 조합에서는 절대값이 달라지므로 조합 밖과 비교하지 않는다"
    ),
    "excluded_tags": (
        "압력·양정·ΔP 태그를 싣지 않았다 — 계통 정압이 모델에 없고 잔여저항이 "
        "집중저항 하나다(미해결 #24). 2차측 환수온도는 모델에 없어 뺐다"
        "(절대 규칙 7). demo/tag-map.md 를 본다"
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# 케이스 실행
# ─────────────────────────────────────────────────────────────────────────────
def _cdu_tags(
    result: CduSteadyStateResult,
    secondary_share_Lps: float,
    plant_solver_ier: int | None,
) -> dict[str, Any]:
    """CDU 1대분 태그 묶음 — `demo/tag-map.md` 「1. 채운 태그」 그대로.

    절대 규칙 5: solver 플래그를 전부 싣는다. **실패해도 버리지 않는다.**
    """
    case = result.case
    # 질량유량의 ρ 는 5-1 「수력 계산의 물성 평가 온도」 규칙 그대로 **1차측
    # 벌크평균온도**에서 얻는다(세션 7.2 C3). 새 상수를 만들지 않는다.
    density_kgm3 = coolant_density_kgm3(
        bulk_mean_temperature_C(result.thermal.T_supply_C, result.thermal.T_return_C)
    )

    def kgph(flow_Lps: float) -> float:
        return flow_Lps * density_kgm3 * LPS_TO_KGPH_PER_DENSITY

    return {
        "load_percent": case.load_percent,
        "load_kW": case.rack_load_kW * SCENARIO.racks_per_cdu,
        "T_supply_C": result.thermal.T_supply_C,
        "T_return_C": result.thermal.T_return_C,
        "total_flow_Lps": result.flow.total_flow_Lps,
        "total_flow_kgph": kgph(result.flow.total_flow_Lps),
        "hx_duty_kW": result.thermal.hx_duty_kW,
        "T_secondary_supply_C": case.T_secondary_supply_C,
        "secondary_share_Lps": secondary_share_Lps,
        "secondary_share_kgph": kgph(secondary_share_Lps),
        "rack_flows_Lps": list(result.flow.rack_flows_Lps),
        "rack_flows_kgph": [kgph(q) for q in result.flow.rack_flows_Lps],
        "rack_outlet_C": list(result.thermal.rack_return_temps_C),
        "solver": {
            "hydraulic_ier": result.flow.solver_ier,
            "thermal_converged": result.thermal.solver_converged,
            "outer_converged": result.outer_solver_converged,
            "plant_ier": plant_solver_ier,
            "all_converged": (
                result.solver_converged and plant_solver_ier in (None, 1)
            ),
        },
    }


def _case_header(
    case_id: str,
    cdu_config: str,
    load_percent: float,
    level: ResistanceLevel,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "cdu_config": cdu_config,
        "driven_load_percent": load_percent,
        "resistance_increase_percent": level.increase_percent,
        "resistance_k_multiplier": level.k_multiplier,
        "resistance_label": level.label,
        "resistance_rack_index": RESISTANCE_RACK_INDEX,
        "resistance_cdu_index": RESISTANCE_CDU_INDEX,
        "leak_model": LEAK_MODEL_K_APPROX,
        "caveats": CASE_CAVEATS,
    }


def single_case(load_percent: float, level: ResistanceLevel) -> dict[str, Any]:
    """단일 CDU 1케이스. 2차측은 공유가 없으므로 정격 고정값을 받는다."""
    case = leak_case(
        replace(fixed_case_template(), load_percent=load_percent),
        _leak_level_for(level),
        RESISTANCE_RACK_INDEX,
    )
    result = solve_cdu_steady_state(case)
    header = _case_header(
        f"single/L{load_percent:g}/R{level.increase_percent:g}",
        CONFIG_SINGLE,
        load_percent,
        level,
    )
    header["cdus"] = [_cdu_tags(result, HEAT_EXCHANGER.secondary_flow_Lps, None)]
    return header


def dual_case(load_percent: float, level: ResistanceLevel) -> dict[str, Any]:
    """다중 CDU(2대) 1케이스 — CDU 0 만 부하를 올리고 CDU 1 은 정격 고정.

    **연동의 실체는 공유 2차측 총유량이 고정이라는 것이다**(5-1). CDU 0 의 부하가
    오르면 그쪽 1차측 유량이 이동하고 비례 배분이 따라 이동해 **CDU 1 의 배분과
    온도가 바뀐다** — 부하를 올리지 않은 쪽이 움직이는 것이 시연 재료다.

    저항 증가는 CDU 0 의 **가장 먼 랙**에만 건다(5-1 「「막힘」 주입 지점」 · 랙 1개).
    """
    template = fixed_case_template()
    driven = leak_case(
        replace(template, load_percent=load_percent),
        _leak_level_for(level),
        RESISTANCE_RACK_INDEX,
    )
    fixed = replace(template, load_percent=FIXED_LOAD_PERCENT)
    cdus = (driven, fixed) if RESISTANCE_CDU_INDEX == 0 else (fixed, driven)

    case_id = f"dual/L{load_percent:g}/R{level.increase_percent:g}"
    plant = solve_plant_steady_state(PlantCase(label=case_id, cdus=cdus))

    header = _case_header(case_id, CONFIG_DUAL_ASYMMETRIC, load_percent, level)
    header["fixed_cdu_load_percent"] = FIXED_LOAD_PERCENT
    header["cdus"] = [
        _cdu_tags(result, share, plant.top_level_solver_ier)
        for result, share in zip(
            plant.cdu_results, plant.secondary_shares_Lps, strict=True
        )
    ]
    return header


def build_cases() -> list[dict[str, Any]]:
    """단일 CDU + 다중 CDU × 부하 축 × 저항 수준."""
    loads = load_points_percent()
    levels = resistance_levels()
    return [
        *(single_case(load, level) for load in loads for level in levels),
        *(dual_case(load, level) for load in loads for level in levels),
    ]


def expected_case_count() -> int:
    """실행 **전에** 규모를 알기 위한 값 — 구성 2종 × 부하 점 × 저항 수준."""
    return 2 * len(load_points_percent()) * len(resistance_levels())


def build_document() -> dict[str, Any]:
    """JSON 최상위 구조."""
    template = fixed_case_template()
    return {
        "demo_version": DEMO_VERSION,
        "meta": {
            "purpose": "PFD 시연용 정상상태 데이터. 표시는 HTML 이 맡는다",
            "assumption_tag": ASSUMPTION_TAG,
            "leak_model": LEAK_MODEL_K_APPROX,
            "tag_map": "demo/tag-map.md",
            "racks_per_cdu": SCENARIO.racks_per_cdu,
            "cdu_count_dual": PLANT.cdu_count,
            "secondary_total_flow_Lps": PLANT.secondary_total_flow_Lps,
            "mass_flow_density_rule": (
                "질량유량 [kg/h] = 유량 [L/s] × ρ [kg/m³] × 3.6. ρ 는 5-1 "
                "「수력 계산의 물성 평가 온도」 규칙대로 **1차측 벌크평균온도**"
                "(T_supply+T_return)/2 에서 CoolProp 으로 얻는다 — 새 상수를 "
                "만들지 않았다. **secondary_share_kgph 도 같은 ρ 를 쓴다** — "
                "2차측 유체는 1차측과 같은 PG25 이나 온도가 다르므로(2차측 공급 "
                "27℃) 그 스트림의 실제 밀도와는 다르다. 2차측 물성 평가 온도는 "
                "5-1 이 정한 바 없어 새 규칙을 만들지 않았다"
            ),
            "rack_equivalent_length_m": (
                list(PIPING.rack_equivalent_lengths_m(SCENARIO.racks_per_cdu))
                if USE_RACK_LENGTH_DISTRIBUTION
                else None
            ),
            "rack_branch_K_multipliers": (
                list(PIPING.rack_branch_K_multipliers(SCENARIO.racks_per_cdu))
                if USE_RACK_LENGTH_DISTRIBUTION
                else None
            ),
            "rack_length_note": (
                "랙별 등가길이 배분을 **켠 상태**로 냈다(5-1 「랙별 등가길이 "
                "배분」 · 세션 7.2). 5장 20~30 m 를 랙 8개의 공간 분포로 읽고 "
                "K 를 8랙 평균(25 m) 기준으로 스케일했다 — 계통 전체 저항은 "
                "보존되고 랙 간 배분만 바뀐다. 모델 기본값은 균등이며 "
                "데이터셋(52열본)과 기존 게이트 기록은 균등 그대로다"
            ),
            "load_step_percent": LOAD_STEP_PERCENT,
            "load_points_percent": list(load_points_percent()),
            "resistance_increase_percent_levels": [
                level.increase_percent for level in resistance_levels()
            ],
            "fixed_axes": {
                "hydraulic_label": template.hydraulic.label,
                "pump_head_rated_mAq": PUMP.rated_head_mAq.low,
                "branch_dp_rated_mAq": PIPING.dP_per_rack_mAq.low,
                "valve_dp_rated_mAq": VALVE.dP_at_rated_opening_mAq.low,
                "ntu": template.ntu,
                "T_secondary_supply_C": template.T_secondary_supply_C,
                "holdup_mass_kg": None,
                "holdup_note": (
                    "M(보유수량)은 이 시연에 쓰이지 않는다 — 정상상태 해가 M 에 "
                    "불변임이 세션 5.7 에서 확인됐다(M 을 8랙 계통 전체로 정정해 "
                    "8배가 됐어도 정상상태 해는 그대로였고 tau·t63·t95 만 8배 "
                    "이동했다). 전이를 돌리지 않으므로 고정할 축이 아니다"
                ),
                "note": (
                    "5장·5-1 범위 축의 **하단 조합**이다. 세션 5 대표 표"
                    "(plant.format_plant_table)와 같은 조합이라 기존 기록과 "
                    "나란히 읽을 수 있다"
                ),
            },
            "excluded_tags": {
                "T_secondary_return_C": (
                    "모델에 없는 양이다 — 2차측은 고정 경계조건이다"
                    "(절대 규칙 7 · 5-1). ε-NTU 로 역산하면 그것은 새 모델이다"
                ),
                "pump_head_mAq · dP · static_pressure": (
                    "계통 정압이 모델에 없고 잔여저항이 집중저항 하나다"
                    "(미해결 #24) — 세션 7.0 §3⑵ 사람 결정"
                ),
            },
            "caveats": CASE_CAVEATS,
        },
        "cases": build_cases(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 쓰기 · 다시 읽어 검증 (세션 7.0 C4·C5)
# ─────────────────────────────────────────────────────────────────────────────
#: 케이스마다 반드시 채워져 있어야 하는 태그 — `demo/tag-map.md` 「1. 채운 태그」.
TAG_KEYS: tuple[str, ...] = (
    "load_percent",
    "load_kW",
    "T_supply_C",
    "T_return_C",
    "total_flow_Lps",
    "hx_duty_kW",
    "T_secondary_supply_C",
    "secondary_share_Lps",
    "rack_flows_Lps",
    "rack_outlet_C",
    # 질량유량 (세션 7.2 C3) — L/s 열은 지우지 않고 함께 둔다.
    "total_flow_kgph",
    "secondary_share_kgph",
    "rack_flows_kgph",
)


def verify(document: dict[str, Any]) -> list[str]:
    """케이스 수 · 태그 빈 칸 · 표기 존재를 확인한다. 반환은 **문제 목록**이다."""
    problems: list[str] = []
    cases = document["cases"]
    if len(cases) != expected_case_count():
        problems.append(
            f"케이스 수 {len(cases)} 가 예상 {expected_case_count()} 와 다르다"
        )
    for case in cases:
        case_id = case.get("case_id", "?")
        if case.get("caveats") != CASE_CAVEATS:
            problems.append(f"{case_id}: 표기(caveats)가 없거나 다르다")
        if not case.get("leak_model"):
            problems.append(f"{case_id}: leak_model 표기가 없다")
        for cdu in case["cdus"]:
            for key in TAG_KEYS:
                if cdu.get(key) in (None, "", []):
                    problems.append(f"{case_id}: 태그 {key} 가 비어 있다")
            for key in ("rack_flows_Lps", "rack_outlet_C"):
                if len(cdu[key]) != SCENARIO.racks_per_cdu:
                    problems.append(f"{case_id}: {key} 길이가 랙 수와 다르다")
    return problems


def failed_cases(document: dict[str, Any]) -> list[str]:
    """solver 가 수렴하지 않은 케이스 — **버리지 않고 목록으로 낸다**(절대 규칙 5)."""
    return [
        f"{case['case_id']}#cdu{index}"
        for case in document["cases"]
        for index, cdu in enumerate(case["cdus"])
        if not cdu["solver"]["all_converged"]
    ]


def write(document: dict[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / OUTPUT_FILENAME
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return path


def main() -> int:
    print(f"시연용 정상상태 실행 — {expected_case_count()} 케이스")
    print("※ " + ASSUMPTION_TAG)

    path = write(build_document())
    reread = json.loads(path.read_text(encoding="utf-8"))
    problems = verify(reread)
    failures = failed_cases(reread)

    print(
        f"기록: {path.relative_to(DEFAULT_OUTPUT_DIR.parent)}"
        f" · {path.stat().st_size / 1024:.0f} KB"
        f" · 케이스 {len(reread['cases'])}"
    )
    print(f"solver 미수렴: {len(failures)} 건" + (f" — {failures}" if failures else ""))
    if problems:
        print("검증 실패:")
        for problem in problems:
            print("  - " + problem)
        return 1
    print("검증 통과 — 케이스 수 · 태그 빈 칸 0 · 표기 존재")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
