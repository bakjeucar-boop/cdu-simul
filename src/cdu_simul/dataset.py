"""시나리오 배치 실행 → 텍스트 데이터셋 (세션 5.5-B · 파일럿 마지막 판).

**AI 학습 데이터셋으로 쓸 산출물이다.** 그래서 전 행에 출처와 한계를 붙인다
(`CLAUDE.md` 절대 규칙 11 · 세션 5.5 게이트) — 이 데이터로 학습한 모델의 한계를
아는 유일한 단서가 될 수 있기 때문이다.

**규모·표기·부하 수준·누출 랙은 사람이 정했다**(세션 5.5-B 「사람이 정한 것」).
이 모듈이 다시 정하지 않는다. 규약은 5-1 「데이터셋 스키마 규약」에 있다.

행 단위 = **(시나리오 × CDU × 시각)** 하나당 한 행. 랙별 값은 열로 편다
(rack0..rack7) — 행으로 펴면 CDU 총계가 8번 중복된다.
**열 목록은 이 코드가 정본이다**(5-1 「데이터셋 스키마 규약」).

**전이 행을 `cdu_config` 로 갈라 읽어야 한다 — 자극 형태가 다르다** [세션 5.5-D]::

    cdu_config = single           누출이 t=0 에 **스텝으로** 들어온다. 부하는 불변.
                                  → 관측되는 전이는 **누출이 만든 것**이다.
    cdu_config = dual_asymmetric  **부하가 t=0 에 스텝**한다(CDU A 만).
                                  누출은 **t=0 이전부터 있는 조건**이고 CDU
                                  `LEAK_CDU_INDEX` 하나에만 걸린다.
                                  → 관측되는 전이는 **부하 스텝이 만든 것**이고,
                                    누출은 그 전이의 **모양을 바꾸는** 조건이다.

두 전이를 같은 자극으로 보면 안 된다. 다중 CDU 전이에서 누출의 효과를 보려면
같은 부하 스텝 안에서 `leak_level_percent` 를 가로질러 비교한다.
**다중 전이도 누출을 스텝으로 넣을지는 사람이 정한다** — 세션 5.5-D 는 바꾸지
않았고 의견만 보고했다.

**실패한 solver 의 행을 버리지 않는다.** 버리면 데이터셋이 성공만 담아 편향된다 —
플래그를 전 행에 담고 실패도 그대로 남긴다.

**출력은 저장소 밖이다**(`results/`, `.gitignore` 대상). 생성 스크립트와 메타데이터만
커밋한다.

**모든 수치는 가정값 기반이며 실측이 아니다.**
"""

from __future__ import annotations

import csv
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from cdu_simul.assumptions import (
    ASSUMPTION_TAG,
    HEAT_EXCHANGER,
    LEAK,
    LOAD_PROFILE,
    PLANT,
    SCENARIO,
)
from cdu_simul.dataset_plan import PROVENANCE_COLUMNS
from cdu_simul.dynamics import (
    LeakStepCase,
    holdup_bounds,
    integrate_leak_step,
)
from cdu_simul.leak import leak_case, leak_levels
from cdu_simul.model import (
    CduCase,
    default_cdu_cases,
    energy_balance_residual_percent,
    solve_cdu_steady_state,
)
from cdu_simul.plant import (
    PlantCase,
    PlantLoadStepCase,
    integrate_plant_load_step,
    solve_plant_steady_state,
)

#: 출력 위치 — 저장소 루트 기준 상대경로(절대 규칙 15). `.gitignore` 대상이다.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results"

#: CDU 구성 축 [시나리오 정의: 세션 5.5-A C2 · 사람이 정한 규모].
CONFIG_SINGLE = "single"
CONFIG_DUAL_SYMMETRIC = "dual_symmetric"
CONFIG_DUAL_ASYMMETRIC = "dual_asymmetric"

#: 전이에서 제외하는 구성 — 2대 대칭은 단일 CDU 를 **정확히** 재현한다
#: (세션 5-B: 편차 0.000e+00 K). 정상상태에는 남긴다: 케이스당 수십 ms 이고,
#: 사용자가 그 동치성을 데이터셋 안에서 직접 확인할 수 있는 재료가 된다.
TRANSIENT_CONFIGS = (CONFIG_SINGLE, CONFIG_DUAL_ASYMMETRIC)

#: 누출을 거는 CDU 번호 [시나리오 정의: 5-1 「누출 주입 지점」의 적용 · 세션 5.5-D].
#:
#: 5-1 은 "**랙 1개**에 주입한다 · 전 랙 동시 누출은 5장에 근거가 없어 돌리지
#: 않는다" 라고만 적는다. CDU 가 2대인 플랜트에서 그 규정을 지키려면 누출점이
#: 계 전체에 **하나**여야 하고, 그러려면 어느 CDU 인지를 골라야 한다. 5-1 이 랙
#: 번호에 대해 쓴 논리(대칭이므로 번호 선택이 새 숫자를 만들지 않는다)를 그대로
#: 쓴다 — CDU 사양은 전부 같다(`PlantCase` docstring · 5-1 「CDU 대수」).
#: **정상상태 경로는 세션 5.5-B 부터 이미 CDU 0 에만 걸고 있었다** — 이 상수는
#: 그 이미 있던 선택에 이름을 붙이고 전이 경로에도 같게 적용하는 것이다.
#:
#: **한계**: 비대칭 전이에서 CDU A 는 부하가 **스텝하는** 쪽이고 B 는 **불변인**
#: 쪽이다. 즉 두 CDU 는 사양은 같아도 **역할이 다르므로**, 누출을 A 에 고정하면
#: 데이터셋은 "부하가 움직이는 CDU 에 누출이 있는 경우"만 담는다. 「누출이 정지
#: 상태의 CDU 에 있고 옆 CDU 가 부하 변동을 겪는」 조합은 이 데이터셋에 없다.
LEAK_CDU_INDEX: int = 0

#: 데이터셋 판본 식별자 [사람 결정 · 세션 5.7].
#:
#: 값은 **그 판본을 만든 세션 번호**다. 이 저장소의 판본 축이 세션이기 때문이고,
#: 다음 판본은 자기 세션 번호를 붙여 그대로 이어붙이면 된다. 앞선 두 판본은 이
#: 열 자체가 없다 — **열의 부재가 곧 5.7 이전**이라는 표지이며, 열 수로도 갈린다
#: (48열본 = 세션 5.5-B · 49열본 = 세션 5.5-D · **52열본 = 세션 5.7**).
#: 파일명은 바꾸지 않는다(`cdu_dataset.csv`) — 판본은 파일 안에서 읽는다.
DATASET_VERSION: str = "session-5.7"

#: 누출을 무엇으로 모사했는가 [사람 결정 · 세션 5.7]. **전 행 상수**다.
#:
#: 이 데이터셋의 누출은 전부 **배관 K값 증가 근사**다(절대 규칙 8 · 5장 누출
#: 시나리오). 질량손실 모사는 `leak_massloss.py` 에만 있고 데이터셋 생성 경로에
#: 들어가지 않는다(세션 5.6). 두 모사는 총유량·누출랙 유량의 **부호가 반대**이므로
#: (세션 5.6 · 미해결 #36) 이 열이 없으면 나중에 어느 쪽 데이터인지 알 수 없다.
LEAK_MODEL_K_APPROX: str = "K_approx"

#: 전이 행의 자극 종류 — `ScenarioSpec.stimulus_kind` 가 이 셋 중 하나를 낸다.
#:
#: **새 시나리오를 만들지 않는다.** 이미 있는 행이 무엇으로 자극됐는지를 라벨로
#: 적을 뿐이다 [사람이 정한 것 — 세션 5.7 ⑵]. 단일 CDU 전이는 정격 운전 중
#: **누출이 계단으로** 들어오고(부하 불변), 다중 CDU 전이는 **부하가 계단**이고
#: 누출은 t=0 이전부터 있다. 두 전이를 같은 자극으로 읽으면 안 된다.
STIMULUS_NONE: str = "none"
STIMULUS_LEAK_STEP: str = "leak_step"
STIMULUS_LOAD_STEP: str = "load_step"


@dataclass(frozen=True)
class ScenarioSpec:
    """시나리오 하나의 식별자. 이것만으로 결과가 결정되어야 한다.

    **상태 이월이 없다는 것은 이 명제가 참이라는 뜻이다** — 같은 spec 이면 배치
    안에서 몇 번째로 돌든, 혼자 돌든 같은 행이 나와야 한다(세션 5.5 게이트).
    """

    scenario_id: str
    regime: str  # "steady" · "transient"
    cdu_config: str
    template: CduCase
    load_percent: float
    leak_multiplier: float
    holdup_index: int | None = None  # transient 에만

    @property
    def leak_level_percent(self) -> float:
        return (self.leak_multiplier - 1.0) * 100.0

    @property
    def stimulus_kind(self) -> str:
        """이 행을 움직인 자극 [세션 5.7]. **기존 케이스 정의에서 읽는다.**

        새 시나리오를 만들지 않는다 — `regime` 과 `cdu_config` 가 이미 결정하고
        있던 것에 이름을 붙이는 것뿐이다(`transient_rows` 참조)::

            steady                      → none        (자극 없음)
            transient · single          → leak_step   (부하 불변 · 누출이 계단)
            transient · dual_asymmetric → load_step   (누출은 t=0 이전 · 부하가 계단)

        정상상태 행도 빈 칸이 아니라 `none` 을 싣는다 — 전 행 일관이 요건이고,
        빈 칸은 이 스키마에서 「해당 없음」이 아니라 「이 행에 값이 없다」로 이미
        쓰이고 있다(`t_s`·`holdup_mass_kg`).
        """
        if self.regime == "steady":
            return STIMULUS_NONE
        if self.cdu_config == CONFIG_SINGLE:
            return STIMULUS_LEAK_STEP
        return STIMULUS_LOAD_STEP

    @property
    def leak_cdu_index(self) -> int | str:
        """누출이 걸린 CDU 번호. **누출이 없으면 빈 값**이다 [세션 5.5-D].

        빈 값을 쓰는 이유: 이 스키마에서 「이 행에 해당 없음」은 이미 빈 칸으로
        표기한다(`t_s`·`holdup_mass_kg`·`hx_duty_kW`·랙별 열). -1 같은 표지값을
        새로 만들면 규약이 둘이 된다.

        **주의 — `leak_rack_index` 와 다르게 동작한다.** `leak_rack_index` 는
        세션 5.5-B 부터 누출 유무와 무관하게 항상 0 을 싣는다("누출이 있다면
        걸릴 자리"). 그 동작은 이 판의 범위 밖이라 바꾸지 않았다. 누출 유무의
        판정에는 `leak_level_percent == 0` 또는 이 열의 빈 칸을 쓴다.
        """
        if self.leak_multiplier == 1.0:
            return ""
        return LEAK_CDU_INDEX

    @property
    def scenario_kind(self) -> str:
        """5장 시나리오 구분. 이 파일럿에서 「이상」은 부하 극단으로 본다."""
        if self.leak_multiplier != 1.0:
            return "누출"
        if self.load_percent == LOAD_PROFILE.idle_load_percent:
            return "이상"  # 유휴 — 정격에서 벗어난 운전
        return "정상"


def _range_axis_values(template: CduCase) -> dict[str, float]:
    """케이스 라벨이 아니라 **실제 5장 범위 축 값**을 뽑는다."""
    label = template.hydraulic.label  # 예: "H22.4/dPb2/dPv3"
    head, branch, valve = label.split("/")
    return {
        "pump_head_rated_mAq": 20.0 if head.startswith("H22") else 30.0,
        "branch_dp_rated_mAq": float(branch.removeprefix("dPb")),
        "valve_dp_rated_mAq": float(valve.removeprefix("dPv")),
        "ntu": template.ntu,
        "T_secondary_supply_C": template.T_secondary_supply_C,
    }


def enumerate_specs() -> list[ScenarioSpec]:
    """전체 시나리오 목록 — 사람이 정한 규모 그대로.

    정상상태: 32조합 × 부하 2 × 누출 4 × 구성 3 = 768
    전이:     32조합 × 부하 2 × 누출 4 × 구성 2 × M 2 = 1,024
    """
    templates = default_cdu_cases()  # 32조합 (부하는 아래에서 덮어쓴다)
    loads = (LOAD_PROFILE.idle_load_percent, LOAD_PROFILE.rated_load_percent)
    multipliers = [level.k_multiplier for level in leak_levels()]

    specs: list[ScenarioSpec] = []
    for template in templates:
        for load in loads:
            for multiplier in multipliers:
                for config in (
                    CONFIG_SINGLE,
                    CONFIG_DUAL_SYMMETRIC,
                    CONFIG_DUAL_ASYMMETRIC,
                ):
                    key = (
                        f"{template.hydraulic.label}|NTU{template.ntu:g}"
                        f"|T2nd{template.T_secondary_supply_C:g}"
                        f"|L{load:g}|K{multiplier:g}|{config}"
                    )
                    specs.append(
                        ScenarioSpec(
                            scenario_id=f"steady|{key}",
                            regime="steady",
                            cdu_config=config,
                            template=template,
                            load_percent=load,
                            leak_multiplier=multiplier,
                        )
                    )
                    if config not in TRANSIENT_CONFIGS:
                        continue
                    for holdup_index in range(len(holdup_bounds())):
                        specs.append(
                            ScenarioSpec(
                                scenario_id=f"transient|{key}|M{holdup_index}",
                                regime="transient",
                                cdu_config=config,
                                template=template,
                                load_percent=load,
                                leak_multiplier=multiplier,
                                holdup_index=holdup_index,
                            )
                        )
    return specs


def _load_percents(spec: ScenarioSpec) -> tuple[float, ...]:
    """CDU 별 부하율.

    비대칭은 CDU A 가 `load_percent`, B 가 **다른 쪽 수준**을 받는다. 부하 축이
    2수준뿐이라 (100,20) 과 (20,100) 이 서로 거울상이 된다 — 정보로는 중복이지만
    **사람이 정한 규모를 코드가 줄이지 않는다**(세션 5.5-B 보고에 적었다).
    """
    idle = LOAD_PROFILE.idle_load_percent
    rated = LOAD_PROFILE.rated_load_percent
    if spec.cdu_config == CONFIG_SINGLE:
        return (spec.load_percent,)
    if spec.cdu_config == CONFIG_DUAL_SYMMETRIC:
        return (spec.load_percent,) * PLANT.cdu_count
    other = idle if spec.load_percent == rated else rated
    return (spec.load_percent, other)


def _apply_leak(case: CduCase, spec: ScenarioSpec) -> CduCase:
    """누출을 랙 0 에 건다. 배율 1.0(정상)도 **같은 경로**를 통과한다(세션 4 C2)."""
    level = next(
        lv for lv in leak_levels() if lv.k_multiplier == spec.leak_multiplier
    )
    return leak_case(case, level, LEAK.injection_rack_index)


def _blank_rack_columns() -> dict[str, float | str]:
    return {
        **{f"rack{i}_flow_Lps": "" for i in range(SCENARIO.racks_per_cdu)},
        **{f"rack{i}_outlet_C": "" for i in range(SCENARIO.racks_per_cdu)},
    }


PROVENANCE_ROW: dict[str, str] = {name: text for name, text in PROVENANCE_COLUMNS}


def _base_row(spec: ScenarioSpec, cdu_index: int) -> dict[str, object]:
    holdup_mass_kg = (
        holdup_bounds()[spec.holdup_index].mass_kg
        if spec.holdup_index is not None
        else ""
    )
    return {
        "dataset_version": DATASET_VERSION,
        "scenario_id": spec.scenario_id,
        "scenario_kind": spec.scenario_kind,
        "regime": spec.regime,
        "stimulus_kind": spec.stimulus_kind,
        "cdu_config": spec.cdu_config,
        "cdu_index": cdu_index,
        "leak_level_percent": spec.leak_level_percent,
        "leak_rack_index": LEAK.injection_rack_index,
        "leak_cdu_index": spec.leak_cdu_index,
        "leak_model": LEAK_MODEL_K_APPROX,
        "load_percent": _load_percents(spec)[cdu_index],
        "holdup_mass_kg": holdup_mass_kg,
        "t_s": "",
        **_range_axis_values(spec.template),
        **_blank_rack_columns(),
        **PROVENANCE_ROW,
    }


def steady_rows(spec: ScenarioSpec) -> list[dict[str, object]]:
    """정상상태 시나리오 하나의 행들 (CDU 마다 한 행)."""
    loads = _load_percents(spec)
    rows: list[dict[str, object]] = []

    if spec.cdu_config == CONFIG_SINGLE:
        case = _apply_leak(replace(spec.template, load_percent=loads[0]), spec)
        result = solve_cdu_steady_state(case)
        results = [result]
        shares = [HEAT_EXCHANGER.secondary_flow_Lps]
        plant_ier: object = ""
    else:
        cdus = tuple(
            _apply_leak(replace(spec.template, load_percent=load), spec)
            if index == LEAK_CDU_INDEX
            else replace(spec.template, load_percent=load)
            for index, load in enumerate(loads)
        )
        plant = solve_plant_steady_state(PlantCase(label=spec.scenario_id, cdus=cdus))
        results = list(plant.cdu_results)
        shares = list(plant.secondary_shares_Lps)
        plant_ier = plant.top_level_solver_ier

    for cdu_index, (result, share) in enumerate(zip(results, shares, strict=True)):
        row = _base_row(spec, cdu_index)
        for rack_index, (flow, temp) in enumerate(
            zip(
                result.flow.rack_flows_Lps,
                result.thermal.rack_return_temps_C,
                strict=True,
            )
        ):
            row[f"rack{rack_index}_flow_Lps"] = flow
            row[f"rack{rack_index}_outlet_C"] = temp
        row.update(
            total_flow_Lps=result.flow.total_flow_Lps,
            pump_head_mAq=result.flow.pump_head_mAq,
            T_supply_C=result.thermal.T_supply_C,
            T_return_C=result.thermal.T_return_C,
            hx_duty_kW=result.thermal.hx_duty_kW,
            secondary_share_Lps=share,
            heat_capacity_ratio=result.thermal.hx_effectiveness,
            energy_balance_residual_percent=energy_balance_residual_percent(
                result.thermal
            ),
            hydraulic_solver_ier=result.flow.solver_ier,
            thermal_solver_converged=result.thermal.solver_converged,
            plant_solver_ier=plant_ier,
            integrator_success="",
        )
        rows.append(row)
    return rows


def transient_rows(spec: ScenarioSpec) -> list[dict[str, object]]:
    """전이 시나리오 하나의 행들 (CDU × 시각).

    **자극이 구성마다 다르다**(모듈 docstring 참조). 단일 CDU 는 **누출 스텝** —
    정격 운전 중 t=0 에 K값이 계단으로 오른다. 다중 CDU 는 **부하 스텝**이고
    누출은 t=0 이전부터 CDU `LEAK_CDU_INDEX` 하나에 걸려 있다.
    배율 1.0(정상)도 같은 경로를 돌아 "누출이 없는" 기준선이 된다.

    **전이 행에는 랙별 값을 담지 않는다** — 2노드 온도 모델이 시간에 따라 내는
    것은 공급·환수 헤더 온도이고, 랙별 분해는 정상상태 대수식에서만 나온다.
    빈 칸으로 두어 `regime` 으로 구분되게 한다(스키마 조정 사항으로 보고했다).
    """
    assert spec.holdup_index is not None
    holdup = holdup_bounds()[spec.holdup_index]
    loads = _load_percents(spec)

    if spec.cdu_config == CONFIG_SINGLE:
        single = integrate_leak_step(
            LeakStepCase(
                label=spec.scenario_id,
                holdup=holdup,
                hydraulic=spec.template.hydraulic,
                k_multiplier=spec.leak_multiplier,
                T_secondary_supply_C=spec.template.T_secondary_supply_C,
                ntu=spec.template.ntu,
                load_percent=loads[0],
            )
        )
        return _transient_rows_from(
            spec=spec,
            times=single.t_s,
            supply_series=(single.T_supply_C,),
            return_series=(single.T_return_C,),
            flow_pairs=(
                (single.total_flow_initial_Lps, single.total_flow_final_Lps),
            ),
            head_pairs=(
                (single.pump_head_initial_mAq, single.pump_head_final_mAq),
            ),
            shares=(HEAT_EXCHANGER.secondary_flow_Lps,),
            integrator_ok=single.solver_success,
            hydraulic_ok=single.hydraulic_solver_converged,
            plant_ier="",
        )

    # **자극이 필요하다.** `load_before == load_after` 로 두면 계가 정상상태에서
    # 출발해 아무것도 하지 않아 「전이」 행이 평평한 직선이 된다 — 그런 행으로
    # 학습하면 누출이 전이를 만들지 않는다고 배운다. 그래서 CDU A 에 **부하 스텝**을
    # 준다(다른 쪽 수준 → `spec.load_percent`), CDU B 는 그대로 둔다.
    #
    # 누출은 **t=0 이전부터 있는 조건**이고 **CDU `LEAK_CDU_INDEX` 하나에만** 건다
    # [세션 5.5-D · 미해결 #34 닫음]. 세션 5.5-B 는 `PlantLoadStepCase` 가 CDU 마다
    # 다른 템플릿을 받지 못해 **두 CDU 의 랙 0 에 동시에** 걸고 있었고, 그것은
    # 5-1 「누출 주입 지점」(랙 1개)의 규정 밖이었으며 양쪽 대칭이라 2차측 배분이
    # 움직이지 않아 **CDU 간 연동이 상쇄**됐다.
    #
    # **자극 형태가 단일 CDU 전이와 다르다는 사실은 그대로 남는다** — 단일은 누출이
    # **스텝으로** 들어오고(부하 불변), 다중은 **부하가 스텝**이고 누출은 t=0
    # 이전부터 있다. `cdu_config` 로 구분해 읽어야 하며 두 전이를 같은 자극으로 보면
    # 안 된다. 다중 전이도 누출을 스텝으로 넣을지는 **사람이 정한다**(세션 5.5-D
    # C4 의견 — 이 판에서 바꾸지 않았다).
    other = (
        LOAD_PROFILE.idle_load_percent
        if spec.load_percent == LOAD_PROFILE.rated_load_percent
        else LOAD_PROFILE.rated_load_percent
    )
    dual = integrate_plant_load_step(
        PlantLoadStepCase(
            label=spec.scenario_id,
            holdup=holdup,
            template=spec.template,
            templates=tuple(
                _apply_leak(spec.template, spec) if index == LEAK_CDU_INDEX
                else spec.template
                for index in range(len(loads))
            ),
            load_before_percents=(other,) + tuple(loads[1:]),
            load_after_percents=loads,
        )
    )
    # 저장 격자를 `plant.py` 가 직접 `storage_times_s` 로 잡는다(세션 5.5-D C3) —
    # 세션 5.5-B 가 여기서 하던 최근접 점 선택이 필요 없어졌다.
    return _transient_rows_from(
        spec=spec,
        times=dual.t_s,
        supply_series=dual.T_supply_C,
        return_series=dual.T_return_C,
        flow_pairs=((None, None),) * len(dual.T_supply_C),
        head_pairs=((None, None),) * len(dual.T_supply_C),
        shares=dual.secondary_shares_final_Lps,
        integrator_ok=dual.solver_success,
        hydraulic_ok=dual.hydraulic_solver_converged,
        plant_ier=1 if dual.hydraulic_solver_converged else 0,
    )


def _transient_rows_from(
    *,
    spec: ScenarioSpec,
    times: np.ndarray,
    supply_series: Sequence[np.ndarray],
    return_series: Sequence[np.ndarray],
    flow_pairs: Sequence[tuple[float | None, float | None]],
    head_pairs: Sequence[tuple[float | None, float | None]],
    shares: Sequence[float],
    integrator_ok: bool,
    hydraulic_ok: bool,
    plant_ier: object,
) -> list[dict[str, object]]:
    """전이 행을 만든다 — 단일/다중 두 경로가 **같은 코드**로 행을 낸다.

    두 경로가 행 생성을 각자 하면 열 채우기가 갈릴 수 있다(collaboration.md ③).
    """
    rows: list[dict[str, object]] = []
    for cdu_index in range(len(supply_series)):
        for sample_index, t_s in enumerate(times):
            edge = 0 if sample_index == 0 else 1
            flow = flow_pairs[cdu_index][edge]
            head = head_pairs[cdu_index][edge]
            row = _base_row(spec, cdu_index)
            row.update(
                t_s=float(t_s),
                total_flow_Lps="" if flow is None else flow,
                pump_head_mAq="" if head is None else head,
                T_supply_C=float(supply_series[cdu_index][sample_index]),
                T_return_C=float(return_series[cdu_index][sample_index]),
                hx_duty_kW="",
                secondary_share_Lps=shares[cdu_index],
                heat_capacity_ratio="",
                energy_balance_residual_percent="",
                hydraulic_solver_ier=1 if hydraulic_ok else 0,
                thermal_solver_converged=hydraulic_ok,
                plant_solver_ier=plant_ier,
                integrator_success=integrator_ok,
            )
            rows.append(row)
    return rows


def rows_for(spec: ScenarioSpec) -> list[dict[str, object]]:
    """시나리오 하나의 행들.

    **spec 하나가 결과를 완전히 결정한다** — 배치 문맥을 읽지 않는다. 그것이
    세션 5.5 게이트(상태 이월 없음)가 요구하는 성질이고, 이 함수가 그 계약이다.
    """
    return steady_rows(spec) if spec.regime == "steady" else transient_rows(spec)


def column_names() -> list[str]:
    """CSV 열 순서 — 첫 행을 만들어 그 키 순서를 쓴다."""
    template = default_cdu_cases()[0]
    probe = ScenarioSpec(
        scenario_id="probe",
        regime="steady",
        cdu_config=CONFIG_SINGLE,
        template=template,
        load_percent=LOAD_PROFILE.rated_load_percent,
        leak_multiplier=1.0,
    )
    return list(steady_rows(probe)[0].keys())


def generate(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    specs: list[ScenarioSpec] | None = None,
    progress_every: int = 100,
) -> dict[str, object]:
    """배치를 돌려 CSV 를 쓴다. 반환: 메타데이터(행 수·시간·실패 수 등).

    한 시나리오씩 써 내려간다 — 전량을 메모리에 담지 않는다.
    """
    specs = enumerate_specs() if specs is None else specs
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cdu_dataset.csv"
    fields = column_names()

    started = time.perf_counter()
    row_count = 0
    failure_rows = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, spec in enumerate(specs, start=1):
            for row in rows_for(spec):
                writer.writerow(row)
                row_count += 1
                if row.get("hydraulic_solver_ier") not in (1, "") or (
                    row.get("integrator_success") is False
                ):
                    failure_rows += 1
            if progress_every and index % progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  {index}/{len(specs)} 시나리오 · {row_count:,} 행 · "
                    f"{elapsed / 60:.1f}분 경과",
                    flush=True,
                )
    elapsed_s = time.perf_counter() - started
    return {
        "path": str(path),
        "scenarios": len(specs),
        "rows": row_count,
        "columns": len(fields),
        "failure_rows": failure_rows,
        "elapsed_s": elapsed_s,
        "bytes": path.stat().st_size,
        "assumption_tag": ASSUMPTION_TAG,
    }


def main() -> int:
    import sys

    if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔 기본 인코딩 대비
        sys.stdout.reconfigure(encoding="utf-8")
    specs = enumerate_specs()
    steady = sum(1 for s in specs if s.regime == "steady")
    print(f"시나리오 {len(specs):,}개 (정상상태 {steady:,} · 전이 {len(specs)-steady:,})")
    meta = generate()
    print()
    for key, value in meta.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
