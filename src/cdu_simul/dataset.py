"""시나리오 배치 실행 → 텍스트 데이터셋 (세션 5.5-B · 파일럿 마지막 판).

**AI 학습 데이터셋으로 쓸 산출물이다.** 그래서 전 행에 출처와 한계를 붙인다
(`CLAUDE.md` 절대 규칙 11 · 세션 5.5 게이트) — 이 데이터로 학습한 모델의 한계를
아는 유일한 단서가 될 수 있기 때문이다.

**규모·표기·부하 수준·막힘 랙은 사람이 정했다**(세션 5.5-B 「사람이 정한 것」).
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
같은 부하 스텝 안에서 `blockage_level_percent` 를 가로질러 비교한다.
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
from cdu_simul.dataset_plan import MASSLOSS_PROVENANCE, PROVENANCE_COLUMNS
from cdu_simul.dynamics import (
    LeakStepCase,
    holdup_bounds,
    integrate_leak_step,
)
from cdu_simul.hydraulics import rated_property_temperature_C
from cdu_simul.leak import leak_case, leak_levels
from cdu_simul.massloss import (
    SWEEP_FRACTIONS,
    MassLossTopology,
    massloss_topologies,
)
from cdu_simul.massloss_thermal import (
    massloss_sizes_Lps,
    solve_massloss_steady,
    solve_plant_massloss,
)
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

#: 「막힘」을 거는 CDU 번호 [시나리오 정의: 5-1 「「막힘」 주입 지점」의 적용 ·
#: 세션 5.5-D].
#:
#: 5-1 은 "**랙 1개**에 주입한다 · 전 랙 동시 「막힘」은 5장에 근거가 없어 돌리지
#: 않는다" 라고만 적는다. CDU 가 2대인 플랜트에서 그 규정을 지키려면 주입점이
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
#: (48열본 = 세션 5.5-B · 49열본 = 세션 5.5-D · 52열본 = 세션 5.7 ·
#: **58열본 = 세션 7.39** — 「샘」 행이 처음 들어온 판본이다).
#: 파일명은 바꾸지 않는다(`cdu_dataset.csv`) — 판본은 파일 안에서 읽는다.
DATASET_VERSION: str = "session-7.39"

#: 누출을 무엇으로 모사했는가 [사람 결정 · 세션 5.7]. **전 행 상수가 아니다**
#: — 세션 7.39 가 「샘」 행을 얹으면서 값이 둘이 됐다(아래 `LEAK_MODEL_MASSLOSS`).
#:
#: 이 데이터셋의 이상 상태는 전부 **「막힘」(배관 K값 증가)** 이다(절대 규칙 8 ·
#: 5장 「누출 시나리오(「막힘」)」). 「샘」(질량손실)은 `massloss.py`·
#: `massloss_thermal.py` 에만 있고 데이터셋 생성 경로에 들어가지 않는다(세션 5.6).
#: 두 기구는 총유량·주입 랙 유량의 **부호가 정반대**이므로(세션 5.6 · 미해결 #36)
#: 이 열이 없으면 나중에 어느 쪽 데이터인지 알 수 없다. **열 이름과 값
#: (`leak_model` · "K_approx")은 이미 나간 것이라 바꾸지 않는다**(세션 7.32 ·
#: 대응표 `docs/leak-naming-map.md`).
LEAK_MODEL_K_APPROX: str = "K_approx"

#: 「샘」(질량손실) 행의 `leak_model` 값 [세션 7.39]. **라벨 문자열 하나이고 물리
#: 가정이 아니다** — 값을 `massloss` 로 적는 것은 대응표 규칙 1(「샘」 전용 이름에서
#: `leak` 를 빼고 정의어 `massloss` 만 남긴다 · `docs/leak-naming-map.md`)이다.
#: **「막힘」 쪽 값 `K_approx` 는 이미 나간 것이라 바꾸지 않는다**(세션 7.32).
LEAK_MODEL_MASSLOSS: str = "massloss"

#: 「샘」 크기 스윕 지점 — **비영 4수준** [세션 7.25 C4 가 규모 B 를 센 축].
#: `massloss.SWEEP_FRACTIONS` 에서 0 을 뺀 것이다. 0 은 누출이 없는 해라
#: 「정상」 행과 같은 것을 「이상」 라벨로 한 번 더 싣게 된다.
#: **새 값이 아니다** — 크기는 5-1 「「샘」(질량손실) 크기 수준」의 역산 규칙으로
#: 케이스마다 다시 나온다(`massloss_sizes_Lps`).
MASSLOSS_SIZE_FRACTIONS: tuple[float, ...] = tuple(
    fraction for fraction in SWEEP_FRACTIONS if fraction > 0.0
)

#: 전이 행의 자극 종류 — `ScenarioSpec.stimulus_kind` 가 이 셋 중 하나를 낸다.
#:
#: **새 시나리오를 만들지 않는다.** 이미 있는 행이 무엇으로 자극됐는지를 라벨로
#: 적을 뿐이다 [사람이 정한 것 — 세션 5.7 ⑵]. 단일 CDU 전이는 정격 운전 중
#: **누출이 계단으로** 들어오고(부하 불변), 다중 CDU 전이는 **부하가 계단**이고
#: 누출은 t=0 이전부터 있다. 두 전이를 같은 자극으로 읽으면 안 된다.
#: **네 번째 값이 세션 7.39 에서 붙었다** — 「샘」 단일 CDU 전이는 「샘」이 계단으로
#: 들어오므로 「막힘」 계단(`leak_step`)과 같은 자극이 아니다. `leak_step` 은 이미
#: 나간 값이라 이름을 바꾸지 않고, 새 값만 대응표 규칙 1 대로 `massloss` 로 적는다.
#: **이 판의 데이터셋에는 이 값을 갖는 행이 아직 없다** — 규모 B 는 정상상태만이고
#: 「샘」 전이는 미해결 #40 이 열린 채라 붙이지 않았다.
STIMULUS_NONE: str = "none"
STIMULUS_LEAK_STEP: str = "leak_step"
STIMULUS_LOAD_STEP: str = "load_step"
STIMULUS_MASSLOSS_STEP: str = "massloss_step"

#: 이상 기구 — 절대 규칙 8 의 둘. `ScenarioSpec.mechanism` 이 이 셋 중 하나다.
#:
#: **라벨 축을 K 배수에서 떼어내려고 둔다** [사람 결정 · 세션 7.35]. 이전에는
#: `scenario_kind` 가 `leak_multiplier != 1.0` 에서 파생돼, K 배수를 쓰지 않는
#: 「샘」(질량손실) 행이 K=1.0 이라는 이유로 「정상」으로 실렸다(세션 7.34 C1).
#: 기구는 이제 spec 이 직접 들고, 라벨은 여기서만 읽는다.
#:
#: **CSV 열이 아니다** — 기구를 행에 싣는 열은 `leak_model` 이다(세션 7.35 결정:
#: 한 값을 두 열에 두지 않는다). 세션 7.39 가 「샘」 행을 실제로 얹었다.
MECHANISM_NONE: str = "none"
MECHANISM_BLOCKAGE: str = "blockage"  # 「막힘」 — 배관 K값 증가
MECHANISM_MASSLOSS: str = "massloss"  # 「샘」 — 질량손실(계통 밖 유출)


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
    #: 이상 기구 [세션 7.35]. **K 배수와 독립된 축이다** — 「샘」은 K=1.0 이다.
    mechanism: str = MECHANISM_NONE
    #: 「샘」 크기 스윕 지점 [세션 7.39]. 「샘」 행에만 있다.
    massloss_size_fraction: float | None = None
    #: 「샘」의 계통 배치 — 5장·5-1 이 정하지 않는 구조 자유도 둘(g · 펌프가 보는
    #: 유량)을 담는다. **전수로 돈다 — 하나도 고정하지 않는다**(고정하면 5-1 에
    #: 새 값이 든다 · 절대 규칙 1).
    massloss_topology: MassLossTopology | None = None
    #: 셋째 구조 자유도 — 공유 2차측 배분을 환수유량으로 읽는가. 다중 CDU 에만
    #: 있다(`solve_plant_massloss`).
    share_uses_return_flow: bool | None = None

    @property
    def blockage_level_percent(self) -> float | str:
        """「막힘」의 K값 증가율 [%]. **「샘」 행은 빈 값**이다 [세션 7.39].

        「샘」은 K 배수를 쓰지 않으므로 이 열에 실을 값이 **없다** — 0 을 실으면
        「증가율 0 = 정상」으로 읽힌다(세션 7.34 C1 ⑶). 빈 값은 이 스키마에서
        「이 행에 해당 없음」 표기다(`anomaly_cdu_index` 참조).
        """
        if self.mechanism == MECHANISM_MASSLOSS:
            return ""
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
            # 단일 CDU 전이는 이상 기구가 계단으로 들어온다 — 어느 기구가
            # 들어왔는지로 갈린다 [세션 7.39].
            if self.mechanism == MECHANISM_MASSLOSS:
                return STIMULUS_MASSLOSS_STEP
            return STIMULUS_LEAK_STEP
        return STIMULUS_LOAD_STEP

    @property
    def anomaly_cdu_index(self) -> int | str:
        """누출이 걸린 CDU 번호. **누출이 없으면 빈 값**이다 [세션 5.5-D].

        빈 값을 쓰는 이유: 이 스키마에서 「이 행에 해당 없음」은 이미 빈 칸으로
        표기한다(`t_s`·`holdup_mass_kg`·`hx_duty_kW`·랙별 열). -1 같은 표지값을
        새로 만들면 규약이 둘이 된다.

        **K 배수가 아니라 기구로 판정한다** [세션 7.39]. 「샘」 행은 K=1.0 이지만
        이상 기구가 있고 `solve_plant_massloss` 가 「샘」을 CDU
        `LEAK_CDU_INDEX` 하나에만 걸므로(5-1 「「샘」 주입 지점」·#34) 그 번호를
        싣는다. 종전 판정(`leak_multiplier == 1.0`)은 「막힘」 행에서는 이것과
        같은 결과였다.

        누출 유무의 판정에는 이 열의 빈 칸을 쓴다 — `blockage_level_percent` 는
        「샘」 행에서 비어 있고 「막힘」 유무만 가른다.
        """
        if self.mechanism == MECHANISM_NONE:
            return ""
        return LEAK_CDU_INDEX

    @property
    def scenario_kind(self) -> str:
        """정상 / 이상 — **기구 유무 하나로만 갈린다** [사람 결정 · 세션 7.35].

        절대 규칙 8 이 「이상 상태는 「막힘」과 「샘」 둘이고 이 둘이 이상
        시나리오의 전부」라고 못박았으므로, 이상 여부는 `mechanism` 이 결정한다.
        **어느 기구인지는 이 열이 갖지 않는다** — 기구 열(`leak_model`)이 이미
        싣고 있고, 한 값을 두 열에 두면 언젠가 갈린다.

        **바뀐 것 둘** (이전 값 셋은 「정상 / 이상 / 누출」이었다):

        · 「누출」 값을 뺐다. 「누출」은 상위 개념 전용이고 모델 안의 기구는
          「막힘」·「샘」으로 적는다(절대 규칙 8 낱말 규약 · 세션 7.27).
        · **유휴 부하를 「이상」에서 뺐다.** 이전 판까지 「이 파일럿에서
          「이상」은 부하 극단으로 본다」로 유휴 20% 를 「이상」으로 실었으나,
          5장 표는 「유휴 20% ~ 정격 100%」를 **부하 프로파일** 행으로 주고
          이상 시나리오는 K값 행에서 따로 준다 — 정본에 근거가 없었다(세션
          7.35 확인). 운전점은 `load_percent` 열이 그대로 싣는다.
        """
        return "정상" if self.mechanism == MECHANISM_NONE else "이상"


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

    「막힘」(K값 증가) — 세션 5.5-B 규모::

        정상상태: 32조합 × 부하 2 × 누출 4 × 구성 3 = 768
        전이:     32조합 × 부하 2 × 누출 4 × 구성 2 × M 2 = 1,024

    「샘」(질량손실) — 세션 7.39 가 얹었다. **정상상태만**이다(전이는 미해결 #40
    이 열린 채라 붙이지 않았다 · 규모 B)::

        단일:   32조합 × 부하 2 × 크기 4 × 배치 6                = 1,536
        다중:   32조합 × 부하 2 × 크기 4 × 배치 6 × 구성 2 × 배분읽기 2 = 6,144

    **「샘」 spec 을 뒤에 붙인다** — 앞의 1,792개 `scenario_id` 가 그대로 남는다
    (48·52열본이 그 키로 나갔다 · 세션 7.34 C3 ⑴).
    """
    templates = default_cdu_cases()  # 32조합 (부하는 아래에서 덮어쓴다)
    loads = (LOAD_PROFILE.idle_load_percent, LOAD_PROFILE.rated_load_percent)
    multipliers = [level.k_multiplier for level in leak_levels()]

    specs: list[ScenarioSpec] = []
    for template in templates:
        for load in loads:
            for multiplier in multipliers:
                # 이 데이터셋의 이상 기구는 전부 「막힘」이다(`LEAK_MODEL_K_APPROX`).
                # 막힘은 K값 증가로 정의되므로 여기서는 배수로 갈리는 것이 맞다 —
                # 「샘」은 K=1.0 이라 이 자리가 아니라 자기 축으로 들어온다.
                mechanism = (
                    MECHANISM_NONE if multiplier == 1.0 else MECHANISM_BLOCKAGE
                )
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
                            mechanism=mechanism,
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
                                mechanism=mechanism,
                            )
                        )

    specs.extend(_massloss_specs(templates, loads))
    return specs


def _massloss_specs(
    templates: list[CduCase], loads: tuple[float, ...]
) -> list[ScenarioSpec]:
    """「샘」 정상상태 시나리오 [세션 7.39].

    **케이스 축은 「막힘」과 같은 것을 쓴다** — 같은 32조합 × 부하 2 다(세션 7.34
    C4: `massloss_thermal.thermal_cases()` 와 값으로는 이미 같고 다른 것은 순서와
    표현뿐이다). 그래서 관측 판의 케이스 순서를 건드리지 않는다.

    **구조 자유도 셋을 전수로 돈다** — g 3 × 펌프가 보는 유량 2 = 배치 6, 다중
    CDU 는 여기에 2차측 배분 읽기 2 가 더 곱해진다. 하나라도 고정하면 5-1 에 새
    값이 들어간다(절대 규칙 1 · 5-1 「「샘」 구조 자유도」 한계 ⑵ · 세션 7.26 결정).
    """
    specs: list[ScenarioSpec] = []
    for template in templates:
        for load in loads:
            for fraction in MASSLOSS_SIZE_FRACTIONS:
                for topology in massloss_topologies():
                    for config in (
                        CONFIG_SINGLE,
                        CONFIG_DUAL_SYMMETRIC,
                        CONFIG_DUAL_ASYMMETRIC,
                    ):
                        share_readings: tuple[bool | None, ...] = (
                            (None,) if config == CONFIG_SINGLE else (True, False)
                        )
                        for share_uses_return_flow in share_readings:
                            key = (
                                f"{template.hydraulic.label}|NTU{template.ntu:g}"
                                f"|T2nd{template.T_secondary_supply_C:g}"
                                f"|L{load:g}|Qfrac{fraction:g}"
                                f"|g{topology.residual_return_share:g}"
                                f"|pump"
                                f"{'sup' if topology.pump_sees_supply_flow else 'ret'}"
                                f"|{config}"
                            )
                            if share_uses_return_flow is not None:
                                key += (
                                    "|share"
                                    f"{'ret' if share_uses_return_flow else 'sup'}"
                                )
                            specs.append(
                                ScenarioSpec(
                                    scenario_id=f"steady|massloss|{key}",
                                    regime="steady",
                                    cdu_config=config,
                                    template=template,
                                    load_percent=load,
                                    leak_multiplier=1.0,
                                    mechanism=MECHANISM_MASSLOSS,
                                    massloss_size_fraction=fraction,
                                    massloss_topology=topology,
                                    share_uses_return_flow=share_uses_return_flow,
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

#: 기구별 표기 [세션 7.35]. **열은 늘지 않는다** — 같은 다섯 열에 다른 문언이
#: 실릴 뿐이다. 「샘」 문언은 셋만 갈리고 나머지 둘은 위 것을 그대로 쓴다.
_PROVENANCE_BY_MECHANISM: dict[str, dict[str, str]] = {
    MECHANISM_MASSLOSS: {**PROVENANCE_ROW, **MASSLOSS_PROVENANCE},
}


def _blank_if_none(value: object) -> object:
    """`None` 을 스키마의 「해당 없음」 표기(빈 값)로 바꾼다."""
    return "" if value is None else value


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
        "blockage_level_percent": spec.blockage_level_percent,
        # 「샘」 행은 **빈 값**이다 [세션 7.39]. 5-1 「「샘」 주입 지점」이 「랙
        # 하나에 걸지만 계통은 그것을 국소로 보지 않는다」를 항목의 핵심으로
        # 못박았으므로, 랙 번호를 실으면 「랙0 에서 샌다」로 오독된다(랙 간 비대칭
        # 2.220e-16 L/s · 세션 5.6). 「막힘」 행은 종전대로 주입 랙을 싣는다.
        "anomaly_rack_index": (
            "" if spec.mechanism == MECHANISM_MASSLOSS else LEAK.injection_rack_index
        ),
        "anomaly_cdu_index": spec.anomaly_cdu_index,
        "leak_model": (
            LEAK_MODEL_MASSLOSS
            if spec.mechanism == MECHANISM_MASSLOSS
            else LEAK_MODEL_K_APPROX
        ),
        # 「샘」 축 넷 — 「막힘」·정상 행에서는 전부 빈 값이다(그 기구에 이 자유도가
        # 없다 · 세션 7.34 C2 표 ⓑ).
        "massloss_size_fraction": _blank_if_none(spec.massloss_size_fraction),
        "residual_return_share": (
            "" if spec.massloss_topology is None
            else spec.massloss_topology.residual_return_share
        ),
        "pump_sees_supply_flow": (
            "" if spec.massloss_topology is None
            else spec.massloss_topology.pump_sees_supply_flow
        ),
        "share_uses_return_flow": _blank_if_none(spec.share_uses_return_flow),
        "load_percent": _load_percents(spec)[cdu_index],
        "holdup_mass_kg": holdup_mass_kg,
        "t_s": "",
        **_range_axis_values(spec.template),
        **_blank_rack_columns(),
        **_PROVENANCE_BY_MECHANISM.get(spec.mechanism, PROVENANCE_ROW),
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
            # 「막힘」·정상은 밀폐루프라 **공급 = 환수**다. 「샘」에서만 갈린다
            # (5-1 「「샘」 크기 수준」 한계 ⑶ · 세션 5.6-B) — 그래서 이 열이
            # 신설됐다 [세션 7.39].
            return_flow_Lps=result.flow.total_flow_Lps,
            massloss_flow_Lps="",
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
    # 5-1 「「막힘」 주입 지점」(랙 1개)의 규정 밖이었으며 양쪽 대칭이라 2차측 배분이
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
                # 전이 행은 「막힘」뿐이라 공급 = 환수다 [세션 7.39].
                return_flow_Lps="" if flow is None else flow,
                massloss_flow_Lps="",
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


def massloss_steady_rows(spec: ScenarioSpec) -> list[dict[str, object]]:
    """「샘」(질량손실) 정상상태 시나리오 하나의 행들 [세션 7.39].

    「막힘」 경로와 **다른 solver 를 탄다** — 「샘」은 K값을 바꾸지 않고 계통 밖으로
    질량이 나가는 것이라, 헤더 압력평형 식 자체가 다르다(`massloss.solve_massloss`).
    그래서 `steady_rows` 를 재사용하지 않는다.

    **크기는 케이스마다 다시 역산한다** — 5-1 「「샘」(질량손실) 크기 수준」이
    「막힘」 +50% 해의 막힘 랙 유량 감소량을 상한으로 두라고 정한다. 코드에 숫자를
    박지 않는다(절대 규칙 1·2).

    **energy balance 열은 비운다.** 6장 1-B 기준은 밀폐루프 잔차를 재는 것이고,
    「샘」 행은 누출 엔탈피 항 없이는 닫히지 않는다(5-1 「「샘」 크기 수준」 한계 ⑷ ·
    세션 5.7-D: 빼면 최대 1.2129% · 넣으면 0.000062%). **이 판은 그 항을 넣지
    않는다** — 넣는 것은 스키마·게이트 결정이 따라붙는 별건이다. 대신 이 행을
    balance 판정 대상에서 빼고, 그 사실을 `signal_sign_caveat` 이 행마다 싣는다.
    """
    assert spec.massloss_topology is not None
    assert spec.massloss_size_fraction is not None
    topology = spec.massloss_topology
    loads = _load_percents(spec)
    cases = tuple(replace(spec.template, load_percent=load) for load in loads)
    size_Lps = massloss_sizes_Lps(
        spec.template.hydraulic, rated_property_temperature_C()
    )[SWEEP_FRACTIONS.index(spec.massloss_size_fraction)]

    if spec.cdu_config == CONFIG_SINGLE:
        results = [solve_massloss_steady(cases[0], size_Lps, topology)]
        shares = [HEAT_EXCHANGER.secondary_flow_Lps]
        plant_ier: object = ""
    else:
        assert spec.share_uses_return_flow is not None
        plant = solve_plant_massloss(
            cases, size_Lps, topology, spec.share_uses_return_flow
        )
        results = list(plant.cdu_results)
        shares = list(plant.secondary_shares_Lps)
        plant_ier = plant.top_level_solver_ier

    rows: list[dict[str, object]] = []
    for cdu_index, (result, share) in enumerate(zip(results, shares, strict=True)):
        row = _base_row(spec, cdu_index)
        for rack_index, (flow, temp) in enumerate(
            zip(result.rack_flows_Lps, result.rack_outlet_temps_C, strict=True)
        ):
            row[f"rack{rack_index}_flow_Lps"] = flow
            row[f"rack{rack_index}_outlet_C"] = temp
        row.update(
            # 「샘」에서만 공급 ≠ 환수다. `total_flow_Lps` 는 「막힘」 행과 같은
            # 뜻(랙이 받는 헤더 공급유량)을 유지한다.
            total_flow_Lps=result.supply_flow_Lps,
            return_flow_Lps=result.return_flow_Lps,
            massloss_flow_Lps=result.massloss_flow_Lps,
            pump_head_mAq=result.pump_head_mAq,
            T_supply_C=result.T_supply_C,
            T_return_C=result.T_return_C,
            hx_duty_kW=result.hx_duty_kW,
            secondary_share_Lps=share,
            heat_capacity_ratio=result.hx_effectiveness,
            energy_balance_residual_percent="",
            hydraulic_solver_ier=1 if result.hydraulic_solver_converged else 0,
            thermal_solver_converged=result.outer_solver_ier == 1,
            plant_solver_ier=plant_ier,
            integrator_success="",
        )
        rows.append(row)
    return rows


def rows_for(spec: ScenarioSpec) -> list[dict[str, object]]:
    """시나리오 하나의 행들.

    **spec 하나가 결과를 완전히 결정한다** — 배치 문맥을 읽지 않는다. 그것이
    세션 5.5 게이트(상태 이월 없음)가 요구하는 성질이고, 이 함수가 그 계약이다.
    """
    if spec.mechanism == MECHANISM_MASSLOSS:
        return massloss_steady_rows(spec)
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
    """산출물 `results/cdu_dataset.csv` 를 **규모 B** 로 만든다 [사람 결정].

    규모 B = 정상 + 「막힘」 + 「샘」 · **정상상태만**(세션 7.25 C4 의 이름).
    전이를 빼는 이유는 미해결 **#40**(전이 파형이 노드 수에 지배되고 N≤64 에서
    수렴하지 않았다)이 열린 채이기 때문이고, 「샘」 전이는 그 위에 상수 M 근사
    유보(#38)까지 얹힌다. **`generate()` 자체는 전이를 그대로 만들 수 있다** —
    거르는 것은 이 산출물 하나다.
    """
    import sys

    if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔 기본 인코딩 대비
        sys.stdout.reconfigure(encoding="utf-8")
    specs = [s for s in enumerate_specs() if s.regime == "steady"]
    massloss = sum(1 for s in specs if s.mechanism == MECHANISM_MASSLOSS)
    print(
        f"규모 B — 정상상태 시나리오 {len(specs):,}개 "
        f"(「막힘」·정상 {len(specs) - massloss:,} · 「샘」 {massloss:,}) · "
        "전이는 제외 (미해결 #40)"
    )
    meta = generate(specs=specs)
    print()
    for key, value in meta.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
