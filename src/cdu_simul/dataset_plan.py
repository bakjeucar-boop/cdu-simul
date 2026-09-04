"""데이터셋 설계 보조 — 조합 계수 · 실행시간 측정 · 크기 추정 (세션 5.5-A).

**이 모듈은 데이터셋을 생성하지 않는다.** 세션 5.5-A 는 설계와 측정만 하고,
생성은 사람이 규모를 정한 뒤 세션 5.5-B 에서 한다. 여기 있는 축 정의와 계수
함수는 5.5-B 가 그대로 재사용한다 — 그때 축을 다시 세지 않게 하려는 것이다.

**여기 적힌 스키마는 제안이지 확정이 아니다.** 5-1 에도 `assumptions.py` 에도
기록하지 않는다(세션 5.5-A C1).

**축의 값을 새로 만들지 않는다.** 전부 5장·5-1 에서 온다. 5장이 범위로만 주고
어떤 값을 쓸지 정해지지 않은 축(부하)은 **후보를 나열만** 하고 고르지 않는다.

**모든 수치는 가정값 기반이며 실측이 아니다.**
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

from cdu_simul.assumptions import (
    ASSUMPTION_TAG,
    LEAK,
    LOAD_PROFILE,
    PLANT,
    SCENARIO,
)
from cdu_simul.dynamics import holdup_bounds
from cdu_simul.hydraulics import default_cases as default_hydraulic_cases
from cdu_simul.model import default_cdu_cases, solve_cdu_steady_state
from cdu_simul.plant import (
    default_plant_load_step_cases,
    integrate_plant_load_step,
    plant_case,
    solve_plant_steady_state,
)


# ─────────────────────────────────────────────────────────────────────────────
# C2. 조합 공간 — 축 정의
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Axis:
    """조합 공간의 축 하나.

    `note` 에는 **중복·무의미 조합**이나 축으로 둘 필요가 있는지에 대한 의견을
    적는다 — 계수 결과만 보고는 그 판단을 할 수 없기 때문이다.
    """

    name: str
    size: int
    source: str
    applies_to: str  # "정상상태" · "전이" · "둘 다"
    note: str = ""


def scenario_axes(load_levels: int) -> tuple[Axis, ...]:
    """조합 공간의 축 전부. `load_levels` 는 부하 축의 값 수(후보별로 바뀐다)."""
    return (
        Axis(
            name="5장 범위 조합 (양정 2 × 분기ΔP 2 × 밸브ΔP 2)",
            size=len(default_hydraulic_cases()),
            source="5장 펌프 양정 20~30 mAq · 배관 ΔP 2~3 mAq · 밸브 ΔP 3~5 mAq",
            applies_to="둘 다",
            note="방침 (B) — 범위 양 끝만 쓴다. 중점을 만들지 않는다",
        ),
        Axis(
            name="NTU",
            size=2,
            source="5장 열교환기 NTU 2~3",
            applies_to="둘 다",
        ),
        Axis(
            name="2차측 공급온도",
            size=2,
            source="5장 2차측 27~30℃ (고정 경계조건 · 절대 규칙 7)",
            applies_to="둘 다",
        ),
        Axis(
            name="부하",
            size=load_levels,
            source="5장 부하 프로파일 유휴 20% ~ 정격 100%",
            applies_to="둘 다",
            note=(
                "**어떤 값을 쓸지 아직 정해지지 않았다.** 5장은 범위만 준다 — "
                "중간값(예: 60%)을 쓰려면 5장에 없는 선택이 된다. 후보별 영향은 "
                "`load_axis_candidates()` 참조"
            ),
        ),
        Axis(
            name="누출 수준 (정상 포함)",
            size=len(LEAK.k_multiplier_levels),
            source="5장 누출 K값 +5/+20/+50% + 정상(배율 1.0)",
            applies_to="둘 다",
            note="정상이 배율 1.0 으로 같은 경로를 돈다(세션 4 C2)",
        ),
        Axis(
            name="누출 랙 번호",
            size=SCENARIO.racks_per_cdu,
            source="5-1 「「막힘」 주입 지점」 — 랙 1개",
            applies_to="둘 다",
            note=(
                "**축으로 둘 필요가 없다는 의견이다.** 8랙이 전부 동일하므로 "
                "랙 번호는 결과에 영향하지 않는다(5-1 이 명시). 축으로 두면 "
                "행 수가 8배가 되고 그 8배는 전부 열 이름만 다른 같은 물리다. "
                "**랙 0 고정을 권한다** — 대신 '누출 랙' 열은 남겨 두어야 "
                "5.5-B 이후 비대칭 랙이 생겼을 때 스키마가 안 바뀐다"
            ),
        ),
        Axis(
            name="CDU 구성",
            size=3,
            source="5-1 「CDU 대수」 2대 + 세션 4까지의 단일 CDU 경로",
            applies_to="둘 다",
            note=(
                "단일 / 2대 대칭 / 2대 비대칭. **단일 CDU 에 '비대칭'은 성립하지 "
                "않으므로** 이 축은 부하 축과 독립이 아니다 — 비대칭은 부하가 "
                "2수준 이상일 때만 의미가 있다. 세션 5-B 이후 2대 대칭은 단일 CDU 를 "
                "**정확히 재현**하므로(편차 0.000e+00 K) 둘 중 하나는 중복이다"
            ),
        ),
        Axis(
            name="계통 보유수량 M",
            size=len(holdup_bounds()),
            source="5-1 「계통 보유수량 M」 하한·상한",
            applies_to="전이",
            note=(
                "정상상태 해는 M 에 의존하지 않는다(세션 3-B·5 확인) — "
                "**정상상태 행에 M 축을 두면 전부 중복이다**"
            ),
        ),
    )


@dataclass(frozen=True)
class CombinationCount:
    """한 조합 공간의 계수 결과."""

    label: str
    steady_cases: int
    transient_cases: int
    axes: tuple[Axis, ...]

    @property
    def total_cases(self) -> int:
        return self.steady_cases + self.transient_cases


def count_combinations(
    load_levels: int,
    leak_rack_axis: bool = False,
    cdu_configurations: int = 3,
) -> CombinationCount:
    """정상상태·전이 케이스 수를 **따로** 센다.

    `leak_rack_axis=False` 가 기본이다 — 8랙 대칭이라 랙 번호가 결과에 영향하지
    않기 때문이다(위 `Axis.note` 참조). 켜면 8배가 된다.
    """
    axes = scenario_axes(load_levels)
    base = 1
    for axis in axes:
        if axis.applies_to == "전이":
            continue
        if axis.name.startswith("누출 랙"):
            base *= axis.size if leak_rack_axis else 1
        elif axis.name.startswith("CDU 구성"):
            base *= cdu_configurations
        else:
            base *= axis.size
    holdup_levels = len(holdup_bounds())
    return CombinationCount(
        label=f"부하 {load_levels}수준 · CDU 구성 {cdu_configurations}종"
        + (" · 누출 랙 축 켬" if leak_rack_axis else ""),
        steady_cases=base,
        transient_cases=base * holdup_levels,
        axes=axes,
    )


def load_axis_candidates() -> tuple[tuple[str, int, str], ...]:
    """부하 축 후보 — (설명, 값 수, 5장 근거).

    **고르지 않는다.** 5장이 범위(20~100%)만 주므로 어떤 값을 쓸지는 사람의
    결정이다. 중간값을 쓰면 5장에 없는 숫자를 만드는 것이므로, 그 사실을 후보
    설명에 적었다.
    """
    return (
        (
            "양 끝만 (20% · 100%)",
            2,
            "5장 부하 프로파일의 양 끝 그대로. **새 숫자 0개** — 방침 (B) 와 같다",
        ),
        (
            "양 끝 + 중점 (20 · 60 · 100%)",
            3,
            "60% 는 **5장에 없는 값**이다. 중간 거동을 보려면 필요하지만 "
            "5-1 빈칸 처리 순서를 태워 사람이 확정해야 한다",
        ),
        (
            "4수준 (20 · 40 · 60 · 80 · 100% 중 4개)",
            4,
            "중간값을 **셋** 만든다. AI 학습 데이터셋으로는 부하-신호 관계를 "
            "더 잘 보여주지만 5장 밖 숫자가 그만큼 늘어난다",
        ),
        (
            "5수준 (20 · 40 · 60 · 80 · 100%)",
            5,
            "위와 같은 성격. 등간격이라 만들기는 쉽지만 등간격 자체가 "
            "5장에 없는 선택이다",
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# C3. 실행시간 실측
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TimingSample:
    """한 종류의 케이스를 여러 번 돌린 시간 [s]."""

    label: str
    seconds: tuple[float, ...]

    @property
    def mean_s(self) -> float:
        return sum(self.seconds) / len(self.seconds)

    @property
    def min_s(self) -> float:
        return min(self.seconds)

    @property
    def max_s(self) -> float:
        return max(self.seconds)

    @property
    def spread_ratio(self) -> float:
        return self.max_s / self.min_s if self.min_s > 0 else float("inf")


def _time_each(label: str, runners: list[Callable[[], object]]) -> TimingSample:
    """각 케이스를 한 번씩 돌려 시간을 잰다 (순서대로, 반복 없음)."""
    seconds: list[float] = []
    for run in runners:
        start = time.perf_counter()
        run()
        seconds.append(time.perf_counter() - start)
    return TimingSample(label=label, seconds=tuple(seconds))


def measure_steady_single(sample_size: int = 5) -> TimingSample:
    """단일 CDU 정상상태 — 서로 다른 조합 `sample_size` 개."""
    cases = default_cdu_cases()[:sample_size]
    return _time_each(
        "정상상태 · 단일 CDU",
        [partial(solve_cdu_steady_state, case) for case in cases],
    )


def measure_steady_plant(sample_size: int = 5) -> TimingSample:
    """2대 결합 정상상태 — 서로 다른 조합 `sample_size` 개 (비대칭 부하)."""
    templates = default_cdu_cases()[:sample_size]
    loads = (LOAD_PROFILE.rated_load_percent, LOAD_PROFILE.idle_load_percent)
    return _time_each(
        "정상상태 · 2대 결합",
        [
            partial(solve_plant_steady_state, plant_case("계측", loads, template))
            for template in templates
        ],
    )


def measure_transient_single(sample_size: int = 3) -> TimingSample:
    """단일 CDU 전이 — 누출 스텝 `sample_size` 개."""
    from cdu_simul.dynamics import integrate_leak_step
    from cdu_simul.leak import default_leak_transient_cases

    cases = default_leak_transient_cases()[:sample_size]
    return _time_each(
        "전이 · 단일 CDU",
        [partial(integrate_leak_step, case) for case in cases],
    )


def measure_transient_plant(sample_size: int = 3) -> TimingSample:
    """2대 결합 전이 — 비대칭 부하 스텝 `sample_size` 개.

    기본 케이스가 2개뿐이라 모자라면 2차측 온도를 5장 양 끝으로 바꿔 채운다
    (새 숫자를 만들지 않는다).
    """
    cases = list(default_plant_load_step_cases())
    while len(cases) < sample_size:
        base = cases[len(cases) % 2]
        template = replace(
            base.template, T_secondary_supply_C=SCENARIO.T_secondary_supply_C.high
        )
        cases.append(replace(base, template=template, label=f"{base.label} · T2nd 상한"))
    return _time_each(
        "전이 · 2대 결합",
        [partial(integrate_plant_load_step, case) for case in cases[:sample_size]],
    )


def measure_repeat_effect(repeats: int = 3) -> TimingSample:
    """**같은 케이스**를 반복해 캐시·워밍업 영향을 본다.

    1회 측정으로 대표하지 않기 위한 것이다 — 시간이 회차마다 달라지면 그 사실이
    추정의 불확실성이 된다.
    """
    case = default_cdu_cases()[0]
    return _time_each(
        "반복 영향 · 같은 케이스",
        [partial(solve_cdu_steady_state, case) for _ in range(repeats)],
    )


# ─────────────────────────────────────────────────────────────────────────────
# C5. 산출물 크기 추정
# ─────────────────────────────────────────────────────────────────────────────
#: 스키마 초안의 열 수 (아래 `SCHEMA_DRAFT` 참조).
SCHEMA_COLUMN_COUNT: int = 46
#: 출처·한계 표기를 **행마다** 반복할 때의 추가 열 수와 바이트.
INLINE_PROVENANCE_COLUMNS: int = 5
INLINE_PROVENANCE_BYTES_PER_ROW: int = 210
#: 표기를 별도 메타 파일로 뺄 때 행마다 남는 것은 **참조 키 하나**뿐이다.
META_FILE_COLUMNS: int = 1
META_FILE_BYTES_PER_ROW: int = 12
#: 관측량 한 칸의 평균 바이트 (부호·소수 6자리·구분자 포함 추정).
BYTES_PER_NUMERIC_CELL: int = 14


def estimate_bytes(rows: int, inline_provenance: bool) -> int:
    """CSV 파일 크기 [byte] 추정 — 헤더는 무시할 수 있는 크기라 세지 않는다."""
    per_row = SCHEMA_COLUMN_COUNT * BYTES_PER_NUMERIC_CELL
    per_row += (
        INLINE_PROVENANCE_BYTES_PER_ROW
        if inline_provenance
        else META_FILE_BYTES_PER_ROW
    )
    return rows * per_row


#: 전이 1케이스가 내는 시각 표본 수. **현재 적분기 설정값**(`t_eval` 4001점)이며
#: 데이터셋에 몇 점을 남길지는 5.5-B 에서 사람이 정한다 — 줄이면 행 수가 그만큼 준다.
TRANSIENT_SAMPLES_PER_CASE: int = 4001


def rows_for(count: CombinationCount, transient_samples: int) -> tuple[int, int]:
    """(정상상태 행 수, 전이 행 수). 행은 **(시나리오 × CDU) 하나당 하나**다.

    랙별 값은 열로 편다(rack0..rack7) — 행으로 펴면 CDU 총계가 8번 중복된다.
    """
    steady_rows = count.steady_cases * PLANT.cdu_count
    transient_rows = count.transient_cases * PLANT.cdu_count * transient_samples
    return steady_rows, transient_rows


@dataclass(frozen=True)
class ConfigCount:
    """CDU 구성 하나에 대한 케이스 수 — 시간 추정이 구성별로 달라서 나눈다."""

    config: str
    cdus_per_case: int
    steady_cases: int
    transient_cases: int


def count_by_configuration(load_levels: int) -> tuple[ConfigCount, ...]:
    """CDU 구성별 케이스 수. 구성마다 케이스당 시간이 크게 달라 따로 센다.

    구성 하나의 기본 조합 = 5장 범위 32 × 부하 × 누출 수준 4.
    전이는 여기에 M 2수준이 곱해진다(정상상태 해는 M 에 의존하지 않는다).
    """
    base = (
        len(default_hydraulic_cases())
        * 2  # NTU
        * 2  # 2차측 공급온도
        * load_levels
        * len(LEAK.k_multiplier_levels)
    )
    holdup_levels = len(holdup_bounds())
    return (
        ConfigCount("단일 CDU", 1, base, base * holdup_levels),
        ConfigCount("2대 대칭", 2, base, base * holdup_levels),
        ConfigCount("2대 비대칭", 2, base, base * holdup_levels),
    )


@dataclass(frozen=True)
class RunEstimate:
    """전량 생성 추정 — **추정이지 실측이 아니다.**"""

    label: str
    steady_cases: int
    transient_cases: int
    steady_seconds: float
    transient_seconds: float
    steady_rows: int
    transient_rows: int

    @property
    def total_seconds(self) -> float:
        return self.steady_seconds + self.transient_seconds

    @property
    def total_rows(self) -> int:
        return self.steady_rows + self.transient_rows


def estimate_run(
    load_levels: int,
    transient_samples: int,
    steady_single_s: float,
    steady_plant_s: float,
    transient_single_s: float,
    transient_plant_s: float,
    include_transient_configs: tuple[str, ...] = ("단일 CDU", "2대 대칭", "2대 비대칭"),
    label: str = "",
) -> RunEstimate:
    """C3 실측 시간으로 전량 소요를 **추정**한다.

    `include_transient_configs` 로 전이 쪽 구성을 줄일 수 있다 — 전이가 시간의
    대부분을 차지하므로 규모 축소는 대개 여기서 일어난다.
    """
    steady_seconds = 0.0
    transient_seconds = 0.0
    steady_rows = 0
    transient_rows = 0
    for count in count_by_configuration(load_levels):
        per_steady = steady_single_s if count.cdus_per_case == 1 else steady_plant_s
        per_transient = (
            transient_single_s if count.cdus_per_case == 1 else transient_plant_s
        )
        steady_seconds += count.steady_cases * per_steady
        steady_rows += count.steady_cases * count.cdus_per_case
        if count.config in include_transient_configs:
            transient_seconds += count.transient_cases * per_transient
            transient_rows += (
                count.transient_cases * count.cdus_per_case * transient_samples
            )
    return RunEstimate(
        label=label or f"부하 {load_levels}수준 · 전이 표본 {transient_samples}점",
        steady_cases=sum(c.steady_cases for c in count_by_configuration(load_levels)),
        transient_cases=sum(
            c.transient_cases
            for c in count_by_configuration(load_levels)
            if c.config in include_transient_configs
        ),
        steady_seconds=steady_seconds,
        transient_seconds=transient_seconds,
        steady_rows=steady_rows,
        transient_rows=transient_rows,
    )


# ─────────────────────────────────────────────────────────────────────────────
# C1. 스키마 초안 — **제안이지 확정이 아니다**
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA_DRAFT: tuple[tuple[str, str, str], ...] = (
    # (열 이름, 단위·형, 설명)
    ("scenario_id", "str", "시나리오 고유 키 — 재실행 시 같은 키가 같은 결과여야 한다"),
    ("scenario_kind", "str", "정상 / 이상 / 누출 — 5장 시나리오 구분"),
    ("regime", "str", "**steady / transient** — 사람이 정한 것 2. 섞어 읽지 않는다"),
    ("cdu_config", "str", "single / dual_symmetric / dual_asymmetric"),
    ("cdu_index", "int", "CDU 식별 (0-based)"),
    ("leak_level_percent", "%", "0 / 5 / 20 / 50 — 5장 누출 K값 증가율"),
    ("leak_rack_index", "int", "누출 주입 랙. 8랙 대칭이라 결과 불변(5-1)"),
    ("load_percent", "%", "5장 부하 프로파일"),
    ("pump_head_rated_mAq", "mAq", "5장 범위 축 — 20 또는 30"),
    ("branch_dp_rated_mAq", "mAq", "5장 범위 축 — 2 또는 3"),
    ("valve_dp_rated_mAq", "mAq", "5장 범위 축 — 3 또는 5"),
    ("ntu", "-", "5장 범위 축 — 2 또는 3"),
    ("T_secondary_supply_C", "℃", "5장 범위 축 — 27 또는 30 (고정 경계조건)"),
    ("holdup_mass_kg", "kg", "5-1 M 하한/상한. **transient 행에만 의미가 있다**"),
    ("t_s", "s", "전이 시각. steady 행은 비운다"),
    # ── 랙별 관측량 (rack0..rack7 로 편다) ──
    ("rack{i}_flow_Lps", "L/s", "랙별 유량 — 누출의 **가장 강한 신호**(세션 4)"),
    ("rack{i}_outlet_C", "℃", "랙별 출구온도 — 두 번째로 강한 신호"),
    # ── CDU 총계 ──
    ("total_flow_Lps", "L/s", "**부호 불확실** — 사람이 정한 것 1 참조"),
    ("pump_head_mAq", "mAq", "**부호 불확실** — 사람이 정한 것 1 참조"),
    ("T_supply_C", "℃", "1차측 공급온도"),
    ("T_return_C", "℃", "1차측 환수온도 (유량가중 혼합)"),
    ("hx_duty_kW", "kW", "열교환기 방열량"),
    ("secondary_share_Lps", "L/s", "공유 2차측 배분 (단일 CDU 는 정격 15.5)"),
    ("heat_capacity_ratio", "-", "Cr — **유도값**이다(세션 5-B). 선언값이 아니다"),
    ("energy_balance_residual_percent", "%", "6장 1-B 게이트 잔차"),
    # ── solver 플래그 (절대 규칙 5) ──
    ("hydraulic_solver_ier", "int", "헤더 압력평형 fsolve"),
    ("thermal_solver_converged", "bool", "물성 온도 고정점 fsolve"),
    ("plant_solver_ier", "int", "상위 연립 fsolve (단일 CDU 는 비운다)"),
    ("integrator_success", "bool", "solve_ivp (steady 행은 비운다)"),
)

#: 출처·한계 표기 — **행마다 반복하는 안(A)** 에서 추가되는 열.
PROVENANCE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("assumption_tag", ASSUMPTION_TAG),
    (
        "signal_sign_caveat",
        "총유량·양정은 K값 증가 근사(5장)의 결과다 — 실제 누출은 계통 밖 유출이라 "
        "부호가 반대일 수 있다. 랙 유량·랙 출구온도는 이 불확실성에 해당하지 않는다",
    ),
    (
        "transient_caveat",
        "전이 시간축은 M 노드 배분 규약(50:50)에 t63 기준 ±15% 의존한다(세션 4 C5). "
        "M 의 8랙 해석(#31)은 세션 5.7 에서 닫혔고 tau·t63·t95 가 정확히 8배로 "
        "이동했다. 그래도 절대 시간 규모는 해석하지 않는다 — M 결손(#21) · "
        "등가길이의 과대 방향 오차(#25) · 배관 규격 계열(#23)이 열려 있다",
    ),
    (
        "total_flow_caveat",
        "총유량만 보면 +50% 누출도 −0.16~0.37% 라 놓칠 수 있다(세션 4 관측 ③). "
        "랙 단위 열과 함께 읽는다",
    ),
    (
        "cr_caveat",
        "Cr 은 물성에서 유도된 값이다(세션 5-B). 세션 5-B 이전 로그의 수치와 "
        "나란히 읽을 수 없다",
    ),
)


def format_plan_report() -> str:
    """계수·측정·추정을 한 번에 낸다 (세션 5.5-A 산출물).

    절대 규칙 11: "가정값 기반 — 실측 아님" 표시를 반드시 넣는다.
    """
    lines = [
        "세션 5.5-A · 데이터셋 설계 보조 — 계수 · 실행시간 · 크기 추정",
        "※ " + ASSUMPTION_TAG,
        "※ 이 판은 **데이터셋을 생성하지 않는다.** 스키마는 제안이지 확정이 아니다.",
        "",
        "── 부하 축 후보별 조합 수 ─────────────────────────────────────────",
        f"{'부하 축':<34}{'정상상태':>10}{'전이':>10}{'합계':>10}",
    ]
    for description, levels, _basis in load_axis_candidates():
        count = count_combinations(levels)
        lines.append(
            f"{description:<34}{count.steady_cases:>10}"
            f"{count.transient_cases:>10}{count.total_cases:>10}"
        )
    lines += [
        "",
        "(CDU 구성 3종 · 누출 랙 축 끔 · 정상상태에는 M 축을 두지 않는다)",
        "",
        "── 실행시간 실측 ──────────────────────────────────────────────────",
    ]
    for sample in (
        measure_repeat_effect(),
        measure_steady_single(),
        measure_steady_plant(),
        measure_transient_single(),
        measure_transient_plant(),
    ):
        lines.append(
            f"{sample.label:<24} 평균 {sample.mean_s:8.4f} s · "
            f"범위 {sample.min_s:.4f}~{sample.max_s:.4f} s · "
            f"산포 {sample.spread_ratio:.2f}배 · n={len(sample.seconds)}"
        )
    lines += [
        "",
        "※ " + ASSUMPTION_TAG,
        "※ 실행시간은 이 PC 1회 측정이다. 다른 PC·다른 부하에서 달라진다.",
    ]
    return "\n".join(lines)


def main() -> int:
    import sys

    if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔 기본 인코딩 대비
        sys.stdout.reconfigure(encoding="utf-8")
    print(format_plan_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
