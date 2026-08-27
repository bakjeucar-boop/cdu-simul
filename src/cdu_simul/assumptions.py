"""5장 가정치 전사 — 이 파일이 시나리오 수치의 **유일한 출처**다.

`AI서버_CDU_디지털트윈_프로젝트정리.md` 5장 「가상 시나리오 가정치」 표를 그대로
옮긴 것이다. 실측·설계값이 아니다. 설계데이터 확보 시 이 파일 하나만 교체한다
(CLAUDE.md 절대 규칙 2 · collaboration.md 판단원칙 ④).

전사 규칙 (세션 1-A에서 정함):
- 5장 표에 **적힌 값만** 옮긴다. 표에 없는 파생값(밸브 Cv 역산 결과, 펌프
  특성곡선 계수 a·b, 배관 K값 등 계산이 필요한 값)은 이 파일에 없다.
  필요해지면 코드에서 임의로 만들지 않고 사람에게 확인한다(절대 규칙 1).
- 표에 범위로 적힌 값은 `Range` 로 **범위 그대로** 둔다. 대표값(중앙값 등)을
  임의로 고르지 않는다 — 대표값이 필요해지는 시점에 사람이 정한다.
- 단위는 **5장 표 원단위 그대로**(mAq · L/s · ℃ · kW · %)이며 변환하지 않는다.
  변환 지점은 세션 1-B에서 정한다(절대 규칙 9).
- 모든 수치에 출처 태그가 붙는다: [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
"""

from __future__ import annotations

from dataclasses import dataclass

#: 산출물(데이터셋·리포트)에 반드시 붙이는 출처 표시 (절대 규칙 11)
ASSUMPTION_TAG = "[가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장] 실측 아님"


@dataclass(frozen=True)
class Range:
    """5장 표에 범위로 적힌 값. 대표값은 이 판에서 고르지 않는다.

    의도적으로 중앙값·평균 등 대표값 추출 메서드를 두지 않는다 — 대표값 선정은
    사람의 판단이며, 편의 메서드가 있으면 무심코 임의값이 코드에 들어간다.
    """

    low: float
    high: float
    unit: str


# ─────────────────────────────────────────────────────────────────────────────
# 5장 「시나리오 조건」 표
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ScenarioConditions:
    """5장 시나리오 조건 표 전사."""

    # 데이터센터 유형 [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    datacenter_type: str = "AI 트레이닝/추론 GPU 클러스터"

    # 랙당 구성 [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    rack_configuration: str = "8-GPU 서버 다수 (H100급 상당)"

    # 랙당 IT 발열량 80 kW [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    rack_it_load_kW: float = 80.0

    # CDU당 연결 랙 수 8개 [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    racks_per_cdu: int = 8

    # CDU당 총 발열량 640 kW [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    # (표에 적힌 값 그대로 둔다 — 80×8 로 코드에서 재계산하지 않는다)
    cdu_total_load_kW: float = 640.0

    # 여유율 20% 감안 CDU 정격 ~750 kW [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    # 표기가 "~750"(근사)이므로 근사값임을 잊지 않는다.
    cdu_design_margin_percent: float = 20.0
    cdu_rated_capacity_kW: float = 750.0

    # 냉각액 PG25 (25wt% 프로필렌글리콜 수용액)
    # [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    coolant_label: str = "PG25 (25wt% 프로필렌글리콜 수용액)"
    #: CoolProp 유체 문자열 — **여기 한 곳에서만 정의한다** (collaboration.md ④).
    #: 다른 파일은 이 상수를 참조한다. fluid.py 는 이것을 import 해서 쓴다.
    coolant_coolprop_id: str = "INCOMP::MPG-25%"

    # 1차측 공급/환수 온도 32℃ / 42℃ (ΔT 10℃)
    # [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    T_primary_supply_C: float = 32.0
    T_primary_return_C: float = 42.0
    dT_primary_C: float = 10.0  # 표에 명시된 값 (재계산하지 않는다)

    # 2차측 공급온도(경계조건) 27~30℃ 고정
    # [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    # 절대 규칙 7: 2차측은 다중 CDU 확장 전까지 이 고정 경계조건만 쓴다.
    T_secondary_supply_C: Range = Range(27.0, 30.0, "degC")


# ─────────────────────────────────────────────────────────────────────────────
# 5장 「주요 구성요소」 표
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PumpAssumptions:
    """5장 구성요소 표 — 펌프."""

    # 정격유량 약 15.5 L/s (930 L/min) [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    # 표가 두 단위를 병기하므로 둘 다 원단위로 보존한다(변환하지 않는다).
    rated_flow_Lps: float = 15.5
    rated_flow_Lpm: float = 930.0

    # 정격양정 20~30 mAq [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    rated_head_mAq: Range = Range(20.0, 30.0, "mAq")

    # 특성곡선 형태 H = H0 - a*Q - b*Q**2
    # [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    # 계수 H0·a·b 는 5장 표에 없다 — 이 파일에 두지 않는다(#2 미해결).
    curve_form: str = "H = H0 - a*Q - b*Q**2"


@dataclass(frozen=True)
class ValveAssumptions:
    """5장 구성요소 표 — 밸브(랙별)."""

    # 랙당 정격유량 약 1.94 L/s [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    rated_flow_per_rack_Lps: float = 1.94

    # 정격개도 80% [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    rated_opening_percent: float = 80.0

    # 정격개도에서 ΔP 3~5 mAq [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    dP_at_rated_opening_mAq: Range = Range(3.0, 5.0, "mAq")

    # Cv 는 위 조건에서 "역산"하는 파생값이다 — 5장 표에 수치가 없으므로
    # 이 판에서 만들지 않는다(#1 미해결).


@dataclass(frozen=True)
class HeatExchangerAssumptions:
    """5장 구성요소 표 — 열교환기."""

    # 정격용량 750 kW [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    rated_capacity_kW: float = 750.0

    # ε-NTU법, NTU 2~3 [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    method: str = "ε-NTU"
    ntu: Range = Range(2.0, 3.0, "-")

    # 1차:2차 유량비 1:1 가정 [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    flow_ratio_primary_to_secondary: float = 1.0


@dataclass(frozen=True)
class PipingAssumptions:
    """5장 구성요소 표 — 배관(1차측)."""

    # 랙 분기 25A(1") [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    rack_branch_nominal_size: str = '25A (1")'

    # CDU 헤더 65~80A [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    header_nominal_size_A: Range = Range(65.0, 80.0, "A")

    # 등가길이 20~30 m [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    equivalent_length_m: Range = Range(20.0, 30.0, "m")

    # 랙당 ΔP 2~3 mAq [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    dP_per_rack_mAq: Range = Range(2.0, 3.0, "mAq")

    # 배관 K값(저항계수)은 위 ΔP·유량에서 역산하는 파생값이다 — 5장 표에 수치가
    # 없으므로 이 판에서 만들지 않는다(#1 미해결).

    # ── 배관 내경 ────────────────────────────────────────────────────────────
    # [규격값: ASME B36.10M Sch40 · 프로젝트정리 5-1 · 세션 2 확정]
    # 5장 배관 항목이 호칭경만 적고 내경이 없어 계통 보유수량(열용량 M)을 낼 수
    # 없었다. 호칭경→내경은 배관 규격표의 **정의값**이므로 [가정값] 이 아니라
    # [규격값] 으로 구분한다 — 설계데이터로 교체될 성질의 값이 아니다.
    # 5-1 기록: KS D 3507 계열(25A 내경 27.5 mm)을 쓰면 약 7% 커지고 시간상수가
    # 그만큼 이동한다. 실제 배관 재질·규격 계열은 설계데이터 확보 시 확정.
    rack_branch_inner_diameter_mm: float = 26.64  # 25A (NPS 1)
    header_65A_inner_diameter_mm: float = 62.71  # 65A (NPS 2½)
    header_80A_inner_diameter_mm: float = 77.92  # 80A (NPS 3)

    @property
    def holdup_bound_inner_diameters_mm(self) -> tuple[float, float]:
        """계통 보유수량 M 의 범위를 내는 두 극단 구경 (하한, 상한) [mm].

        [가정값: 프로젝트정리 5-1 「계통 보유수량 M」 · 세션 2 확정]

        5장 등가길이를 랙 1개 회로의 **왕복 전체 등가길이**로 읽되, 구경 배분
        (25A 대 헤더)이 5장에 없으므로 **전부 25A(하한) / 전부 80A(상한)** 두
        극단으로 M 의 범위를 낸다. 배분을 특정하면 5장에 없는 숫자를 만드는 것이
        된다 — 범위값 방침 (B)(양 끝을 둘 다 돌린다)를 그대로 적용한 것이다.

        위 내경 상수를 다시 적지 않고 참조만 한다(절대 규칙 2).

        **한계(5-1 기록)**: 열교환기·CDU 내부 보유량과 랙 콜드플레이트 보유량이
        5장에 없어 M 에서 빠진다 — 실제보다 M 이 작고 수렴시간도 실제보다 짧게
        나온다. 이 사실 없이 수렴시간을 "합리적"이라고 읽지 않는다.
        """
        return (self.rack_branch_inner_diameter_mm, self.header_80A_inner_diameter_mm)


@dataclass(frozen=True)
class LoadProfileAssumptions:
    """5장 구성요소 표 — 부하 프로파일."""

    # 유휴 20% ~ 정격 100% [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    idle_load_percent: float = 20.0
    rated_load_percent: float = 100.0

    # 스텝/랜덤 변화 [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    variation_pattern: str = "스텝/랜덤 변화"


@dataclass(frozen=True)
class LeakScenarioAssumptions:
    """5장 구성요소 표 — 누출 시나리오.

    절대 규칙 8: 누출은 배관저항(K값) 변화로 근사한다. 다른 방식으로 바꾸려면
    먼저 사람에게 확인한다.
    """

    # 배관 K값 정상 대비 +5% (미량) [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    k_increase_percent_minor: float = 5.0
    # +20% (중간) [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    k_increase_percent_moderate: float = 20.0
    # +50% (대규모) [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    k_increase_percent_major: float = 50.0


# ─────────────────────────────────────────────────────────────────────────────
# 모듈 수준 단일 인스턴스 — 코드는 여기서만 읽는다 (절대 규칙 2)
# ─────────────────────────────────────────────────────────────────────────────
SCENARIO = ScenarioConditions()
PUMP = PumpAssumptions()
VALVE = ValveAssumptions()
HEAT_EXCHANGER = HeatExchangerAssumptions()
PIPING = PipingAssumptions()
LOAD_PROFILE = LoadProfileAssumptions()
LEAK = LeakScenarioAssumptions()
