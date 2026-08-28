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

**5-1 「파생 가정치」 경로 (세션 1-B 이후)**: 5장 표에 없는 값이 필요해지면
프로젝트정리 5-1 「빈칸 처리 순서」(역산 → 규격·정의값 → 범위 양 끝 → 추가
파라미터 0의 규약)를 거쳐 **사람이 5-1 에 확정한 뒤** 이 파일로 옮긴다. 이
파일이 5-1 값을 스스로 유도하지 않는다 — **5-1 에 적힌 숫자를 그대로 전사**한다.
태그는 근거의 성격에 따라 [규약: 5-1] · [역산: 5-1] · [규격값: 5-1] ·
[가정값: 5-1] 로 구분한다.
"""

from __future__ import annotations

from dataclasses import dataclass

#: 산출물(데이터셋·리포트)에 반드시 붙이는 출처 표시 (절대 규칙 11)
ASSUMPTION_TAG = "[가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장] 실측 아님"

#: 압력 단위 규약 — 1 mAq 를 몇 Pa 로 읽을 것인가.
#: [규약: 프로젝트정리 5-1 「압력 단위 규약 mAq」 · 세션 3 확정]
#: 5장이 mAq·L/s·℃ 를 섞어 쓰는데, mAq 를 PG25 액주 높이로 읽으면 값이 1.3%
#: 달라진다. 배관 실무에서 mAq 는 통상 **수두 환산 압력 단위**이므로 그쪽을 택했다.
#: **물리 가정이 아니라 단위 규약**이다 — 냉각액 밀도와 무관하게 고정이다.
#: mAq↔Pa 변환은 이 상수 하나로만 한다(절대 규칙 9).
PASCAL_PER_MAQ: float = 9806.65


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


@dataclass(frozen=True)
class PumpCurveCoefficients:
    """펌프 특성곡선 H = H0 - a*Q - b*Q**2 의 계수 한 세트.

    단위는 **5-1 이 적은 그대로** H·H0 는 mAq, Q 는 L/s 다(절대 규칙 9).
    """

    label: str
    H0_mAq: float
    a_mAq_per_Lps: float
    b_mAq_per_Lps2: float


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
    curve_form: str = "H = H0 - a*Q - b*Q**2"

    # ── 특성곡선 계수 H0·a·b ─────────────────────────────────────────────────
    # [가정값: 프로젝트정리 5-1 「펌프 특성곡선 계수 H0·a·b」 · 세션 3 확정]
    # 5-1 에 적힌 숫자를 그대로 전사한다 — 여기서 다시 유도하지 않는다.
    # 근거(5-1): 곡선 **형상만** 타 프로젝트 사례(4 kW 인라인·3550 rpm·정격
    # 210 L/min·25 m·체절 28 m)에서 차용하고, 정격점은 5장 값(15.5 L/s ·
    # 20~30 mAq)으로 스케일했다. 무차원 형상은
    # H/H정격 = 1.12 - 0.00296q - 0.11704q^2 (q = Q/Q정격).
    # **한계(5-1)**: 차용 곡선은 물 기준이며 PG25 점도 영향이 반영돼 있지 않다.
    # 원 펌프와 용량이 4.4배 차이 나 비속도가 달라 실제 형상은 다를 수 있다.
    # 5장 정격양정이 범위(20~30 mAq)이므로 계수도 **양 끝 두 세트**다
    # (범위값 방침 (B) — 양 끝을 둘 다 돌린다).
    curve_coefficients_at_head_low: PumpCurveCoefficients = PumpCurveCoefficients(
        label="H_rated=20mAq", H0_mAq=22.40,
        a_mAq_per_Lps=0.003819, b_mAq_per_Lps2=0.009743,
    )
    curve_coefficients_at_head_high: PumpCurveCoefficients = PumpCurveCoefficients(
        label="H_rated=30mAq", H0_mAq=33.60,
        a_mAq_per_Lps=0.005729, b_mAq_per_Lps2=0.014615,
    )

    @property
    def curve_coefficient_bounds(self) -> tuple[PumpCurveCoefficients, ...]:
        """5장 정격양정 범위 양 끝에 대응하는 계수 두 세트 (하한, 상한)."""
        return (
            self.curve_coefficients_at_head_low,
            self.curve_coefficients_at_head_high,
        )


@dataclass(frozen=True)
class ValveAssumptions:
    """5장 구성요소 표 — 밸브(랙별)."""

    # 랙당 정격유량 약 1.94 L/s [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    rated_flow_per_rack_Lps: float = 1.94

    # 정격개도 80% [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    rated_opening_percent: float = 80.0

    # 정격개도에서 ΔP 3~5 mAq [가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장]
    dP_at_rated_opening_mAq: Range = Range(3.0, 5.0, "mAq")

    # ── 개도 특성 ────────────────────────────────────────────────────────────
    # [규약: 프로젝트정리 5-1 「밸브 개도 특성」 · 세션 3 확정]
    # **선형** Kv(x) = Kv_max * x (x 는 개도 분율).
    # 근거(5-1): 등비율(equal-percentage)이 실제 제어밸브에 더 흔하나
    # rangeability R 이라는 **새 숫자가 하나 더 필요**하고 그것을 정할 근거가 없다.
    # 선형은 추가 파라미터가 0이다. 개도가 80% 고정인 동안은 어느 쪽이든 결과가
    # 같으므로, **개도가 변수가 되는 시점(세션 4)에 재검토**한다.
    opening_characteristic: str = "linear: Kv(x) = Kv_max * x"

    # ── Kv (개도 100% 기준) — **상수가 아니라 역산 규칙이다** ────────────────
    # [역산 규칙: 프로젝트정리 5-1 「밸브 Kv (개도 100% 기준)」 · 세션 3-A2]
    # **이 파일에 Kv 숫자를 두지 않는다.** 5-1 이 값이 아니라 규칙을 주기 때문이다:
    # 위 5장 원 조건(랙당 1.94 L/s · 개도 80% · ΔP 3~5 mAq)에 Kv = Q*sqrt(SG/ΔP)
    # 를 적용해 80% 값을 구하고 선형 개도 특성으로 100% 로 환산한다.
    # 역산 함수는 `hydraulics.valve_Kv_max_m3h_from_rated_dP` 다 — 물성(SG)이
    # CoolProp 래퍼에서 오고 ΔP 식이 그 모듈에 있으므로 그쪽에 둔다(이 파일이
    # `fluid` 를 import 하면 순환 import 가 된다).
    # 세션 3-A 는 5-1 이 적어 둔 숫자(16.19 / 12.54)를 전사했는데, 그 숫자가
    # 5-1 의 SG(1.0124)로 역산된 것이라 CoolProp 값(1.01147)과 0.09% 갈렸고
    # 정격점 ΔP 가 5장 표값(3/5 mAq)을 재현하지 못했다(미해결 #27).

    @property
    def rated_opening_fraction(self) -> float:
        """정격개도를 분율로 (80% → 0.8). %↔분율 변환 지점을 한 곳에 둔다."""
        return self.rated_opening_percent / 100.0


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

    # ── 배관 K값 (랙 분기) — **상수가 아니라 역산 규칙이다** ─────────────────
    # [역산 규칙: 프로젝트정리 5-1 「배관 K값 (랙 분기)」 · 세션 3-A2]
    # **이 파일에 K 숫자를 두지 않는다.** 위 5장 원 조건("랙당 ΔP 2~3 mAq @
    # 1.94 L/s")과 아래 배관 내경(25A)으로 ΔP = K*rho*v^2/2 를 역산한다.
    # rho 는 CoolProp 에서 오므로 역산 함수는 `hydraulics.branch_K_from_rated_dP`
    # 에 있다(이 파일이 `fluid` 를 import 하면 순환 import 가 된다).
    # K 는 무차원이며 **25A 내경 기준 유속으로 정의**된다 — 다른 구경에 그대로
    # 쓰면 안 된다. 누출 시나리오의 K값 +5/+20/+50%(`LEAK`)가 이 값에 걸린다
    # (절대 규칙 8).
    # **관측(5-1)**: 25A 에 1.94 L/s 는 유속 3.48 m/s 로 냉각수 통상 설계유속
    # (2~3 m/s)보다 높다 — 설계데이터 확보 시 랙 분기 구경이 커질 가능성이 있고,
    # 그러면 K 와 M 이 함께 이동한다.

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

    # ── 계통 보유수량 M 의 노드 배분 ────────────────────────────────────────
    # [규약: 프로젝트정리 5-1 「계통 보유수량 M의 노드 배분」 · 세션 3 확정]
    # 공급 노드 50% · 환수 노드 50%.
    # 근거(5-1): 5-1 의 M 규칙이 등가길이를 "왕복 전체"로 읽으므로 왕복은 두
    # 다리이고, 비대칭 배분은 5장에 없는 배분비를 새로 요구한다. 50:50 은
    # **추가 파라미터가 0인 유일한 배분**이다. **물리 가정이 아니라 수치처리
    # 규약**이다.
    # 배분비는 정상상태 해에 영향하지 않고 **전이 파형에만** 영향한다 —
    # 수렴시간이 이미 판정 불가(미해결 #21)이므로 현재 게이트를 바꾸지 않는다.
    # 누출 신호가 전이 파형에 걸리는 **세션 4에서 민감도를 확인**한다.
    # **이 값을 쓰는 것은 열모델 결합(세션 3-B)이다** — 세션 3-A 는 수력만 푼다.
    holdup_supply_node_fraction: float = 0.5
    holdup_return_node_fraction: float = 0.5


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
