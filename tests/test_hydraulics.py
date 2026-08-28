"""세션 3-A/3-A2 — 수력 모듈 검사 (유량분배 단독 · 열모델 미결합).

CLAUDE.md 절대 규칙 10: 사람이 눈으로 보지 않고 **테스트가 판정한다.**

**이 파일은 6장 feasibility 기준을 하나도 판정하지 않는다.**
세션 3 게이트는 "헤더 압력평형 기반 유량분배(`fsolve`) 수렴 + 극단 케이스
(부하 0/최대) 비발산" 둘인데, 뒤쪽은 부하가 온도를 통해 들어오므로 열모델이
붙는 **세션 3-B** 의 것이다. 여기서 확인하는 것은 앞쪽 절반 — `fsolve` 수렴 —
과, 5-1 이 예고한 구조적 성질(균등분배)이 실제로 나오는지다.

`test_single_rack_K_increase_direction` 은 **누출 시나리오가 아니다.** 방향성
검사 하나이며, 신호 패턴 판정(세션 4 게이트)을 여기서 하지 않는다.

**세션 3-A2 가 더한 것**: 정격점 ΔP 항등성(역산의 역이 5장 표값을 그대로 낸다) ·
물성 온도에 대한 ΔP 의 밀도 비례성 · Kv·K 의 온도 불변성. 이 셋은 물성원이 하나
(CoolProp 래퍼)임을 고정하는 검사이며, feasibility 판정이 아니다.
"""

from __future__ import annotations

import math

import pytest

from cdu_simul.assumptions import PIPING, PUMP, SCENARIO, VALVE
from cdu_simul.fluid import coolant_density_kgm3
from cdu_simul.hydraulics import (
    HydraulicCase,
    branch_dp_mAq,
    branch_K_from_rated_dP,
    bulk_mean_temperature_C,
    default_cases,
    pump_head_mAq,
    rated_property_temperature_C,
    residual_share_at_rated_percent,
    solve_flow_distribution,
    valve_dp_mAq,
    valve_Kv_max_m3h_from_rated_dP,
)

#: 정격 물성 온도 [℃] — Kv·K 역산 기준이자 이 파일의 기본 ΔP 평가 온도.
RATED_T_C: float = rated_property_temperature_C()

#: 랙 간 유량 균등성 허용오차 [상대값].
#: 8랙이 완전히 동일한 조건이므로 해석적으로는 정확히 균등해야 한다. 남는 것은
#: `fsolve` 의 수치 잔차뿐이라 1e-9 로 잡는다 — 실측 잔차보다 느슨하면서,
#: 분배가 실제로 깨졌을 때(랙 하나만 K 를 바꿔도 % 단위로 벌어진다)보다는
#: 7자리 이상 엄격하다.
UNIFORMITY_TOLERANCE: float = 1.0e-9

#: 압력평형식 잔차 허용오차 [mAq]. 양정이 20~30 mAq 이므로 1e-8 은 상대 1e-9 다.
EQUATION_RESIDUAL_TOLERANCE_mAq: float = 1.0e-8


@pytest.mark.parametrize("case", default_cases(), ids=lambda c: c.label)
def test_fsolve_converges_for_all_eight_cases(case: HydraulicCase) -> None:
    """8조합 전부에서 `fsolve` 가 수렴한다 (`ier == 1`).

    세션 3 게이트의 **절반**이다(나머지 절반인 극단 케이스 비발산은 3-B).
    절대 규칙 5: `ier` 를 확인하지 않고 결과값을 쓰지 않는다.
    """
    result = solve_flow_distribution(case, RATED_T_C)
    assert result.solver_ier == 1, f"{case.label}: ier={result.solver_ier}"
    assert result.solver_converged
    assert result.max_abs_equation_residual_mAq < EQUATION_RESIDUAL_TOLERANCE_mAq


@pytest.mark.parametrize("case", default_cases(), ids=lambda c: c.label)
def test_uniform_racks_get_uniform_flow(case: HydraulicCase) -> None:
    """8랙 동일 조건에서 랙별 유량이 균등하다.

    5-1 「계통 잔여저항」의 예측을 실제로 확인하는 것이다 — 잔여저항이 랙 공통
    경로에 있으므로 정상상태 랙 간 유량은 균등해지고, 랙 간 상호작용은 펌프곡선을
    통해서만 생긴다. 이것이 나오지 않으면 5-1 배정 규칙을 코드가 잘못 구현한 것이다.

    **이것은 물리의 확인이 아니라 구현이 5-1 규칙과 맞는지의 확인이다** — 균등
    분배는 배정 방식이 강제하는 결과이지 모델이 발견한 사실이 아니다.
    """
    result = solve_flow_distribution(case, RATED_T_C)
    assert len(result.rack_flows_Lps) == SCENARIO.racks_per_cdu
    mean_Lps = result.mean_rack_flow_Lps
    for index, Q_Lps in enumerate(result.rack_flows_Lps):
        assert math.isclose(Q_Lps, mean_Lps, rel_tol=UNIFORMITY_TOLERANCE), (
            f"{case.label}: 랙 {index} 유량 {Q_Lps} vs 평균 {mean_Lps}"
        )


@pytest.mark.parametrize("case", default_cases(), ids=lambda c: c.label)
def test_single_rack_K_increase_direction(case: HydraulicCase) -> None:
    """랙 하나의 K값을 +50% 했을 때의 **방향성**만 본다.

    기대: 그 랙 유량 **감소** · 총유량 **감소** · 나머지 랙 유량 **증가**.
    나머지 랙이 늘어나는 이유는 총유량이 줄어 공통 경로의 잔여저항 강하가 줄고,
    그만큼 헤더에 남는 양정이 커지기 때문이다(5-1: 랙 간 상호작용은 펌프곡선과
    공통 경로를 통해서만 생긴다).

    **누출 시나리오가 아니다.** 절대 규칙 8 이 누출을 K값 변화로 근사한다고
    정했으므로 배수 크기만 5장 「대규모」 값을 빌려 왔을 뿐, 신호 패턴 판정
    (세션 4 게이트)은 여기서 하지 않는다.
    """
    from cdu_simul.assumptions import LEAK

    baseline = solve_flow_distribution(case, RATED_T_C)
    multipliers = (1.0 + LEAK.k_increase_percent_major / 100.0,) + (1.0,) * (
        case.n_racks - 1
    )
    perturbed = solve_flow_distribution(
        HydraulicCase(
            label=f"{case.label}+K50%@rack0",
            pump=case.pump,
            branch_K=case.branch_K,
            valve_Kv_max_m3h=case.valve_Kv_max_m3h,
            opening_fraction=case.opening_fraction,
            n_racks=case.n_racks,
            branch_K_multipliers=multipliers,
        ),
        RATED_T_C,
    )
    assert perturbed.solver_ier == 1

    assert perturbed.rack_flows_Lps[0] < baseline.rack_flows_Lps[0], (
        f"{case.label}: K 를 올린 랙의 유량이 줄지 않았다"
    )
    assert perturbed.total_flow_Lps < baseline.total_flow_Lps, (
        f"{case.label}: 총유량이 줄지 않았다"
    )
    for index in range(1, case.n_racks):
        assert perturbed.rack_flows_Lps[index] > baseline.rack_flows_Lps[index], (
            f"{case.label}: 랙 {index}(무변화) 유량이 늘지 않았다"
        )


@pytest.mark.parametrize(
    "coeffs", PUMP.curve_coefficient_bounds, ids=lambda c: c.label
)
def test_pump_curve_is_monotonically_decreasing(coeffs) -> None:  # type: ignore[no-untyped-def]
    """펌프곡선이 운전 유량 범위에서 단조감소한다.

    범위는 0 부터 정격유량의 2배까지 잡는다 — 5장이 운전 범위를 주지 않으므로
    정격을 기준으로 넉넉히 덮는 구간이며, 새 가정치가 아니라 **검사 구간**이다.
    단조감소가 깨지면 시스템곡선과 교점이 여러 개 생겨 `fsolve` 해가 초기값에
    의존하게 된다.
    """
    upper_Lps = 2.0 * PUMP.rated_flow_Lps
    samples = [upper_Lps * i / 200.0 for i in range(201)]
    heads = [pump_head_mAq(Q, coeffs) for Q in samples]
    for index in range(1, len(heads)):
        assert heads[index] < heads[index - 1], (
            f"{coeffs.label}: Q={samples[index]:.3f} L/s 에서 단조감소가 깨졌다"
        )


@pytest.mark.parametrize("case", default_cases(), ids=lambda c: c.label)
def test_residual_share_is_recorded_not_judged(case: HydraulicCase) -> None:
    """잔여저항 몫이 5-1 이 적은 60~83% 범위 안에 들어온다.

    5-1 「계통 잔여저항」의 한계 기록이 코드에서 실제로 재현되는지의 확인이다.
    **이 몫이 작아야 한다는 판정이 아니다** — 크다는 것이 5-1 이 이미 기록한
    사실이고(미해결 #24), 여기서는 그 사실이 유지되는지만 본다. 범위를 벗어나면
    전사값이나 배정 규칙이 어긋난 것이므로 사람이 봐야 한다.
    """
    share_percent = residual_share_at_rated_percent(case)
    assert 60.0 <= share_percent <= 84.0, f"{case.label}: {share_percent:.2f} %"


def test_assumptions_are_transcribed_not_derived() -> None:
    """5-1 전사값이 assumptions.py 에 그대로 있는지 확인한다 (절대 규칙 1·2).

    코드가 이 숫자들을 유도하지 않았음을 고정한다 — 누가 나중에 "역산해서 채우는"
    코드를 넣으면 이 테스트가 5-1 값과 어긋나며 깨진다.

    **Kv·K 는 여기 없다** — 세션 3-A2 에서 5-1 이 그 둘을 값이 아니라 **역산 규칙**
    으로 바꿨으므로, 전사가 아니라 재산출 대상이다(`test_valve_Kv_and_branch_K_...`).
    """
    low, high = PUMP.curve_coefficient_bounds
    assert (low.H0_mAq, low.a_mAq_per_Lps, low.b_mAq_per_Lps2) == (
        22.40,
        0.003819,
        0.009743,
    )
    assert (high.H0_mAq, high.a_mAq_per_Lps, high.b_mAq_per_Lps2) == (
        33.60,
        0.005729,
        0.014615,
    )
    assert PIPING.holdup_supply_node_fraction == 0.5
    assert PIPING.holdup_return_node_fraction == 0.5
    assert (
        PIPING.holdup_supply_node_fraction + PIPING.holdup_return_node_fraction == 1.0
    )


def test_valve_Kv_and_branch_K_are_not_hardcoded_constants() -> None:
    """5-1 이 Kv·K 를 역산 규칙으로 바꿨으므로 상수가 남아 있으면 안 된다.

    세션 3-A 가 전사했던 이름들(`VALVE.Kv_max_m3h_at_dP_low` 등)이 되살아나면
    물성원이 다시 둘로 갈린다(collaboration.md 결함유형 ③). 이름의 부재로 고정한다.
    """
    removed_names = (
        "Kv_max_m3h_at_dP_low",
        "Kv_max_m3h_at_dP_high",
        "Kv_max_bounds_m3h",
    )
    for removed in removed_names:
        assert not hasattr(VALVE, removed), f"VALVE.{removed} 가 되살아났다"
    assert not hasattr(PIPING, "rack_branch_K"), "PIPING.rack_branch_K 가 되살아났다"


@pytest.mark.parametrize("dP_rated_mAq", [2.0, 3.0])
def test_branch_dp_identity_at_rated_point(dP_rated_mAq: float) -> None:
    """역산한 K 로 정격 조건 ΔP 를 되계산하면 5장 표값이 그대로 나온다.

    **역산의 역이므로 항등적으로 성립해야 한다.** 세션 3-A 는 5-1 이 SG 를 숫자로
    적어 두는 바람에 이 항등성이 깨져 1.995 / 2.993 mAq 가 나왔다(미해결 #27).
    허용오차는 배정밀도 반올림(1e-12 mAq — 실측 잔차는 4.4e-16)이며, 물성원이
    다시 갈리면 0.1% 단위(1e-3 mAq)로 벌어져 이 검사가 깨진다.
    """
    K = branch_K_from_rated_dP(dP_rated_mAq)
    reproduced = branch_dp_mAq(VALVE.rated_flow_per_rack_Lps, K, RATED_T_C)
    assert reproduced == pytest.approx(dP_rated_mAq, abs=1.0e-12)


@pytest.mark.parametrize("dP_rated_mAq", [3.0, 5.0])
def test_valve_dp_identity_at_rated_point(dP_rated_mAq: float) -> None:
    """역산한 Kv 로 정격 조건 ΔP 를 되계산하면 5장 표값이 그대로 나온다.

    선형 개도 특성으로 100% 환산했다가 정격개도 80% 로 되돌아오는 경로까지
    함께 검사한다. 허용오차 근거는 위 분기 검사와 같다.
    """
    Kv_max = valve_Kv_max_m3h_from_rated_dP(dP_rated_mAq)
    reproduced = valve_dp_mAq(
        VALVE.rated_flow_per_rack_Lps,
        Kv_max,
        VALVE.rated_opening_fraction,
        RATED_T_C,
    )
    assert reproduced == pytest.approx(dP_rated_mAq, abs=1.0e-12)


def test_dp_scales_with_density_when_property_temperature_changes() -> None:
    """물성 온도를 바꾸면 ΔP 가 밀도에 비례해 움직인다 (32℃ 대 42℃).

    분기는 ΔP = K·ρ·v²/2 이고 유속 v 는 밀도와 무관하므로 ΔP ∝ ρ 다. 밸브는
    ΔP = SG·(Q/Kv)² 이고 SG = ρ/1000 이므로 역시 ΔP ∝ ρ 다. 두 경로가 같은
    물성원(CoolProp)을 쓰는지 확인하는 것이 이 검사의 목적이다 — 한쪽이 밀도를
    고정하고 있으면 비가 어긋난다.
    """
    T_cold_C = SCENARIO.T_primary_supply_C
    T_hot_C = SCENARIO.T_primary_return_C
    density_ratio = coolant_density_kgm3(T_cold_C) / coolant_density_kgm3(T_hot_C)
    assert density_ratio != 1.0, "두 온도의 밀도가 같으면 이 검사가 무의미하다"

    Q_Lps = VALVE.rated_flow_per_rack_Lps
    K = branch_K_from_rated_dP(PIPING.dP_per_rack_mAq.low)
    Kv_max = valve_Kv_max_m3h_from_rated_dP(VALVE.dP_at_rated_opening_mAq.low)

    branch_ratio = branch_dp_mAq(Q_Lps, K, T_cold_C) / branch_dp_mAq(
        Q_Lps, K, T_hot_C
    )
    valve_ratio = valve_dp_mAq(
        Q_Lps, Kv_max, VALVE.rated_opening_fraction, T_cold_C
    ) / valve_dp_mAq(Q_Lps, Kv_max, VALVE.rated_opening_fraction, T_hot_C)

    assert branch_ratio == pytest.approx(density_ratio, rel=1.0e-12)
    assert valve_ratio == pytest.approx(density_ratio, rel=1.0e-12)


def test_Kv_and_K_do_not_depend_on_property_temperature() -> None:
    """Kv·K 는 기기·형상 특성이므로 물성 온도에 따라 변하지 않는다(5-1).

    역산 함수가 온도 인자를 받지 않는 것으로 구조상 보장되지만, 누가 나중에
    "온도를 받게" 바꾸면 정격점 항등성(위 두 검사)과 Kv 의 의미가 함께 무너진다.
    반복 호출이 같은 값을 주는지, 그리고 온도 인자를 받지 않는지로 고정한다.
    """
    import inspect

    for func in (branch_K_from_rated_dP, valve_Kv_max_m3h_from_rated_dP):
        params = inspect.signature(func).parameters
        assert "T_property_C" not in params, f"{func.__name__} 이 온도를 받게 됐다"

    assert branch_K_from_rated_dP(2.0) == branch_K_from_rated_dP(2.0)
    assert valve_Kv_max_m3h_from_rated_dP(3.0) == valve_Kv_max_m3h_from_rated_dP(3.0)


def test_rated_property_temperature_follows_bulk_mean_rule() -> None:
    """정격 물성 온도가 5장 1차측 공급·환수의 벌크평균인지 확인한다 (5-1 규약)."""
    assert rated_property_temperature_C() == bulk_mean_temperature_C(
        SCENARIO.T_primary_supply_C, SCENARIO.T_primary_return_C
    )


def test_pump_curve_reproduces_five_chapter_rated_head() -> None:
    """전사한 계수가 5장 정격점(15.5 L/s → 20 / 30 mAq)을 재현하는지 확인한다.

    5-1 이 "정격점은 5장 값으로 스케일했다"고 적었으므로, 그 진술이 전사된
    숫자에서 실제로 성립하는지 본다. 허용오차 1e-3 mAq 는 5-1 계수의 유효자리
    (소수 6자리)에서 오는 반올림 폭이다.
    """
    low, high = PUMP.curve_coefficient_bounds
    rated_Lps = PUMP.rated_flow_Lps
    assert pump_head_mAq(rated_Lps, low) == pytest.approx(
        PUMP.rated_head_mAq.low, abs=1.0e-3
    )
    assert pump_head_mAq(rated_Lps, high) == pytest.approx(
        PUMP.rated_head_mAq.high, abs=1.0e-3
    )
