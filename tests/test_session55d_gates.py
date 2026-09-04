"""세션 5.5-D — 다중 CDU 「막힘」 주입 지점 정정 (미해결 #34).

CLAUDE.md 절대 규칙 10: 사람이 눈으로 보지 않고 **테스트가 판정한다.**

═══════════════════════════════════════════════════════════════════════════════
**판정 기준 (선기재 — 코드보다 먼저 적었다)**

이 판은 **새 feasibility 기준을 만들지 않는다.** 세션 5.5 게이트와 앞 게이트는
각자의 파일이 그대로 판정하고, 여기서는 **이 판이 고친 것이 실제로 고쳐졌는지**를
본다. 그래서 아래 넷은 게이트가 아니라 **정정의 확인**이다.

**기준 ㉠ — 주입점이 계 전체에 하나다.** 다중 CDU 전이 케이스에서 「막힘」 K값이
적용된 랙이 **CDU 를 통틀어 정확히 1개**여야 한다. 5-1 「「막힘」 주입 지점」이
"랙 1개에 주입한다 · 전 랙 동시 「막힘」은 5장에 근거가 없어 돌리지 않는다" 이므로,
2개면 규정 밖이다. 세션 5.5-B 는 2개였다.

**기준 ㉡ — 대칭 경로가 그대로다.** `templates` 를 주지 않은 `PlantLoadStepCase`
와 같은 템플릿을 n 벌 준 것이 **완전히 같은 결과**를 내야 한다(`==`). 새 인자가
기존 경로를 건드렸다면 여기서 드러난다. **허용오차를 두지 않는다.**

**기준 ㉢ — 저장 격자가 `storage_times_s` 다.** `integrate_plant_load_step` 의
`t_s` 가 `storage_times_s(t_end_s, tau)` 와 **정확히 같아야** 한다(종전에는 균등
2001점이었고 `dataset.py` 가 최근접 점을 골라 맞췄다). 적분 설정(RK45·rtol·atol·
구간)은 손대지 않았으므로 **정상상태 종점은 이동하지 않아야 한다.**

**기준 ㉣ — CDU 간 연동이 전이에 나타난다.** 이 판의 목적이다. 누출을 CDU A 에만
걸었을 때, 부하가 그대로인 CDU B 의
  · 공유 2차측 배분 몫이 누출 수준에 대해 **엄격히 증가**하고
  · 환수온도가 누출 수준에 대해 **엄격히 감소**해야 한다
(A 의 유량이 줄어 A 의 몫이 줄고 B 의 몫이 늘어 B 가 더 냉각된다). 방향은 세션 5
C7 이 정상상태에서 관측한 것과 같아야 한다 — C7 은 +50% 에서 B 가 −0.012649 K,
배분 +0.01905 L/s 였다. **세션 5.5-B 의 양쪽 누출에서는 배분이 거의 움직이지
않아(≈1e-5 L/s) 이 연동이 상쇄됐다** — 그것이 #34 였다.

**이 파일이 판정하지 않는 것**
· 연동의 **크기**가 물리적으로 옳은가 — 실측이 없다. 부호와 단조성만 본다
· 세션 5.5 게이트(상태 이월·전 행 표기) — `test_session55_gates.py` 의 몫이다
· 수렴시간 — 여전히 판정하지 않는다(#21 · #31)
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cdu_simul.assumptions import LOAD_PROFILE, PLANT
from cdu_simul.dataset import (
    CONFIG_DUAL_ASYMMETRIC,
    CONFIG_SINGLE,
    LEAK_CDU_INDEX,
    ScenarioSpec,
    column_names,
    enumerate_specs,
    rows_for,
)
from cdu_simul.dynamics import holdup_bounds, storage_times_s
from cdu_simul.leak import leak_case, leak_levels
from cdu_simul.model import CduCase, default_cdu_cases
from cdu_simul.plant import (
    PlantLoadStepCase,
    default_plant_load_step_cases,
    integrate_plant_load_step,
    plant_case,
    plant_case_from_templates,
)

#: 이 파일은 전이를 여러 번 돌린다(케이스당 수 초). 범위 하단 조합 하나로 고정한다 —
#: 32조합 전수는 앞 게이트 파일들이 이미 돌린다.
TEMPLATE: CduCase = default_cdu_cases()[0]
HOLDUP = holdup_bounds()[0]
IDLE = LOAD_PROFILE.idle_load_percent
RATED = LOAD_PROFILE.rated_load_percent


def _leak_on_a(multiplier: float) -> tuple[CduCase, ...]:
    """CDU A 에만 누출을 건 템플릿 쌍."""
    level = next(lv for lv in leak_levels() if lv.k_multiplier == multiplier)
    return (leak_case(TEMPLATE, level), TEMPLATE)


def _step_case(templates: tuple[CduCase, ...] | None) -> PlantLoadStepCase:
    """A 만 20→100% 스텝. B 는 20% 고정."""
    return PlantLoadStepCase(
        label="세션 5.5-D 확인",
        holdup=HOLDUP,
        template=TEMPLATE,
        templates=templates,
        load_before_percents=(IDLE,) * PLANT.cdu_count,
        load_after_percents=(RATED,) + (IDLE,) * (PLANT.cdu_count - 1),
    )


def _leaked_rack_count(case: PlantLoadStepCase) -> int:
    """이 케이스가 만드는 플랜트에서 누출 K값이 걸린 랙의 **총 수**."""
    plant = case.plant_at(case.load_after_percents)
    normal_K = TEMPLATE.hydraulic.branch_K
    return sum(
        1
        for cdu in plant.cdus
        for K in cdu.hydraulic.rack_branch_K
        if K != normal_K
    )


# ─────────────────────────────────────────────────────────────────────────────
# 기준 ㉠ — 누출점이 계 전체에 하나다
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "multiplier", [lv.k_multiplier for lv in leak_levels() if lv.k_multiplier != 1.0]
)
def test_criterion_1_single_leak_point_in_whole_plant(multiplier: float) -> None:
    """기준 ㉠ — 다중 CDU 전이에서 「막힘」 랙이 계 전체에 정확히 1개다.

    세션 5.5-B 는 `PlantLoadStepCase` 가 CDU 별 템플릿을 못 받아 **2개**였다 —
    5-1 「「막힘」 주입 지점」의 규정 밖이었다(미해결 #34).
    """
    count = _leaked_rack_count(_step_case(_leak_on_a(multiplier)))
    assert count == 1, (
        f"「막힘」 K값이 걸린 랙이 {count}개다 — 5-1 「「막힘」 주입 지점」은 랙 1개다"
    )


def test_criterion_1_leak_sits_on_the_declared_cdu() -> None:
    """누출이 `LEAK_CDU_INDEX` 가 가리키는 CDU 에 있다 — 다른 CDU 는 정상이다."""
    case = _step_case(_leak_on_a(1.5))
    plant = case.plant_at(case.load_after_percents)
    normal_K = TEMPLATE.hydraulic.branch_K
    for index, cdu in enumerate(plant.cdus):
        changed = [K for K in cdu.hydraulic.rack_branch_K if K != normal_K]
        if index == LEAK_CDU_INDEX:
            assert len(changed) == 1, f"CDU {index}: 누출 랙이 {len(changed)}개다"
        else:
            assert not changed, f"CDU {index}: 누출이 걸려 있다 — 걸리면 안 된다"


def test_criterion_1_no_leak_means_no_changed_rack() -> None:
    """배율 1.0(정상)은 어느 CDU 에도 K값을 바꾸지 않는다 — 기준선이 성립한다."""
    assert _leaked_rack_count(_step_case(_leak_on_a(1.0))) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 기준 ㉡ — 대칭 경로가 그대로다
# ─────────────────────────────────────────────────────────────────────────────
def test_criterion_2_plant_case_helpers_agree() -> None:
    """`plant_case` 와 같은 템플릿 n 벌의 `plant_case_from_templates` 가 같다."""
    loads = (RATED, IDLE)
    assert plant_case("x", loads, TEMPLATE) == plant_case_from_templates(
        "x", loads, (TEMPLATE,) * PLANT.cdu_count
    )


def test_criterion_2_templates_none_matches_repeated_template() -> None:
    """`templates=None` 이 같은 템플릿 n 벌과 **완전히 같은 플랜트**를 만든다."""
    without = _step_case(None)
    with_explicit = _step_case((TEMPLATE,) * PLANT.cdu_count)
    assert without.cdu_templates() == with_explicit.cdu_templates()
    assert without.plant_at(without.load_after_percents) == (
        with_explicit.plant_at(with_explicit.load_after_percents)
    )


def test_criterion_2_symmetric_transient_is_bit_identical() -> None:
    """대칭 경로의 **적분 결과**가 `==` 로 같다. 허용오차를 두지 않는다."""
    a = integrate_plant_load_step(_step_case(None))
    b = integrate_plant_load_step(_step_case((TEMPLATE,) * PLANT.cdu_count))
    assert np.array_equal(a.t_s, b.t_s)
    for x, y in zip(a.T_supply_C + a.T_return_C, b.T_supply_C + b.T_return_C):
        assert np.array_equal(x, y), "대칭 경로가 새 인자 때문에 달라졌다"
    assert a.secondary_shares_final_Lps == b.secondary_shares_final_Lps


def test_criterion_2_template_count_is_validated() -> None:
    """CDU 수와 템플릿 수가 어긋나면 조용히 넘어가지 않는다."""
    with pytest.raises(ValueError):
        PlantLoadStepCase(
            label="x",
            holdup=HOLDUP,
            template=TEMPLATE,
            templates=(TEMPLATE,),  # 1개 — CDU 는 2대다
            load_before_percents=(IDLE, IDLE),
            load_after_percents=(RATED, IDLE),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 기준 ㉢ — 저장 격자가 storage_times_s 다
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "case", default_plant_load_step_cases(), ids=lambda c: c.label
)
def test_criterion_3_storage_grid_is_shared_with_dynamics(case) -> None:  # type: ignore[no-untyped-def]
    """기준 ㉢ — 다중 CDU 전이의 `t_s` 가 단일 CDU 경로와 **같은 격자**다.

    종전에는 균등 2001점이었고 `dataset.py` 가 최근접 점을 골라 맞췄다(시각 오차
    최대 0.005τ). 이제 두 경로가 같은 함수를 쓰므로 오차가 0 이다.
    """
    result = integrate_plant_load_step(case)
    expected = storage_times_s(result.t_end_s, result.tau_theory_s)
    assert np.array_equal(result.t_s, expected), (
        f"{case.label}: 저장 격자가 storage_times_s 와 다르다"
    )
    assert result.t_s[0] == 0.0
    assert result.t_s[-1] == result.t_end_s


def test_criterion_3_integration_settings_untouched() -> None:
    """적분 설정은 `dynamics` 의 상수를 그대로 쓴다 — 저장 격자만 바꿨다."""
    import inspect

    from cdu_simul import plant

    source = inspect.getsource(plant.integrate_plant_load_step)
    assert 'method="RK45"' in source
    assert "rtol=INTEGRATION_RTOL" in source
    assert "atol=INTEGRATION_ATOL" in source
    assert "np.linspace" not in source, "균등 격자가 남아 있다"


# ─────────────────────────────────────────────────────────────────────────────
# 기준 ㉣ — CDU 간 연동이 전이에 나타난다 (이 판의 목적)
# ─────────────────────────────────────────────────────────────────────────────
def _b_response() -> list[tuple[float, float, float]]:
    """(누출 배율, B 의 최종 2차측 몫, B 의 최종 환수온도)."""
    out = []
    for level in leak_levels():
        result = integrate_plant_load_step(_step_case(_leak_on_a(level.k_multiplier)))
        assert result.solver_success, f"{level.label}: solve_ivp 실패"
        assert result.hydraulic_solver_converged, f"{level.label}: 수력 미수렴"
        out.append(
            (
                level.k_multiplier,
                result.secondary_shares_final_Lps[1],
                float(result.T_return_C[1][-1]),
            )
        )
    return out


B_RESPONSE = _b_response()


def test_criterion_4_all_values_finite() -> None:
    """연동을 판정하기 전에 값이 유한한지부터 본다."""
    for multiplier, share, temp in B_RESPONSE:
        assert math.isfinite(share) and math.isfinite(temp), f"K×{multiplier}: 비유한값"


def test_criterion_4_b_secondary_share_rises_monotonically() -> None:
    """기준 ㉣-1 — 누출이 커질수록 **B 의 2차측 몫이 엄격히 증가**한다.

    A 의 랙 저항이 커져 A 의 1차측 유량이 줄고, 총 2차측이 고정이므로 B 의 몫이
    늘어난다. 이것이 유량 경로 연동의 실체다(5-1 「공유 2차측 결합 방식」).
    """
    shares = [share for _m, share, _t in B_RESPONSE]
    assert all(b > a for a, b in zip(shares, shares[1:])), (
        f"B 의 2차측 몫이 단조 증가하지 않는다: {shares}"
    )


def test_criterion_4_b_return_temperature_falls_monotonically() -> None:
    """기준 ㉣-2 — 누출이 커질수록 **B 의 환수온도가 엄격히 감소**한다.

    방향이 세션 5 C7(정상상태)과 같아야 한다 — C7 은 +50% 에서 B 가 −0.012649 K
    였다. 부호가 뒤집히면 연동 경로가 잘못 물린 것이다.
    """
    temps = [temp for _m, _s, temp in B_RESPONSE]
    assert all(b < a for a, b in zip(temps, temps[1:])), (
        f"B 의 환수온도가 단조 감소하지 않는다: {temps}"
    )


def test_criterion_4_coupling_is_above_the_symmetric_case() -> None:
    """기준 ㉣-3 — **세션 5.5-B 의 양쪽 누출에서는 이 연동이 상쇄됐다.**

    같은 누출 수준을 두 CDU 에 걸면 B 의 2차측 몫이 거의 움직이지 않는다(대칭이라
    배분비가 그대로다). A 에만 걸면 몫이 실제로 이동한다. 그 차이가 **#34 가
    말한 상쇄**이고, 이 테스트가 그것이 사라졌음을 고정한다.
    """
    level = leak_levels()[-1]  # +50%
    both = _leak_on_a(level.k_multiplier)[0]
    normal = integrate_plant_load_step(_step_case(_leak_on_a(1.0)))
    a_only = integrate_plant_load_step(_step_case(_leak_on_a(level.k_multiplier)))
    symmetric = integrate_plant_load_step(_step_case((both, both)))

    base = normal.secondary_shares_final_Lps[1]
    shift_a_only = abs(a_only.secondary_shares_final_Lps[1] - base)
    shift_symmetric = abs(symmetric.secondary_shares_final_Lps[1] - base)
    assert shift_a_only > 100.0 * shift_symmetric, (
        "A 에만 건 누출의 배분 이동이 양쪽 누출과 크게 다르지 않다 — "
        f"A만 {shift_a_only:.3e} L/s · 양쪽 {shift_symmetric:.3e} L/s"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 데이터셋 스키마 — `leak_cdu_index` 열 (C4)
# ─────────────────────────────────────────────────────────────────────────────
def test_leak_cdu_index_column_exists() -> None:
    """`leak_cdu_index` 열이 스키마에 있다."""
    assert "leak_cdu_index" in column_names()


def _first(regime: str, config: str, multiplier: float) -> ScenarioSpec:
    return next(
        s
        for s in enumerate_specs()
        if s.regime == regime
        and s.cdu_config == config
        and s.leak_multiplier == multiplier
    )


@pytest.mark.parametrize("config", [CONFIG_SINGLE, CONFIG_DUAL_ASYMMETRIC])
def test_leak_cdu_index_is_blank_without_leak(config: str) -> None:
    """누출이 없으면 **빈 값**이다 — 이 스키마의 「해당 없음」 표기 규약이다."""
    for row in rows_for(_first("steady", config, 1.0)):
        assert row["leak_cdu_index"] == ""


@pytest.mark.parametrize("config", [CONFIG_SINGLE, CONFIG_DUAL_ASYMMETRIC])
def test_leak_cdu_index_names_the_leaking_cdu(config: str) -> None:
    """누출이 있으면 누출이 걸린 CDU 번호를 싣는다 — 행마다 같은 값이다."""
    for row in rows_for(_first("steady", config, 1.5)):
        assert row["leak_cdu_index"] == LEAK_CDU_INDEX


def test_dual_transient_row_count_is_unchanged() -> None:
    """저장 격자 변경이 **행 수를 바꾸지 않는다** — 규모는 사람이 정한 것이다.

    종전 최근접 선택도 201점을 골랐고, 지금은 같은 201점을 정확한 시각으로 낸다.
    """
    spec = _first("transient", CONFIG_DUAL_ASYMMETRIC, 1.5)
    rows = rows_for(spec)
    assert len(rows) == PLANT.cdu_count * len(
        storage_times_s(1.0, 1.0 / 30.0)  # 점 수는 t_end·tau 에 무관하다
    )
