"""세션 7.45 — 「샘」 게이트 판정 코드를 고정한다.

**이 파일은 게이트가 아니다.** 판정 기준은
`docs/session745-massloss-gate-criteria.md` 에 계산보다 먼저 적었고(커밋
`3001588`), 여기서는 그 기준을 코드가 **정의대로 셌는지**만 본다 —
통과 건수를 요구하지 않는다(결과를 시험에 박으면 기준을 결과에 맞추는 것이 된다).

판정 시험은 물리 모델을 부르지 않는다 — `results/cdu_dataset.csv` 를 읽는 시험
하나만 파일에 닿고, 나머지는 순수 함수를 작은 표로 확인한다.

**예외 하나**(세션 7.47): 맨 아래 퇴화 배치 항등성 시험은 **물리 모델을 부른다.**
「해당 없음」 규정(2-5-B)이 기대는 성질을 값이 아니라 성질로 고정하기 위해서다.

**모든 수치는 가정값 기반이며 실측이 아니다.**
"""

from __future__ import annotations

import pandas as pd
import pytest

from cdu_simul.dataset import (
    DEFAULT_OUTPUT_DIR,
    LEAK_MODEL_K_APPROX,
    LEAK_MODEL_MASSLOSS,
)
from cdu_simul.massloss import massloss_topologies
from cdu_simul.massloss_gate import (
    FAIL,
    NA,
    PASS,
    SIGNALS,
    criterion_a,
    criterion_b,
    criterion_c,
    format_report,
    read_dataset,
    signal_deltas,
)
from cdu_simul.massloss_thermal import solve_massloss_steady, thermal_cases
from cdu_simul.model import solve_cdu_steady_state


def _long(rows: list[dict[str, object]]) -> pd.DataFrame:
    """긴 표 최소 형태 — 판정 함수가 읽는 열만 담는다."""
    return pd.DataFrame(rows)


#: 신호 라벨 둘 — 수력 하나(「해당 없음」 규정이 걸린다)와 온도(걸리지 않는다).
_FLOW, _TEMP = SIGNALS[0].label, SIGNALS[-1].label

#: 배치 둘. 퇴화 배치만 「해당 없음」이다(기준 문서 2-5-B · 세션 7.47).
_DEGENERATE = {"residual_return_share": 0.0, "pump_sees_supply_flow": True}
_LIVE = {"residual_return_share": 0.5, "pump_sees_supply_flow": True}


def test_criterion_a_reads_expected_sign() -> None:
    """기대 부호와 같으면 통과 · 다르면 실패 · 퇴화 배치의 수력 신호는 해당 없음."""
    long = _long(
        [
            {"delta_abs": +1.0, "expected_sign": +1, "signal": _FLOW, **_LIVE},
            {"delta_abs": -1.0, "expected_sign": +1, "signal": _FLOW, **_LIVE},
            {"delta_abs": -1.0, "expected_sign": +1, "signal": _FLOW, **_DEGENERATE},
            {"delta_abs": -1.0, "expected_sign": -1, "signal": _FLOW, **_LIVE},
        ]
    )
    assert list(criterion_a(long)) == [PASS, FAIL, NA, PASS]


def test_not_applicable_does_not_cover_temperature() -> None:
    """퇴화 배치라도 **온도 ⑸ 는 판정한다** — 그 배치에서도 응답한다(7.46 D6).

    「막힘」 행처럼 배치 열이 빈 값이면 어느 신호도 해당 없음이 아니다.
    """
    long = _long(
        [
            {"delta_abs": -1.0, "expected_sign": -1, "signal": _TEMP, **_DEGENERATE},
            {
                "delta_abs": -1.0,
                "expected_sign": +1,
                "signal": _FLOW,
                "residual_return_share": None,
                "pump_sees_supply_flow": None,
            },
        ]
    )
    assert list(criterion_a(long)) == [PASS, FAIL]


def test_criterion_c_uses_the_unit_threshold() -> None:
    """잡음 임계는 단위마다 다르다 — 유량 1e-3 % · 양정 1e-4 mAq · 온도 1e-3 K."""
    long = _long(
        [
            {"level": 1, "unit": "%", "delta_judged": 2.0e-3, "signal": _FLOW, **_LIVE},
            {"level": 1, "unit": "%", "delta_judged": 5.0e-4, "signal": _FLOW, **_LIVE},
            {
                "level": 1,
                "unit": "mAq",
                "delta_judged": 2.0e-4,
                "signal": SIGNALS[1].label,
                **_LIVE,
            },
            {"level": 1, "unit": "K", "delta_judged": 5.0e-4, "signal": _TEMP, **_LIVE},
            {"level": 2, "unit": "K", "delta_judged": 9.9, "signal": _TEMP, **_LIVE},
        ]
    )
    verdicts = criterion_c(long, smallest=1)
    assert list(verdicts) == [PASS, FAIL, PASS, FAIL]
    assert 4 not in verdicts.index, "가장 작은 수준이 아닌 행은 대상이 아니다"


def test_criterion_b_is_strict_monotone_in_magnitude() -> None:
    """크기(절대값)가 엄격히 커져야 통과 — 퇴화 배치의 무리는 해당 없음."""
    long = _long(
        [
            {"g": "오름", "signal": _FLOW, "level": 1, "delta_abs": -1.0, **_LIVE},
            {"g": "오름", "signal": _FLOW, "level": 2, "delta_abs": -2.0, **_LIVE},
            {"g": "멈춤", "signal": _FLOW, "level": 1, "delta_abs": 3.0, **_LIVE},
            {"g": "멈춤", "signal": _FLOW, "level": 2, "delta_abs": 3.0, **_LIVE},
            {"g": "퇴화", "signal": _FLOW, "level": 1, "delta_abs": 1.0, **_DEGENERATE},
            {"g": "퇴화", "signal": _FLOW, "level": 2, "delta_abs": 2.0, **_DEGENERATE},
        ]
    )
    verdicts = criterion_b(long, ("g",))
    assert verdicts[("오름", _FLOW)] == PASS
    assert verdicts[("멈춤", _FLOW)] == FAIL
    assert verdicts[("퇴화", _FLOW)] == NA, "0 이 아니어도 배치로 갈린다"


# ─────────────────────────────────────────────────────────────────────────────
# CSV 를 읽는 시험 — 파일이 없으면 건너뛴다(데이터셋은 재생성 산출물이다)
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = DEFAULT_OUTPUT_DIR / "cdu_dataset.csv"


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    if not CSV_PATH.exists():
        pytest.skip(f"{CSV_PATH} 이 없다 — 데이터셋을 먼저 만든다")
    return read_dataset(CSV_PATH)


@pytest.mark.parametrize("leak_model", [LEAK_MODEL_MASSLOSS, LEAK_MODEL_K_APPROX])
def test_every_abnormal_row_finds_its_baseline(
    dataset: pd.DataFrame, leak_model: str
) -> None:
    """짝짓는 열 여덟이 이상 행마다 정상 행 하나를 정확히 집는다.

    집지 못하면 `signal_deltas` 가 예외를 던진다(조용히 넘어가지 않는다).
    """
    long = signal_deltas(dataset, leak_model)
    rows = (dataset["leak_model"] == leak_model) & (dataset["scenario_kind"] == "이상")
    assert len(long) == int(rows.sum()) * len(SIGNALS)
    assert long["delta_abs"].notna().all()


def test_verdict_counts_add_up(dataset: pd.DataFrame) -> None:
    """통과 + 실패 + 해당 없음 = 전수. 어느 짝도 판정에서 빠지지 않는다."""
    long = signal_deltas(dataset, LEAK_MODEL_MASSLOSS)
    verdict_a = criterion_a(long)
    assert len(verdict_a) == len(long)
    assert set(verdict_a.unique()) <= {PASS, FAIL, NA}

    smallest = float(long["level"].min())
    verdict_c = criterion_c(long, smallest)
    assert len(verdict_c) == int((long["level"] == smallest).sum())


def test_report_carries_the_assumption_notice(dataset: pd.DataFrame) -> None:
    """가정값 기반 표시와 「통과」의 뜻이 리포트에 남는다 (절대 규칙 11)."""
    report = format_report(dataset)
    assert "실측 아님" in report
    assert "실측 감지 가능성이 아니다" in report
    assert "energy balance 는 미판정" in report


# ─────────────────────────────────────────────────────────────────────────────
# 물리 모델을 부르는 시험 하나 — 「해당 없음」 규정이 기대는 성질 (세션 7.47 C6)
# ─────────────────────────────────────────────────────────────────────────────
#: 퇴화 배치 — `residual_return_share == 0.0` 이고 펌프가 공급유량을 본다.
_DEGENERATE_TOPOLOGY = next(
    t for t in massloss_topologies() if t.residual_return_share == 0.0
    and t.pump_sees_supply_flow
)

#: 기준선 경로 몫의 상한. **새 임계값이 아니라 세션 7.46 D3-가 의 실측**이다 —
#: 배치 6 전수에서 잰 최대(g=0.5 의 부동소수 잡음)이고 퇴화 배치는 정확히 0 이었다.
_BASELINE_PATH_SHARE_MAX_PERCENT = 1.147813e-14


@pytest.mark.parametrize(
    "case", [thermal_cases()[0], thermal_cases()[-1]], ids=lambda c: c.label
)
def test_degenerate_baseline_path_share_is_zero(case) -> None:  # type: ignore[no-untyped-def]
    """퇴화 배치에서 **기준선 경로 차의 몫이 0** 이다.

    이것을 지킨다: 세션 7.47 의 「해당 없음」 규정은 「퇴화 배치의 Δ 는 「샘」의
    수력 응답이 아니라 물성 온도가 남긴 몫」이라는 데 기댄다. 그 논거는 기준선이
    다른 solver 경로를 탄다는 한계(기준 문서 2-2)가 Δ 에 아무것도 얹지 않아야
    성립한다 — `Q_massloss = 0` 이면 두 경로가 같은 답을 내야 한다.

    깨지면 규정의 근거가 무너진다(Δ 에 경로 차가 섞여 있는 것이 된다).
    """
    massloss = solve_massloss_steady(case, 0.0, _DEGENERATE_TOPOLOGY)
    baseline = solve_cdu_steady_state(case)

    assert massloss.solver_converged and baseline.solver_converged
    share_percent = abs(
        (massloss.supply_flow_Lps - baseline.flow.total_flow_Lps)
        / baseline.flow.total_flow_Lps
        * 100.0
    )
    assert share_percent <= _BASELINE_PATH_SHARE_MAX_PERCENT
