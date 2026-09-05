"""세션 7.45 — 「샘」 게이트 판정 코드를 고정한다.

**이 파일은 게이트가 아니다.** 판정 기준은
`docs/session745-massloss-gate-criteria.md` 에 계산보다 먼저 적었고(커밋
`3001588`), 여기서는 그 기준을 코드가 **정의대로 셌는지**만 본다 —
통과 건수를 요구하지 않는다(결과를 시험에 박으면 기준을 결과에 맞추는 것이 된다).

물리 모델을 부르지 않는다. `results/cdu_dataset.csv` 를 읽는 시험 하나만
파일에 닿고, 나머지는 순수 함수를 작은 표로 확인한다.

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
from cdu_simul.massloss_gate import (
    FAIL,
    NA,
    PASS,
    SIGN_ZERO_TOL,
    SIGNALS,
    criterion_a,
    criterion_b,
    criterion_c,
    format_report,
    read_dataset,
    signal_deltas,
)


def _long(rows: list[dict[str, object]]) -> pd.DataFrame:
    """긴 표 최소 형태 — 판정 함수가 읽는 열만 담는다."""
    return pd.DataFrame(rows)


def test_criterion_a_reads_expected_sign() -> None:
    """기대 부호와 같으면 통과 · 다르면 실패 · 흔적이 없으면 해당 없음."""
    long = _long(
        [
            {"delta_abs": +1.0, "expected_sign": +1},
            {"delta_abs": -1.0, "expected_sign": +1},
            {"delta_abs": SIGN_ZERO_TOL / 2, "expected_sign": +1},
            {"delta_abs": -1.0, "expected_sign": -1},
        ]
    )
    assert list(criterion_a(long)) == [PASS, FAIL, NA, PASS]


def test_criterion_c_uses_the_unit_threshold() -> None:
    """잡음 임계는 단위마다 다르다 — 유량 1e-3 % · 양정 1e-4 mAq · 온도 1e-3 K."""
    long = _long(
        [
            {"level": 1, "unit": "%", "delta_abs": 1.0, "delta_judged": 2.0e-3},
            {"level": 1, "unit": "%", "delta_abs": 1.0, "delta_judged": 5.0e-4},
            {"level": 1, "unit": "mAq", "delta_abs": 2.0e-4, "delta_judged": 2.0e-4},
            {"level": 1, "unit": "K", "delta_abs": 5.0e-4, "delta_judged": 5.0e-4},
            {"level": 2, "unit": "K", "delta_abs": 9.9, "delta_judged": 9.9},
        ]
    )
    verdicts = criterion_c(long, smallest=1)
    assert list(verdicts) == [PASS, FAIL, PASS, FAIL]
    assert 4 not in verdicts.index, "가장 작은 수준이 아닌 행은 대상이 아니다"


def test_criterion_b_is_strict_monotone_in_magnitude() -> None:
    """크기(절대값)가 엄격히 커져야 통과 — 전부 0 이면 해당 없음."""
    long = _long(
        [
            {"g": "오름", "signal": "s", "level": 1, "delta_abs": -1.0},
            {"g": "오름", "signal": "s", "level": 2, "delta_abs": -2.0},
            {"g": "멈춤", "signal": "s", "level": 1, "delta_abs": 3.0},
            {"g": "멈춤", "signal": "s", "level": 2, "delta_abs": 3.0},
            {"g": "없음", "signal": "s", "level": 1, "delta_abs": 0.0},
            {"g": "없음", "signal": "s", "level": 2, "delta_abs": 0.0},
        ]
    )
    verdicts = criterion_b(long, ("g",))
    assert verdicts[("오름", "s")] == PASS
    assert verdicts[("멈춤", "s")] == FAIL
    assert verdicts[("없음", "s")] == NA


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
