"""세션 7.45 — 「샘」 게이트 판정 코드를 고정한다.

**「샘」의 A·B·C 는 세션 7.78 부터 게이트다** — 사람이 2026-09-08 에 통과율
100 % 로 올렸고, 자리는 `CLAUDE.md` 게이트 표 세션 4 줄이다. 판정 기준은
`docs/session745-massloss-gate-criteria.md` 에 계산보다 먼저 적었다(커밋
`3001588`).

**이 파일이 그 게이트를 판정한다**(세션 7.79 부터). 두 가지를 한다.

⑴ 기준을 코드가 **정의대로 셌는지** — 작은 표로 판정식과 여유 식을 고정한다.
   건수를 박지 않는다(결과를 시험에 박으면 기준을 결과에 맞추는 것이 된다).
⑵ **게이트 자체** — 「이상 기구를 진 CDU」 전수에서 A·B·C 통과율이 100 % 인가
   (`test_massloss_gate_passes_on_every_leak_cdu_pair`). 합격선도 모집단도
   `CLAUDE.md` 게이트 표 세션 4 줄의 문언 그대로이고, 이 파일이 새로 정하지
   않는다. 모집단은 `massloss_gate.leak_side` 로 고르고 — 리포트가 쓰는 바로 그
   함수다 — 「해당 없음」은 분모에서 뺀다. 건수는 시험이 스스로 센다.

사람이 읽는 판정 전문은 여전히 `python -m cdu_simul.massloss_gate` 가 낸다.

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
    NOISE_THRESHOLD,
    PAIR_COLUMNS,
    PAIR_IDENTIFIER,
    PASS,
    SIGNALS,
    TOPOLOGY_COLUMNS,
    criterion_a,
    criterion_b,
    criterion_b_margin,
    criterion_c,
    format_report,
    leak_side,
    pair_table_ac,
    pair_table_b,
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
# 여유 식 셋 — 정의를 작은 표로 고정한다 (미해결 #89 · 세션 7.79)
#
# 여유는 **임계까지의 거리**이지 새 임계가 아니다. 기대값을 데이터셋에서 가져오지
# 않는다 — 전부 여유 식과 판정식의 정의에서 손으로 세운 수다.
# ─────────────────────────────────────────────────────────────────────────────
def test_criterion_b_margin_is_the_smallest_neighbour_gap() -> None:
    """여유 = 수준 오름차순 |Δ| 의 **이웃 차 최소**. 원단위 · 부호 있는 수다.

    통과 조건이 「이웃 차가 전부 > 0」이므로 그 최소가 임계까지의 거리다.
    """
    long = _long(
        [
            {"g": "오름", "signal": _FLOW, "level": 1, "delta_abs": 1.0, **_LIVE},
            {"g": "오름", "signal": _FLOW, "level": 2, "delta_abs": 3.0, **_LIVE},
            {"g": "오름", "signal": _FLOW, "level": 3, "delta_abs": 4.0, **_LIVE},
            {"g": "내림", "signal": _FLOW, "level": 1, "delta_abs": 4.0, **_LIVE},
            {"g": "내림", "signal": _FLOW, "level": 2, "delta_abs": 1.0, **_LIVE},
        ]
    )
    margin = criterion_b_margin(long, ("g",))
    assert margin[("오름", _FLOW)] == pytest.approx(1.0), "min(3-1, 4-3)"
    assert margin[("내림", _FLOW)] == pytest.approx(-3.0), "떨어지면 음수다"


def test_criterion_b_margin_sign_decides_the_verdict() -> None:
    """여유의 부호가 곧 기준 B 의 판정이다 — 경계는 0 이고 0 은 통과가 아니다.

    · 「멈춤」 — 이웃 차가 0 이면 여유도 0 이고, 「엄격」 단조라 실패다.
      **새 임계가 아니다**: 0 은 `criterion_b` 의 `>` 에서 그대로 나온다.
    · 「음수」 — 여유는 |Δ| 로 잰다. Δ 가 내려가도 크기가 커지면 양수다.
    · 「퇴화」 — 배치로 갈린 「해당 없음」 무리는 여유의 부호와 무관하다.
    """
    long = _long(
        [
            {"g": "멈춤", "signal": _FLOW, "level": 1, "delta_abs": 3.0, **_LIVE},
            {"g": "멈춤", "signal": _FLOW, "level": 2, "delta_abs": 3.0, **_LIVE},
            {"g": "음수", "signal": _FLOW, "level": 1, "delta_abs": -1.0, **_LIVE},
            {"g": "음수", "signal": _FLOW, "level": 2, "delta_abs": -2.0, **_LIVE},
            {"g": "퇴화", "signal": _FLOW, "level": 1, "delta_abs": 1.0, **_DEGENERATE},
            {"g": "퇴화", "signal": _FLOW, "level": 2, "delta_abs": 2.0, **_DEGENERATE},
        ]
    )
    margin = criterion_b_margin(long, ("g",))
    verdict = criterion_b(long, ("g",))

    assert margin[("멈춤", _FLOW)] == 0.0
    assert verdict[("멈춤", _FLOW)] == FAIL, "경계값 0 은 통과쪽이 아니다"
    assert margin[("음수", _FLOW)] == pytest.approx(1.0)
    assert verdict[("음수", _FLOW)] == PASS

    judged = verdict[verdict != NA]
    assert ((margin[judged.index] > 0.0) == (judged == PASS)).all()
    assert verdict[("퇴화", _FLOW)] == NA and margin[("퇴화", _FLOW)] > 0.0


def _ac_long() -> pd.DataFrame:
    """기준 A·C 짝표용 최소 표 — 수준 둘 · 단위 둘 · 퇴화 배치 하나."""
    return _long(
        [
            # 통과/통과 — 부호가 맞고 Δ 가 잡음(1e-3 %) 위에 있다.
            {
                "scenario_id": 1, "cdu_index": 0, "signal": _FLOW, "leak_cdu": True,
                "level": 1.0, "unit": "%", "delta_abs": +2.0, "expected_sign": +1,
                "delta_judged": 2.0e-3, **_LIVE,
            },
            # 실패/실패 — 부호가 어긋나고 Δ 가 잡음 아래다.
            {
                "scenario_id": 2, "cdu_index": 0, "signal": _FLOW, "leak_cdu": True,
                "level": 1.0, "unit": "%", "delta_abs": -2.0, "expected_sign": +1,
                "delta_judged": -5.0e-4, **_LIVE,
            },
            # 가장 작은 수준이 아니라 C 에 실리지 않는다.
            {
                "scenario_id": 3, "cdu_index": 0, "signal": _TEMP, "leak_cdu": False,
                "level": 2.0, "unit": "K", "delta_abs": -1.0, "expected_sign": -1,
                "delta_judged": -1.0, **_LIVE,
            },
            # 퇴화 배치의 수력 신호 — A·C 둘 다 해당 없음.
            {
                "scenario_id": 4, "cdu_index": 0, "signal": _FLOW, "leak_cdu": True,
                "level": 1.0, "unit": "%", "delta_abs": -1.0, "expected_sign": +1,
                "delta_judged": -1.0, **_DEGENERATE,
            },
        ]
    )


def test_pair_table_ac_margin_is_the_distance_to_its_own_threshold() -> None:
    """A 는 원단위 · 임계 0, C 는 판정단위 · 임계 `NOISE_THRESHOLD[unit]`.

    C 는 **가장 작은 수준의 짝만** 싣는다 — `criterion_c` 와 같은 모집단이다.
    """
    long = _ac_long()
    table = pair_table_ac(long)
    a = table[table["criterion"] == "A"]
    c = table[table["criterion"] == "C"]

    assert len(a) == len(long)
    assert len(c) == int((long["level"] == long["level"].min()).sum())
    assert set(c["scenario_id"]) == {1, 2, 4}, "수준 2 인 3번 짝은 대상이 아니다"

    assert (a["threshold"] == 0.0).all() and (a["margin_unit"] == "원단위").all()
    assert list(a["margin"]) == pytest.approx([+2.0, -2.0, +1.0, -1.0])

    assert (c["margin_unit"] == "판정단위").all()
    assert list(c["threshold"]) == pytest.approx([NOISE_THRESHOLD["%"]] * 3)
    assert list(c["margin"]) == pytest.approx([1.0e-3, -5.0e-4, 1.0 - 1.0e-3])


def test_pair_table_ac_margin_sign_decides_the_verdict() -> None:
    """여유가 양수면 통과쪽이다 — 「해당 없음」 짝은 판정에서 빠진다.

    경계(여유 0)는 이 표에 없다. 기준 C 는 `|Δ| > threshold` 라 여유 0 이 실패로
    떨어지지만, 기준 A 는 Δ 를 `> 0` 으로만 읽어 Δ = 0 에서 기대 부호가 −1 이면
    여유 0 인 채 통과가 된다 — 그 한 점에서만 「여유 > 0 = 통과」가 성립하지
    않는다. 그래서 여기서는 여유 ≠ 0 인 짝만 본다.
    """
    table = pair_table_ac(_ac_long())
    judged = table[table["verdict"] != NA]
    assert len(judged) == len(table) - 2, "퇴화 배치의 A·C 짝 둘이 빠진다"
    assert (judged["margin"] != 0.0).all()
    assert ((judged["margin"] > 0.0) == (judged["verdict"] == PASS)).all()


def test_pair_table_b_carries_the_group_verdict_and_its_margin() -> None:
    """(무리 · 신호) 짝마다 한 줄이고, 판정·여유가 판정식이 낸 것 그대로다."""
    long = _long(
        [
            {"g": "오름", "signal": _FLOW, "level": 1, "delta_abs": -1.0, **_LIVE},
            {"g": "오름", "signal": _FLOW, "level": 2, "delta_abs": -2.0, **_LIVE},
            {"g": "멈춤", "signal": _TEMP, "level": 1, "delta_abs": 3.0, **_LIVE},
            {"g": "멈춤", "signal": _TEMP, "level": 2, "delta_abs": 3.0, **_LIVE},
        ]
    )
    table = pair_table_b(long, ("g",)).set_index(["g", "signal"])
    verdict = criterion_b(long, ("g",))
    margin = criterion_b_margin(long, ("g",))

    assert len(table) == len(verdict)
    assert (table["criterion"] == "B").all()
    assert (table["threshold"] == 0.0).all() and (table["margin_unit"] == "원단위").all()
    assert (table["verdict"] == verdict[table.index]).all()
    assert table["margin"].to_numpy() == pytest.approx(margin[table.index].to_numpy())


# ─────────────────────────────────────────────────────────────────────────────
# CSV 를 읽는 시험 — 파일이 없으면 건너뛴다(데이터셋은 재생성 산출물이다)
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = DEFAULT_OUTPUT_DIR / "cdu_dataset.csv"


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    if not CSV_PATH.exists():
        pytest.skip(f"{CSV_PATH} 이 없다 — 데이터셋을 먼저 만든다")
    return read_dataset(CSV_PATH)


@pytest.mark.parametrize("anomaly_mechanism", [LEAK_MODEL_MASSLOSS, LEAK_MODEL_K_APPROX])
def test_every_abnormal_row_finds_its_baseline(
    dataset: pd.DataFrame, anomaly_mechanism: str
) -> None:
    """짝짓는 열 여덟이 이상 행마다 정상 행 하나를 정확히 집는다.

    집지 못하면 `signal_deltas` 가 예외를 던진다(조용히 넘어가지 않는다).
    """
    long = signal_deltas(dataset, anomaly_mechanism)
    rows = (dataset["anomaly_mechanism"] == anomaly_mechanism) & (
        dataset["scenario_kind"] == "이상"
    )
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


#: 실패 메시지에 식별자를 싣는 짝 수 상한. 깨졌을 때 알아야 하는 것은 「어디를
#: 볼지」뿐이고 전수는 짝표(`write_pair_tables`)가 이미 갖고 있으므로 앞 몇 개면
#: 족하다. 3 인 까닭은 기준 B 의 무리 키가 12축이라 한 짝이 한 줄을 넘기 때문이다
#: — 세 개면 A·C 도 B 도 pytest 출력 안에서 읽힌다. 나머지는 건수로만 낸다.
FAIL_SAMPLE = 3


def _failed_pairs(verdicts: pd.Series, long: pd.DataFrame) -> str:  # type: ignore[type-arg]
    """실패한 짝의 식별자 — 몇 건 중 몇 개를 보였는지 함께 낸다 (세션 7.82).

    · A·C — 판정 단위가 (행 · 신호) 짝이라 `PAIR_IDENTIFIER` 셋으로 유일하다.
    · B — 판정 단위가 (무리 · 신호) 짝이고 한 무리가 크기 4수준에 걸쳐 있어
      `scenario_id` 로 못 가리킨다. 무리 키(MultiIndex) 를 그대로 싣는다
      (`pair_table_b` 와 같은 축이다).

    **판정에 끼어들지 않는다** — 실패했을 때만 불린다(assert 메시지는 조건이
    거짓일 때만 계산된다).
    """
    failed = verdicts[verdicts == FAIL]
    shown = failed.index[:FAIL_SAMPLE]
    if isinstance(failed.index, pd.MultiIndex):
        labels = [
            " · ".join(f"{name}={value}" for name, value in zip(failed.index.names, key))
            for key in shown
        ]
    else:
        rows = long.loc[shown, list(PAIR_IDENTIFIER)]
        labels = [
            " · ".join(f"{column}={row[column]}" for column in PAIR_IDENTIFIER)
            for _, row in rows.iterrows()
        ]
    return f"실패 {len(failed):,}짝 중 앞 {len(labels)}개 — " + " | ".join(labels)


def test_massloss_gate_passes_on_every_leak_cdu_pair(dataset: pd.DataFrame) -> None:
    """**게이트** — 「샘」의 A·B·C 가 이상 기구를 진 CDU 전수에서 통과율 100 %.

    합격선·모집단은 `CLAUDE.md` 게이트 표 세션 4 줄의 문언 그대로다(사람이
    2026-09-08 에 정했다). 이 시험이 새로 정하는 것은 없다.

    · 모집단 — `leak_side` 가 고른다. 리포트의 통과율이 쓰는 바로 그 함수라
      둘이 갈리지 않는다. 이웃 CDU 는 여기 들지 않는다(기준 문서 2-4).
    · 「해당 없음」 — 분모에서 뺀다(세션 7.47 규정). 건수는 시험이 스스로 센다.

    **깨지면 기대값을 고치지 않는다.** 기준 B 의 판정된 짝(이상 기구를 진 CDU ·
    「해당 없음」 제외) 가운데 `criterion_b_margin` 최소는 3.654e-04 이고 신호는
    ⑶ 주입랙 통과유량 · ⑷ 타 랙 유량이다(세션 7.119 · 7.123 이
    `results/cdu_dataset.csv` 로 잰 값) — 깨진 것이 게이트가 일한 것인지 먼저
    가른다.
    """
    long = signal_deltas(dataset, LEAK_MODEL_MASSLOSS)
    verdicts = {
        "A": criterion_a(long),
        "B": criterion_b(long, (*PAIR_COLUMNS, *TOPOLOGY_COLUMNS)),
        "C": criterion_c(long, float(long["level"].min())),
    }
    for name, verdict in verdicts.items():
        leak = leak_side(verdict, long)
        decided = leak[leak != NA]
        assert len(decided) > 0, f"기준 {name} — 판정된 짝이 0 이면 통과가 아니다"
        assert (decided == PASS).all(), (
            f"기준 {name} — 실패 {int((decided == FAIL).sum()):,}짝 / "
            f"판정 {len(decided):,}짝 (해당 없음 {int((leak == NA).sum()):,}짝) · "
            f"{_failed_pairs(decided, long)}"
        )


def test_report_carries_the_assumption_notice(dataset: pd.DataFrame) -> None:
    """가정값 기반 표시와 「통과」의 뜻이 리포트에 남는다 (절대 규칙 11)."""
    report = format_report(dataset)
    assert "실측 아님" in report
    assert "실측 감지 가능성이 아니다" in report
    # 세션 7.55 — 「미판정」 문언을 7.53 의 판정 결과에 맞췄다. 이 assert 가 지키는
    # 것은 **리포트가 balance 상태를 밝힌다**이지 그 상태가 「미판정」이라는 것이
    # 아니다(세션 7.54 가 머리 docstring 을 먼저 같은 뜻으로 고쳤다).
    assert "energy balance 는 이 판정기가 재지 않는다" in report


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
