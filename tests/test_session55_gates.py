"""세션 5.5 게이트 — 시나리오 간 상태 이월 없음 · 전 행 표기.

CLAUDE.md 절대 규칙 10: 사람이 눈으로 보지 않고 **테스트가 판정한다.**

`CLAUDE.md` 게이트 표의 세션 5.5 항목:

    "정상/이상/누출 다수 케이스 반복 실행 시 **시나리오 간 상태 이월 없음**
     (collaboration.md 결함유형 ④) · 전 행에 가정치 출처·"가정값 기반 — 실측
     아님" 표기"

═══════════════════════════════════════════════════════════════════════════════
**세션 5.5 게이트 판정 기준 (선기재)**

**기준 A — 순서 무관성.** 시나리오 목록을 **뒤집어** 다시 돌렸을 때 각 시나리오의
행이 원래 순서에서 나온 행과 **완전히 같아야** 한다. 이월은 순서 의존으로
나타나므로 이 검사가 정확히 그것을 잡는다. 세션 3-B 이후 여러 판에서 써 온
"재실행 시 바이트 단위 동일" 기법을 그대로 쓴다.

**기준 B — 단독 재현.** 임의의 시나리오 하나를 **혼자** 돌린 결과가 배치 안에서
나온 그 행과 같아야 한다. A 가 순서 의존을 잡는다면 B 는 **배치라는 문맥 자체가
결과에 남는지**를 잡는다. 둘은 서로를 보완한다.

**기준 C — 정상 복귀.** 누출 시나리오를 돌린 **뒤** 정상 시나리오를 돌렸을 때,
그 정상 행이 정상을 먼저 돌렸을 때의 행과 같아야 한다. 누출이 남긴 상태가
정상값을 오염시키지 않음을 직접 본다.

**기준 D — 전 행 표기**(게이트의 두 번째 절). 모든 행에 다음이 **비어 있지 않게**
들어 있어야 한다: `assumption_tag`("가정값 기반 — 실측 아님" 포함) ·
`signal_sign_caveat` · `transient_caveat` · `total_flow_caveat` · `cr_caveat` ·
solver 플래그 4종의 자리. **육안 확인이 아니라 프로그램으로 본다.**

**새 허용오차를 만들지 않는다** — A·B·C 는 전부 `==` 로 본다. 부동소수점 비교에
허용오차를 두면 "작은 이월"이 통과해 버린다.

**표본 방식**: 전량(1,792 시나리오)을 돌리면 36분이 걸려 테스트로 부적당하다.
`SAMPLE_STRIDE` 간격으로 고르게 뽑아 정상상태·전이·세 CDU 구성·네 누출 수준이
모두 포함되도록 한다 — 아래 `_sample_specs` 가 그 포함을 **테스트로 확인**한다.

**이 게이트가 판정하지 않는 것**
· 데이터셋의 물리가 맞는가 — 6장 기준은 각자의 자리에서 이미 판정했다
· 표기 문구의 내용이 옳은가 — 있는지만 본다. 내용은 사람이 읽는다
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from cdu_simul.dataset import (
    CONFIG_DUAL_ASYMMETRIC,
    CONFIG_DUAL_SYMMETRIC,
    CONFIG_SINGLE,
    MECHANISM_MASSLOSS,
    MECHANISM_NONE,
    PROVENANCE_ROW,
    ScenarioSpec,
    column_names,
    enumerate_specs,
    rows_for,
)
from cdu_simul.dataset_plan import MASSLOSS_PROVENANCE

#: 표본 간격 — 1,792 시나리오에서 고르게 뽑는다.
#: 61 은 시나리오 수의 약수가 아니라 목록을 고르게 훑는다(약수면 특정 축에 몰린다).
SAMPLE_STRIDE: int = 61

#: 전 행에 있어야 하는 표기 열 (기준 D).
REQUIRED_PROVENANCE: tuple[str, ...] = tuple(PROVENANCE_ROW)

#: 전 행에 자리가 있어야 하는 solver 플래그 열 (절대 규칙 5).
REQUIRED_SOLVER_COLUMNS: tuple[str, ...] = (
    "hydraulic_solver_ier",
    "thermal_solver_converged",
    "plant_solver_ier",
    "integrator_success",
)


def _sample_specs() -> list[ScenarioSpec]:
    specs = enumerate_specs()
    return specs[::SAMPLE_STRIDE]


SAMPLE = _sample_specs()


def test_sample_covers_every_axis() -> None:
    """표본이 축을 빠짐없이 덮는지 — 표본이 게이트를 무력화하지 않게.

    표본이 한쪽에 몰리면 A·B·C 를 통과해도 아무것도 보장하지 못한다.
    """
    assert len(SAMPLE) >= 20, f"표본 {len(SAMPLE)}개는 너무 적다"
    assert {s.regime for s in SAMPLE} == {"steady", "transient"}
    assert {s.cdu_config for s in SAMPLE} == {
        CONFIG_SINGLE,
        CONFIG_DUAL_SYMMETRIC,
        CONFIG_DUAL_ASYMMETRIC,
    }
    assert {s.leak_multiplier for s in SAMPLE} == {1.0, 1.05, 1.2, 1.5}
    assert len({s.load_percent for s in SAMPLE}) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 기준 A — 순서 무관성
# ─────────────────────────────────────────────────────────────────────────────
def test_criterion_a_order_independence() -> None:
    """기준 A — 목록을 뒤집어 돌려도 각 시나리오의 행이 완전히 같다.

    이월이 있으면 앞 시나리오가 달라진 만큼 결과가 흔들린다. `==` 로 본다.
    """
    forward = {spec.scenario_id: rows_for(spec) for spec in SAMPLE}
    reverse = {spec.scenario_id: rows_for(spec) for spec in reversed(SAMPLE)}

    assert forward.keys() == reverse.keys()
    for scenario_id, rows in forward.items():
        assert rows == reverse[scenario_id], (
            f"{scenario_id}: 순서를 바꾸니 행이 달라졌다 — 상태 이월이 있다"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 기준 B — 단독 재현
# ─────────────────────────────────────────────────────────────────────────────
def _build_sample_batch() -> dict[str, list[dict[str, object]]]:
    """SAMPLE 전량을 한 번 돌린 배치 — 기준 B 파라미터 10건이 이것 하나를 나눠 쓴다.

    파라미터마다 다시 만들면 같은 배치를 10번 만든다(세션 7.37 실측 747 s · #58).
    """
    return {spec.scenario_id: rows_for(spec) for spec in SAMPLE}


@pytest.fixture(scope="module")
def sample_batch() -> dict[str, list[dict[str, object]]]:
    """공유 배치 — 시험은 **복제본**을 받는다.

    이 파일이 재는 것이 상태 이월이므로, 공유 자체가 이월을 만들면 게이트가
    스스로 무너진다. 시험마다 `deepcopy` 로 끊는다 — 실측 복제 0.16 s 대
    재생성 60.00 s (세션 7.38).
    """
    return _build_sample_batch()


@pytest.mark.parametrize(
    "index", range(0, len(SAMPLE), max(1, len(SAMPLE) // 8)), ids=str
)
def test_criterion_b_standalone_reproduction(
    index: int, sample_batch: dict[str, list[dict[str, object]]]
) -> None:
    """기준 B — 시나리오 하나를 혼자 돌린 결과가 배치 행과 같다.

    배치라는 문맥이 결과에 남지 않음을 직접 본다. 배치를 한 번 돌린 뒤 같은
    시나리오를 **새 프로세스가 아니라 같은 프로세스에서 단독으로** 돌린다 —
    모듈 수준 상태가 남아 있다면 그것이 여기서 드러난다.
    """
    batch = deepcopy(sample_batch)
    target = SAMPLE[index]
    standalone = rows_for(target)
    assert standalone == batch[target.scenario_id], (
        f"{target.scenario_id}: 단독 실행이 배치 행과 다르다 — 배치 문맥이 샌다"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 기준 C — 정상 복귀
# ─────────────────────────────────────────────────────────────────────────────
def test_criterion_c_normal_recovers_after_leak() -> None:
    """기준 C — 누출을 돌린 뒤의 정상 시나리오가 오염되지 않는다.

    누출 → 정상 순서와 정상 단독의 결과를 비교한다. 누출이 solver 초기값이나
    모듈 상태에 남으면 여기서 드러난다.
    """
    specs = enumerate_specs()
    normal = next(s for s in specs if s.leak_multiplier == 1.0)
    leaks = [
        s
        for s in specs
        if s.leak_multiplier != 1.0 and s.regime == normal.regime
    ][:3]
    assert leaks, "누출 시나리오를 찾지 못했다"

    alone = rows_for(normal)
    for leak in leaks:
        rows_for(leak)
    after_leaks = rows_for(normal)

    assert after_leaks == alone, (
        f"{normal.scenario_id}: 누출을 돌린 뒤 정상값이 달라졌다"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 기준 D — 전 행 표기
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("spec", SAMPLE, ids=lambda s: s.scenario_id)
def test_criterion_d_every_row_carries_provenance(spec: ScenarioSpec) -> None:
    """기준 D — 모든 행에 출처·한계 표기와 solver 플래그 자리가 있다.

    `CLAUDE.md` 세션 5.5 게이트가 요구하는 "전 행에" 를 프로그램으로 확인한다.
    표기가 빠진 행이 하나라도 있으면 그 행으로 학습한 모델이 한계를 모르게 된다
    (절대 규칙 11).
    """
    rows = rows_for(spec)
    assert rows, f"{spec.scenario_id}: 행이 하나도 없다"
    columns = set(column_names())

    for row_index, row in enumerate(rows):
        for name in REQUIRED_PROVENANCE:
            assert name in columns, f"{name} 이 스키마에 없다"
            value = row.get(name)
            assert isinstance(value, str) and value.strip(), (
                f"{spec.scenario_id} 행 {row_index}: {name} 이 비어 있다"
            )
        assert "실측 아님" in str(row["assumption_tag"]), (
            f"{spec.scenario_id} 행 {row_index}: 「실측 아님」 표기가 없다"
        )
        for name in REQUIRED_SOLVER_COLUMNS:
            assert name in row, (
                f"{spec.scenario_id} 행 {row_index}: solver 플래그 {name} 자리가 없다"
            )


def test_criterion_d_holds_for_massloss_branch() -> None:
    """기준 D 가 **기구별로 갈린 문언에도** 성립한다 [세션 7.35].

    「샘」 문언은 「막힘」 문언과 다른 문자열이므로, 갈라 둔 쪽이 비어 있거나
    키가 빠지면 그 행만 표기 없이 나간다 — 기준 D 가 「막힘」 행만 보고 통과할
    수 있다. 한 행으로 그 갈래를 고정한다.

    **이 행의 수치는 「샘」 물리가 아니다.** K 배수 1.0 의 정상 해에 라벨과
    표기만 「샘」으로 붙인 것이다 — 「샘」 행을 실제로 얹는 것은 다음 판이다.
    여기서 확인하는 것은 라벨 축이 K 배수와 독립인지, 그리고 갈린 문언이
    전 열에 실리는지 둘뿐이다.
    """
    blockage = next(
        s
        for s in enumerate_specs()
        if s.regime == "steady"
        and s.cdu_config == CONFIG_SINGLE
        and s.mechanism == MECHANISM_NONE
    )
    massloss = replace(blockage, mechanism=MECHANISM_MASSLOSS)

    # 라벨 축이 K 배수에서 떨어졌다 — 같은 K=1.0 인데 라벨이 갈린다.
    assert blockage.scenario_kind == "정상"
    assert massloss.scenario_kind == "이상", "「샘」 행이 「정상」으로 라벨된다"
    assert massloss.leak_multiplier == 1.0

    row = rows_for(massloss)[0]
    for name in REQUIRED_PROVENANCE:
        value = row.get(name)
        assert isinstance(value, str) and value.strip(), f"{name} 이 비어 있다"
    assert "실측 아님" in str(row["assumption_tag"])

    # 갈아 끼운 셋만 갈리고 나머지 둘은 「막힘」 문언 그대로다.
    for name, text in MASSLOSS_PROVENANCE.items():
        assert row[name] == text, f"{name} 에 「막힘」 문언이 실렸다"
    for name in set(REQUIRED_PROVENANCE) - set(MASSLOSS_PROVENANCE):
        assert row[name] == PROVENANCE_ROW[name]


def test_failed_solver_rows_are_not_dropped() -> None:
    """solver 실패 행을 버리지 않는 구조인지 — 행 수가 solver 결과와 무관하다.

    버리면 데이터셋이 성공만 담아 편향된다(5-1 「데이터셋 스키마 규약」).
    행 생성 경로에 실패를 걸러내는 분기가 없음을, 행 수가 **오직 시나리오 종류로만**
    정해진다는 사실로 고정한다.
    """
    for spec in SAMPLE:
        rows = rows_for(spec)
        expected = 1 if spec.cdu_config == CONFIG_SINGLE else 2
        if spec.regime == "transient":
            expected *= len(rows) // expected  # 시각 표본 수는 적분기가 정한다
            assert len(rows) == expected
        else:
            assert len(rows) == expected, (
                f"{spec.scenario_id}: 정상상태 행이 {len(rows)}개다 — "
                "실패 행이 걸러졌을 수 있다"
            )
