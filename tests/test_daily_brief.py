"""`daily_brief.py` 파싱 규칙 시험 (세션 7.94 · 미해결 #97).

물리 모델과 무관하다 — 6장 feasibility 판정이 아니라 **브리핑 스크립트의
문서 파싱 규칙**만 고정한다. 입력은 붙박이 가짜 문서 하나이고, 저장소
`PROCEED.md` 의 건수·문구·날짜에 기대지 않는다(그 문서는 판마다 늘어난다).
마지막 한 건만 실 파일을 쓰는 연기시험이고, 파일이 없으면 skip 한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import daily_brief

#: 실 `PROCEED.md` 대신 쓰는 붙박이 문서. 판 제목 수준을 일부러 섞어 두었다 —
#: `### 세션 9.0`(흔한 쪽)과 `## 세션 9.1`(세션 7.91 이 문제 삼은 쪽).
FAKE_DOC = """# 제목

## 현재 상태

| 세션 | 단계 | 게이트 통과 여부 | 다음 세션 첫 작업 |
|---|---|---|---|
| 9.0 | 앞 판 | **게이트 없음** | 앞 일 |
| 9.1 | 뒤 판 | **게이트 없음** | 첫 일 (가 · 나) · 둘째 일 |

## 세션 로그

### 세션 9.0 · 첫 판
앞 판 본문.

## 세션 9.1 · 둘째 판
뒤 판 본문.

## 미해결 목록

| # | 한 줄 | 크기 | 영향 |
|---|---|---|---|
| 7 | **열린 것** | S | 영향 |
| 8 | **[닫힘 · 세션 9.0]** 닫힌 것 | S | 영향 |
"""

FAKE_LINES = FAKE_DOC.splitlines()


# ⓐ 판 제목을 **수준이 아니라 꼴**로 가린다 — 절 제목과 하위 제목은 물지 않는다.
@pytest.mark.parametrize(
    ("line", "is_heading"),
    [
        ("## 세션 로그", False),  # 절 제목 — `세션` 뒤가 숫자가 아니다
        ("#### 세션 5.7 · 하위 제목", False),  # `#` 넷 — 판 제목이 아니다
        ("### 세션 7.91 · 어떤 판", True),
    ],
)
def test_session_heading_shape(line: str, is_heading: bool) -> None:
    assert bool(daily_brief._SESSION_HEADING.match(line)) is is_heading


# ⓑ 마지막 판 제목이 `## ` 수준이어도 그것을 고른다(세션 7.91 이 고친 자리).
def test_last_session_block_picks_hash_two_heading() -> None:
    block = daily_brief._last_session_block(FAKE_LINES)
    assert block[0] == "## 세션 9.1 · 둘째 판"


# ⓒ 로그 절 안의 `## 세션 N` 은 절 경계가 아니다 — 거기서 끊기면 뒤 판이 사라진다.
def test_section_not_cut_by_session_heading() -> None:
    log = daily_brief._section(FAKE_LINES, daily_brief.SESSION_LOG_HEADING)
    assert "뒤 판 본문." in log
    assert "## 미해결 목록" not in log  # 다음 절에서는 끊긴다


# ⓓ 첫 ` · ` 앞까지 자르되, 괄호 안의 ` · ` 는 경계로 보지 않는다.
def test_first_segment_cuts_at_first_separator() -> None:
    assert daily_brief._first_segment("가 · 나 · 다") == "가"


def test_first_segment_ignores_separator_inside_parens() -> None:
    assert daily_brief._first_segment("첫 일 (가 · 나) · 둘째 일") == "첫 일 (가 · 나)"


# ⓔ 표출 단계에서 판 제목의 우물 정 글자가 「현재:」 줄로 새지 않는다.
def test_web_brief_current_line_has_no_hash() -> None:
    current = next(
        line for line in daily_brief.build_web_brief(FAKE_DOC).splitlines()
        if line.startswith("현재:")
    )
    assert current == "현재: 세션 9.1 · 둘째 판"


# D4 — 실 `PROCEED.md` 연기시험. 내용·건수·문구는 비교하지 않는다.
def test_web_brief_runs_on_real_proceed() -> None:
    proceed = Path(daily_brief.__file__).resolve().parent / "PROCEED.md"
    if not proceed.exists():
        pytest.skip("PROCEED.md 가 없다")
    out = daily_brief.build_web_brief(proceed.read_text(encoding="utf-8")).splitlines()
    current = next(line for line in out if line.startswith("현재:"))
    assert current.removeprefix("현재:").strip()
    head = next(i for i, line in enumerate(out) if line.startswith("다음 첫 작업"))
    assert out[head + 1].strip()
