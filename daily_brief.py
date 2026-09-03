"""대화 시작 시 붙일 브리핑을 PROCEED.md 에서 뽑아 stdout 으로 출력한다.

`collaboration.md` 「하루의 모양」: 아침에 `git pull` 후 이 출력을 새 대화의 첫
메시지에 붙인다.

**기본은 축약형이다**(세션 5.5-C). 자르는 우선순위를 뒤집었다 — 세션 5까지의
브리핑은 「현재 상태」 표와 마지막 세션 로그를 통째로 싣고 **열린 미해결만
88자에서 잘랐다**. 그 결과 미해결 #34 의 선택지 (가)·(나) 처럼 *사람이 정해야 할
것* 이 브리핑에서 사라졌다. 지금은 반대다::

    1. 한 줄 현재 위치 + 다음에 할 일
    2. 「현재 상태」 표 — 최근 2판만 상세, 나머지는 한 줄
    3. 마지막 세션 로그의 「다음이 알아야 할 것」 절만
    4. **열린 미해결 전문** — 어떤 경우에도 자르지 않는다
    5. 참조 한 줄

`--full` 을 주면 세션 5.5-B 까지와 같은 전문(「현재 상태」 절 전체 + 마지막 세션
로그 블록 전체)을 낸다. **열린 미해결은 두 모드 모두 전문이다.**

`--web` 은 **웹 대화창에 붙이는 축약**이다(세션 7.24). 셋만 낸다 — ⓐ 마지막 판
한 줄(세션 로그의 마지막 `### ` 제목) · ⓑ 열린 미해결 **번호와 제목만** ·
ⓒ 「현재 상태」 마지막 행의 「다음 세션 첫 작업」 **첫 문장만**. 본문·근거·크기 칸은 싣지
않는다. 출처는 다른 두 모드와 같은 `PROCEED.md` 하나다 — 목록을 옮겨 적은 새
파일을 만들지 않는다.

**PROCEED.md 파싱 규칙 — 세션 5.5-C 에서 바뀐 것**

- (기존) `## 현재 상태` · `## 세션 로그` · `## 미해결 목록` 절을 `## ` 경계로 자른다.
- (기존) 세션 로그 절의 마지막 `### ` 블록을 마지막 세션으로 본다.
- (신규) 「현재 상태」 표의 **행을 셀 단위로 읽는다** — `| 세션 | 단계 | 게이트 |
  다음 세션 첫 작업 |` 4칸. 셀 안의 `\\|`(escape 된 파이프, 예 `\\|잔차\\|`)는
  구분자로 세지 않는다.
- (신규) 마지막 세션 로그 블록 안에서 **`**다음이 알아야 할 것**` 로 시작하는
  문단**을 찾아 그 뒤부터 블록 끝까지를 축약형에 싣는다. 못 찾으면 블록 전체를
  싣는다(조용히 비우지 않는다).
- (세션 7.24 · `--web` 전용) 미해결 「한 줄」 칸에서 **제목**을 뽑을 때, 머리의
  `[신설 · 세션 7.22]` 같은 대괄호 표시를 떼고 첫 문장 경계(`. ` 또는 ` — `)
  에서 자른다. **이 절단은 `--web` 안에서만 일어난다** — 기본·`--full` 은 그대로
  전문이다.
- (삭제) 열린 미해결의 88자 절단(`SUMMARY_MAX_CHARS`) 을 없앴다. 「한 줄」·「영향」
  두 칸을 **전문 그대로** 싣는다.

표준 라이브러리만 쓴다. 경로는 저장소 루트 기준 상대경로만 쓴다(절대 규칙 15).
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

#: 이 스크립트는 저장소 루트에 있다. 절대경로를 하드코딩하지 않는다.
REPO_ROOT = Path(__file__).resolve().parent
PROCEED_PATH = REPO_ROOT / "PROCEED.md"

CURRENT_STATE_HEADING = "## 현재 상태"
SESSION_LOG_HEADING = "## 세션 로그"
OPEN_ITEMS_HEADING = "## 미해결 목록"

#: 마지막 세션 로그에서 축약형이 실어 나르는 문단의 머리말.
NEXT_SESSION_HEADING = "**다음이 알아야 할 것**"

#: 미해결 항목이 **닫혔다**고 볼 표시. `PROCEED.md` 규약상 닫힌 항목도 지우지 않고
#: 상태를 표시해 남기므로(「닫힌 항목도 지우지 않고 상태를 표시해 남긴다」),
#: 한 줄 안에 이 표시들이 있으면 닫힌 것으로 본다.
CLOSED_MARKERS: tuple[str, ...] = ("[해소", "[닫힘", "[확정", "**해소됨**")

#: 축약형에서 상세하게 싣는 「현재 상태」 표의 마지막 판 수.
RECENT_ROWS = 2

#: 줄바꿈 폭 [문자]. **내용을 자르지 않는다** — 접기만 한다.
WRAP_WIDTH = 88

#: `| a | b |` 행을 셀로 쪼갤 때 escape 된 `\|` 는 구분자가 아니다.
_CELL_SPLIT = re.compile(r"(?<!\\)\|")


def _section(lines: list[str], heading: str) -> list[str]:
    """`heading` 으로 시작하는 `## ` 절을 다음 `## ` 절 직전까지 잘라낸다."""
    try:
        start = lines.index(heading)
    except ValueError:
        return []
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break
    return lines[start:end]


def _last_session_block(lines: list[str]) -> list[str]:
    """세션 로그 절에서 마지막 `### ` 블록만 잘라낸다."""
    log = _section(lines, SESSION_LOG_HEADING)
    starts = [i for i, line in enumerate(log) if line.startswith("### ")]
    if not starts:
        return []
    return log[starts[-1] :]


def _next_session_notes(block: list[str]) -> list[str]:
    """마지막 세션 블록에서 「다음이 알아야 할 것」 문단만 잘라낸다.

    못 찾으면 **블록 전체**를 돌려준다 — 조용히 비우면 브리핑이 거짓말을 한다.
    """
    if not block:
        return []
    for i, line in enumerate(block):
        if line.strip().startswith(NEXT_SESSION_HEADING):
            return [block[0], "", *block[i:]]
    return block


def _strip_markup(text: str) -> str:
    """표 셀에서 강조·인용 표기를 걷어내 한 줄로 만든다. **자르지 않는다.**"""
    for token in ("**", "`"):
        text = text.replace(token, "")
    return " ".join(text.replace("\\|", "|").split())


def _cells(line: str) -> list[str]:
    """마크다운 표의 한 행을 셀 목록으로 쪼갠다(escape 된 `\\|` 는 셀 구분이 아니다)."""
    return [cell.strip() for cell in _CELL_SPLIT.split(line.strip().strip("|"))]


def _wrap(text: str, indent: str = "  ") -> list[str]:
    """폭에 맞춰 접는다. 내용은 그대로 둔다."""
    return textwrap.wrap(text, width=WRAP_WIDTH, subsequent_indent=indent) or [""]


def _gate_verdict(cell: str) -> str:
    """게이트 칸에서 한 낱말짜리 판정만 뽑는다(축약형 한 줄용)."""
    plain = _strip_markup(cell)
    for token in ("게이트 통과", "게이트 없음", "판정 없음", "해당 없음", "통과"):
        if token in plain:
            return token
    return plain[:24]


def current_state_rows(lines: list[str]) -> list[list[str]]:
    """「현재 상태」 표의 데이터 행만 셀 목록으로 뽑는다."""
    rows: list[list[str]] = []
    for line in _section(lines, CURRENT_STATE_HEADING):
        if not line.startswith("| "):
            continue
        cells = _cells(line)
        if len(cells) < 4 or cells[0] == "세션" or set(cells[0]) <= {"-", ":"}:
            continue  # 머리글·구분선
        rows.append(cells)
    return rows


def format_current_state(rows: list[list[str]]) -> str:
    """축약형 「현재 상태」 — 최근 `RECENT_ROWS` 판만 상세, 나머지는 한 줄."""
    if not rows:
        return "## 현재 상태\n\n(「현재 상태」 표를 읽지 못했다 — 표 형식이 바뀌었는가)"
    out = [f"## 현재 상태 ({len(rows)}판)", ""]
    older, recent = rows[:-RECENT_ROWS], rows[-RECENT_ROWS:]
    for cells in older:
        out.append(f"- 세션 {cells[0]} · {_gate_verdict(cells[2])}")
    if older:
        out.append("")
    for cells in recent:
        out.append(f"### 세션 {cells[0]}")
        out += _wrap(f"단계: {_strip_markup(cells[1])}", indent="      ")
        out += _wrap(f"게이트: {_strip_markup(cells[2])}", indent="        ")
        out.append("")
    return "\n".join(out).rstrip()


def open_unresolved_items(lines: list[str]) -> list[tuple[str, str, str]]:
    """「미해결 목록」 표에서 **열려 있는 항목만** (번호, 한 줄, 영향) 으로 뽑는다.

    닫힌 항목은 넣지 않는다 — 브리핑은 "오늘 무엇이 남아 있나"를 보는 자리다.
    **어느 칸도 자르지 않는다**(세션 5.5-C). 표 형식이 바뀌면 조용히 빈 목록을
    내지 말고 그대로 아무것도 못 뽑은 것이 드러나야 하므로, 파싱에 실패한 행은
    건너뛰되 성공한 행만 싣는다.
    """
    items: list[tuple[str, str, str]] = []
    for line in _section(lines, OPEN_ITEMS_HEADING):
        if not line.startswith("| "):
            continue
        cells = _cells(line)
        if len(cells) < 2 or not cells[0].isdigit():
            continue  # 머리글·구분선
        # 닫힘 표시는 한 줄 요약이 아니라 「영향」 칸 끝에 덧붙는 일이 많다
        # (세션 3-A2 이후 규약: 원 문구를 덮어쓰지 않고 뒤에 덧붙인다).
        # 그래서 **행 전체**를 본다.
        if any(marker in line for marker in CLOSED_MARKERS):
            continue
        impact = _strip_markup(cells[3]) if len(cells) > 3 else ""
        items.append((cells[0], _strip_markup(cells[1]), impact))
    return items


def format_open_items(items: list[tuple[str, str, str]]) -> str:
    """열린 미해결 목록을 브리핑 블록 문자열로 만든다 (순수 함수).

    **전문이다 — 어떤 경우에도 자르지 않는다.**
    """
    if not items:
        return "## 열린 미해결 항목\n\n(없음 — 또는 「미해결 목록」 표 형식이 바뀌었다)"
    out = [f"## 열린 미해결 항목 ({len(items)}건 · 전문, 자르지 않는다)", ""]
    for number, summary, impact in items:
        out += _wrap(f"#{number} · {summary}", indent="     ")
        if impact:
            out += _wrap(f"→ 영향: {impact}", indent="        ")
        out.append("")
    return "\n".join(out).rstrip()


#: `--web` 제목 추출 — 머리의 대괄호 표시(`[신설 · 세션 7.22]`)와 첫 문장 경계.
_LEADING_TAG = re.compile(r"^\[[^\]]*\]\s*")
_SENTENCE_END = re.compile(r"\. | — ")


def web_title(summary: str) -> str:
    """미해결 「한 줄」 칸에서 제목 한 줄만 뽑는다 (순수 함수).

    `--web` 전용이다 — 기본·`--full` 은 이 함수를 쓰지 않고 전문을 싣는다.
    """
    return _SENTENCE_END.split(_LEADING_TAG.sub("", summary))[0].rstrip(".")


def build_web_brief(text: str) -> str:
    """웹 대화창용 축약 브리핑 (순수 함수) — ⓐ 마지막 판 · ⓑ 열린 미해결 제목 · ⓒ 다음.

    읽는 곳은 `PROCEED.md` 하나다. 물리 수치를 싣지 않는다.
    """
    lines = text.splitlines()
    block = _last_session_block(lines)
    rows = current_state_rows(lines)
    items = open_unresolved_items(lines)

    head = _strip_markup(block[0].removeprefix("### ")) if block else ""
    out = _wrap(f"현재: {head}" if head else "현재: 세션 로그를 읽지 못했다.")

    out += ["", f"열린 미해결 {len(items)}건 (번호·제목만 — 본문은 PROCEED.md):"]
    if not items:
        out.append("  (없음 — 또는 「미해결 목록」 표 형식이 바뀌었다)")
    for number, summary, _impact in items:
        out += _wrap(f"  #{number} {web_title(summary)}", indent="      ")

    out += ["", "다음 첫 작업 (「현재 상태」 마지막 행 · 첫 문장만):"]
    # 「한 줄」 칸과 같은 규칙으로 첫 문장만 싣는다 — 뒤는 PROCEED.md 를 본다.
    nxt = web_title(_strip_markup(rows[-1][3])) if rows else "(표를 읽지 못했다)"
    out += _wrap(f"  {nxt}")
    return "\n".join(out)


REFERENCE_LINE = (
    "참조: 과대해석 금지 문구와 게이트 근거 수치는 PROCEED.md 「현재 상태」 아래 ※ 절과\n"
    "      `python -m cdu_simul.dataset_report` 에 있다. `daily_brief.py --full` = 전문."
)


def build_brief(text: str, *, full: bool = False) -> str:
    """PROCEED.md 본문에서 브리핑 문자열을 만든다 (순수 함수)."""
    lines = text.splitlines()
    open_items = format_open_items(open_unresolved_items(lines))

    if full:
        parts = _section(lines, CURRENT_STATE_HEADING) + [""] + _last_session_block(lines)
        body = "\n".join(line.rstrip() for line in parts).strip()
        return body + "\n\n" + open_items

    rows = current_state_rows(lines)
    head = _headline(rows)
    state = format_current_state(rows)
    notes = "\n".join(
        line.rstrip() for line in _next_session_notes(_last_session_block(lines))
    ).strip()
    blocks = [head, state, notes, open_items, REFERENCE_LINE]
    return "\n\n".join(block for block in blocks if block)


def _headline(rows: list[list[str]]) -> str:
    """한 줄 현재 위치 + 다음에 할 일 (표의 마지막 행에서 만든다)."""
    if not rows:
        return "현재: PROCEED.md 「현재 상태」 표를 읽지 못했다."
    last = rows[-1]
    out = [f"현재: 세션 {last[0]} — {_gate_verdict(last[2])}"]
    out += _wrap(f"다음: {_strip_markup(last[3])}", indent="      ")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    full = "--full" in args
    web = "--web" in args
    unknown = [a for a in args if a not in ("--full", "--web")]
    if unknown:
        joined = " ".join(unknown)
        print(f"모르는 인자: {joined} (쓸 수 있는 것: --full · --web)", file=sys.stderr)
        return 2
    if full and web:
        print("--full 과 --web 은 함께 쓸 수 없다.", file=sys.stderr)
        return 2

    if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔 기본 인코딩 대비
        sys.stdout.reconfigure(encoding="utf-8")

    if not PROCEED_PATH.exists():
        print(f"PROCEED.md 를 찾을 수 없다: {PROCEED_PATH.name}", file=sys.stderr)
        return 1

    text = PROCEED_PATH.read_text(encoding="utf-8")
    if web:
        print(build_web_brief(text))
        return 0

    brief = build_brief(text, full=full)
    if not brief:
        print("PROCEED.md 에서 「현재 상태」·세션 로그를 찾지 못했다.", file=sys.stderr)
        return 1

    print("=" * 78)
    label = "전문" if full else "축약"
    print(f"CDU Feasibility Study — 브리핑 · {label} (가정값 기반 · 실측 아님)")
    print("=" * 78)
    print(brief)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
