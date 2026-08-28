"""대화 시작 시 붙일 브리핑을 PROCEED.md 에서 뽑아 stdout 으로 출력한다.

`collaboration.md` 「하루의 모양」: 아침에 `git pull` 후 이 출력을 새 대화의 첫
메시지에 붙인다.

출력 내용 = `PROCEED.md` 의 「현재 상태」 표 + 마지막 세션 로그 블록 +
**열려 있는 미해결 항목 목록**(세션 5에서 추가).
표준 라이브러리만 쓴다. 경로는 저장소 루트 기준 상대경로만 쓴다(절대 규칙 15).
"""

from __future__ import annotations

import sys
from pathlib import Path

#: 이 스크립트는 저장소 루트에 있다. 절대경로를 하드코딩하지 않는다.
REPO_ROOT = Path(__file__).resolve().parent
PROCEED_PATH = REPO_ROOT / "PROCEED.md"

CURRENT_STATE_HEADING = "## 현재 상태"
SESSION_LOG_HEADING = "## 세션 로그"
OPEN_ITEMS_HEADING = "## 미해결 목록"

#: 미해결 항목이 **닫혔다**고 볼 표시. `PROCEED.md` 규약상 닫힌 항목도 지우지 않고
#: 상태를 표시해 남기므로(「닫힌 항목도 지우지 않고 상태를 표시해 남긴다」),
#: 한 줄 안에 이 표시들이 있으면 닫힌 것으로 본다.
CLOSED_MARKERS: tuple[str, ...] = ("[해소", "[닫힘", "[확정", "**해소됨**")

#: 열린 항목 한 줄 요약의 최대 길이 [문자]. 브리핑이 표로 뒤덮이지 않게 자른다.
SUMMARY_MAX_CHARS = 88


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


def _strip_markup(text: str) -> str:
    """표 셀에서 강조·인용 표기를 걷어내 한 줄 요약으로 만든다."""
    for token in ("**", "`"):
        text = text.replace(token, "")
    return " ".join(text.split())


def open_unresolved_items(lines: list[str]) -> list[tuple[str, str]]:
    """「미해결 목록」 표에서 **열려 있는 항목만** (번호, 한 줄 요약) 으로 뽑는다.

    닫힌 항목은 넣지 않는다 — 브리핑은 "오늘 무엇이 남아 있나"를 보는 자리다.
    표 형식이 바뀌면 조용히 빈 목록을 내지 말고 그대로 아무것도 못 뽑은 것이
    드러나야 하므로, 파싱에 실패한 행은 건너뛰되 성공한 행만 싣는다.
    """
    items: list[tuple[str, str]] = []
    for line in _section(lines, OPEN_ITEMS_HEADING):
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or not cells[0].isdigit():
            continue  # 머리글·구분선
        number, summary = cells[0], cells[1]
        # 닫힘 표시는 한 줄 요약이 아니라 「영향」 칸 끝에 덧붙는 일이 많다
        # (세션 3-A2 이후 규약: 원 문구를 덮어쓰지 않고 뒤에 덧붙인다).
        # 그래서 **행 전체**를 본다.
        if any(marker in line for marker in CLOSED_MARKERS):
            continue
        plain = _strip_markup(summary)
        if len(plain) > SUMMARY_MAX_CHARS:
            plain = plain[: SUMMARY_MAX_CHARS - 1].rstrip() + "…"
        items.append((number, plain))
    return items


def format_open_items(items: list[tuple[str, str]]) -> str:
    """열린 미해결 목록을 브리핑 블록 문자열로 만든다 (순수 함수)."""
    if not items:
        return "## 열린 미해결 항목\n\n(없음 — 또는 「미해결 목록」 표 형식이 바뀌었다)"
    lines = [f"## 열린 미해결 항목 ({len(items)}건)", ""]
    lines += [f"- #{number} · {summary}" for number, summary in items]
    return "\n".join(lines)


def build_brief(text: str) -> str:
    """PROCEED.md 본문에서 브리핑 문자열을 만든다 (순수 함수)."""
    lines = text.splitlines()
    parts = _section(lines, CURRENT_STATE_HEADING) + [""] + _last_session_block(lines)
    body = "\n".join(line.rstrip() for line in parts).strip()
    return body + "\n\n" + format_open_items(open_unresolved_items(lines))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔 기본 인코딩 대비
        sys.stdout.reconfigure(encoding="utf-8")

    if not PROCEED_PATH.exists():
        print(f"PROCEED.md 를 찾을 수 없다: {PROCEED_PATH.name}", file=sys.stderr)
        return 1

    brief = build_brief(PROCEED_PATH.read_text(encoding="utf-8"))
    if not brief:
        print("PROCEED.md 에서 「현재 상태」·세션 로그를 찾지 못했다.", file=sys.stderr)
        return 1

    print("=" * 78)
    print("CDU Feasibility Study — 브리핑 (가정값 기반 · 실측 아님)")
    print("=" * 78)
    print(brief)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
