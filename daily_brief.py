"""대화 시작 시 붙일 브리핑을 PROCEED.md 에서 뽑아 stdout 으로 출력한다.

`collaboration.md` 「하루의 모양」: 아침에 `git pull` 후 이 출력을 새 대화의 첫
메시지에 붙인다.

출력 내용 = `PROCEED.md` 의 「현재 상태」 표 + 마지막 세션 로그 블록.
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


def build_brief(text: str) -> str:
    """PROCEED.md 본문에서 브리핑 문자열을 만든다 (순수 함수)."""
    lines = text.splitlines()
    parts = _section(lines, CURRENT_STATE_HEADING) + [""] + _last_session_block(lines)
    return "\n".join(line.rstrip() for line in parts).strip()


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
