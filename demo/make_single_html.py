"""시연 화면을 **파일 하나**로 만든다 — 더블클릭으로 열리는 단일 HTML.

`demo/pfd.html` 은 데이터를 `fetch("demo_steady.json")` 으로 읽으므로 `file://`
에서 열리지 않는다(그래서 `start.bat`/`start.sh` 가 간이 서버를 띄운다).
이 스크립트는 그 한 자리만 갈아끼워 데이터를 HTML 안으로 넣는다 — 받는 사람이
압축 해제도 스크립트 실행도 하지 않고 파일 하나를 더블클릭하면 된다.

표준 라이브러리만 쓴다(절대 규칙 12).
경로는 이 파일 위치 기준 상대경로만 쓴다(절대 규칙 15).
`demo/pfd.html` · `demo/demo_steady.json` 은 **읽기만 한다.**

검사(⑴ 데이터 동일 ⑵ fetch 잔존 없음 ⑶ 「실측 아님」 표기 유지)를 통과하지
못하면 예외를 던지고 **산출물을 남기지 않는다.**

    .venv/Scripts/python.exe demo/make_single_html.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

OUT_NAME = "cdu-demo.html"

# 갈아끼울 자리 — 이 문자열이 정확히 한 번 나와야 한다(pfd.html:865)
FETCH_CALL = 'fetch("demo_steady.json")'

# 넣은 데이터를 다시 뽑아내기 위한 표식
DATA_BEGIN = "/*CDU-DEMO-DATA*/"
DATA_END = "/*CDU-DEMO-END*/"

# 산출물에 그대로 남아 있어야 하는 표기(절대 규칙 11)
DISCLAIMER = "가정값 기반 — 실측 아님"

# demo_steady.json 이 담고 있는 케이스 수 — 시연 데이터의 실제 개수이지
# 가정치가 아니다(5장·5-1 의 값이 아니다)
EXPECTED_CASE_COUNT = 136

# 스크립트 안에 넣을 때 막아야 하는 글자
#   `<`      — `</script>` · `<!--` 로 스크립트가 끊긴다. JSON 에서 `<` 는
#              문자열 안에만 나오므로 유니코드 이스케이프로 바꿔도 뜻이 보존된다
#   U+2028·9 — JSON 에서는 문자열 안의 보통 글자지만 예전 JS 문법에서는 줄바꿈
# 어느 쪽이든 `\uXXXX` 로 바꾼다 — JSON 문자열 안에서 유효하고 JS 도 같게 읽는다
UNSAFE_CHARS = ("<", " ", " ")


def embed_payload(json_text: str) -> str:
    """JSON 본문을 `<script>` 안에 넣어도 끊기지 않게 이스케이프한다."""
    payload = json_text
    for char in UNSAFE_CHARS:
        payload = payload.replace(char, f"\\u{ord(char):04x}")
    for char in UNSAFE_CHARS:
        if char in payload:
            raise RuntimeError(f"이스케이프가 남긴 문자가 있다: {char!r}")
    return payload


def build(html_text: str, json_text: str) -> str:
    """pfd.html 의 fetch 한 자리를 데이터로 갈아끼운 HTML 을 낸다."""
    hits = html_text.count(FETCH_CALL)
    if hits != 1:
        raise RuntimeError(
            f"치환 대상 {FETCH_CALL!r} 이 {hits}번 나온다 — 1번이어야 한다"
        )

    payload = embed_payload(json_text)
    # `.then(r => r.json())` 이하가 그대로 돌도록 같은 모양(then 가능·json())으로 준다
    replacement = f"Promise.resolve({{ json: () => ({DATA_BEGIN}{payload}{DATA_END}) }})"
    return html_text.replace(FETCH_CALL, replacement)


def extract_payload(html_text: str) -> Any:
    """산출된 HTML 에서 데이터를 도로 뽑아 파싱한다(검사 ⑴ 용)."""
    start = html_text.index(DATA_BEGIN) + len(DATA_BEGIN)
    end = html_text.index(DATA_END)
    return json.loads(html_text[start:end])


def verify(html_out: str, data: Any) -> None:
    """세 가지를 검사한다. 하나라도 어긋나면 예외를 던진다."""
    # ⑴ 다시 뽑은 데이터가 원본과 완전히 같다 (케이스 수도 센다)
    recovered = extract_payload(html_out)
    if recovered != data:
        raise RuntimeError("검사 ⑴ 실패: 다시 뽑은 데이터가 demo_steady.json 과 다르다")
    n_cases = len(recovered["cases"])
    if n_cases != EXPECTED_CASE_COUNT:
        raise RuntimeError(
            f"검사 ⑴ 실패: 케이스 {n_cases}개 — {EXPECTED_CASE_COUNT}개여야 한다"
        )

    # ⑵ demo_steady.json 을 가리키는 fetch 가 남아 있지 않다
    leftover = re.search(r"""fetch\(\s*["']demo_steady\.json""", html_out)
    if leftover is not None:
        raise RuntimeError("검사 ⑵ 실패: demo_steady.json 을 읽는 fetch 가 남아 있다")

    # ⑶ 「가정값 기반 — 실측 아님」 표기가 그대로 있다
    if DISCLAIMER not in html_out:
        raise RuntimeError(f"검사 ⑶ 실패: {DISCLAIMER!r} 표기가 사라졌다")

    print(f"  검사 ⑴ 데이터 동일 · 케이스 {n_cases}개")
    print("  검사 ⑵ demo_steady.json fetch 잔존 없음")
    print(f"  검사 ⑶ 「{DISCLAIMER}」 표기 유지")


def main() -> None:
    demo_dir = Path(__file__).resolve().parent
    html_text = (demo_dir / "pfd.html").read_text(encoding="utf-8")
    json_text = (demo_dir / "demo_steady.json").read_text(encoding="utf-8")
    data = json.loads(json_text)

    html_out = build(html_text, json_text)
    verify(html_out, data)

    out_path = demo_dir.parent / "dist" / OUT_NAME
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(html_out, encoding="utf-8", newline="\n")

    size = out_path.stat().st_size
    rel = out_path.relative_to(demo_dir.parent)
    print(f"{rel}  {size:,} B ({size / 1024:,.1f} KB)")


if __name__ == "__main__":
    main()
