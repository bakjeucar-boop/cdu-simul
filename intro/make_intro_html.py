"""소개자료를 **파일 하나**로 만든다 — 시연 화면을 iframe 으로 품는다.

`intro/intro.html` 의 `{{DEMO_SRCDOC}}` 한 자리에 `dist/cdu-demo.html` 전체를
넣어 `dist/cdu-intro.html` 을 낸다. 시연 화면은 **읽기만 한다** — 한 글자도
고치지 않고 그대로 들어간다(`srcdoc` 속성값이라 `&` 와 `"` 만 HTML 이스케이프하고,
빌드가 도로 풀어 원본과 같은지 확인한다).

시연 화면은 `demo/make_single_html.py` 를 그대로 실행해 얻는다 — 그 스크립트를
고치지도 베끼지도 않는다.

검사를 통과하지 못하면 예외를 던지고 **산출물을 남기지 않는다.**
검사 대상은 **iframe 바깥(소개자료)뿐**이다 — 시연 화면은 읽기 전용이라 고칠 수
없고, 화면에도 걸리는 말이 없다.

표준 라이브러리만 쓴다(절대 규칙 12).
경로는 이 파일 위치 기준 상대경로만 쓴다(절대 규칙 15).

    .venv/Scripts/python.exe intro/make_intro_html.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUT_NAME = "cdu-intro.html"

#: 시연 화면이 들어갈 자리 — 정확히 한 번 나와야 한다
SLOT = "{{DEMO_SRCDOC}}"

#: iframe 을 감싼 표식 — 검사에서 시연 화면을 떼어내는 데 쓴다
DEMO_BEGIN = "<!--DEMO-BEGIN-->"
DEMO_END = "<!--DEMO-END-->"

#: 모든 화면에 있어야 하는 표기 (절대 규칙 11)
DISCLAIMER = "가정값 기반 — 실측 아님"

#: 절 하나를 여는 표식
SCREEN_TAG = '<section class="screen"'

#: 자료에 나가면 안 되는 말 (사람이 정한 어휘 통제)
BANNED = {
    "누출": "「랙 배관 저항 증가」로 쓴다. 기기 이름도 예외가 아니다",
    "정확": "실측이 없어 「정확도」를 잰 적이 없다",
    "대체": "Aspen 계열은 「용도가 다르다」로 쓴다",
}


def escape_srcdoc(html: str) -> str:
    """`srcdoc="..."` 속성값 안에 넣을 수 있게 최소로만 이스케이프한다."""
    return html.replace("&", "&amp;").replace('"', "&quot;")


def unescape_srcdoc(value: str) -> str:
    """`escape_srcdoc` 를 되돌린다 — 넣은 것과 같은지 확인하는 용도."""
    return value.replace("&quot;", '"').replace("&amp;", "&")


def build_demo(repo: Path) -> str:
    """`demo/make_single_html.py` 를 그대로 실행하고 그 산출물을 읽는다."""
    script = repo / "demo" / "make_single_html.py"
    subprocess.run([sys.executable, str(script)], check=True)
    return (repo / "dist" / "cdu-demo.html").read_text(encoding="utf-8")


def shell_of(html_out: str) -> str:
    """산출물에서 시연 화면(iframe)을 떼어낸 나머지 — 검사 대상이다."""
    head, _, rest = html_out.partition(DEMO_BEGIN)
    _, _, tail = rest.partition(DEMO_END)
    return head + tail


def verify(html_out: str, demo_html: str) -> tuple[int, int]:
    """세 가지를 검사한다. 하나라도 어긋나면 예외를 던진다.

    세션 7.18 이 옛 검사 ⑷(절 수 12~15)를 없앴다. scroll-snap 이 사라진 뒤로
    그 수는 「자료가 몇 판인가」가 아니라 절 개수를 뜻하게 됐고(세션 7.17 보고),
    뜻을 잃은 범위는 아무것도 지키지 않으면서 절을 더할 때만 막았다.
    **분량은 사람이 읽고 판정한다.**

    세션 7.19 가 검사 ⑶ 을 「절 수 = 표기 수」에서 **「최소 한 번」**으로 바꿨다.
    사람이 「가정값 기반 — 실측 아님」을 표지에만 두기로 정했기 때문이다(C2) —
    절마다 반복하면 읽는 사람이 그 줄을 건너뛰게 된다. 지켜야 할 것은
    절대 규칙 11 이 요구하는 「산출물에 표시가 있다」이고, 그것은 자료에
    한 번 있으면 성립한다. 자리를 표지로 정하는 것은 사람의 판단이라
    검사가 개수를 세지 않는다.
    """
    # ⑴ 넣은 시연 화면이 원본과 완전히 같다 (한 글자도 고치지 않았다)
    embedded = html_out.partition(DEMO_BEGIN)[2].partition(DEMO_END)[0]
    start = embedded.index('srcdoc="') + len('srcdoc="')
    recovered = unescape_srcdoc(embedded[start : embedded.index('"></iframe>', start)])
    if recovered != demo_html:
        raise RuntimeError("검사 ⑴ 실패: 품은 시연 화면이 dist/cdu-demo.html 과 다르다")

    shell = shell_of(html_out)

    # ⑵ 금지어가 소개자료 쪽에 하나도 없다 (iframe 안은 검사하지 않는다)
    for word, why in BANNED.items():
        hits = shell.count(word)
        if hits:
            raise RuntimeError(f"검사 ⑵ 실패: 「{word}」 {hits}회 — {why}")

    # ⑶ 「가정값 기반 — 실측 아님」이 자료에 최소 한 번 있다 (절대 규칙 11)
    screens = shell.count(SCREEN_TAG)
    marks = shell.count(DISCLAIMER)
    if marks < 1:
        raise RuntimeError(
            f"검사 ⑶ 실패: 「{DISCLAIMER}」 0회 — 최소 한 번은 있어야 한다"
        )

    print(f"  검사 ⑴ 시연 화면 원본과 동일 ({len(demo_html):,} 자)")
    print(f"  검사 ⑵ 금지어 0회 — {' · '.join(BANNED)} (iframe 안은 대상 아님)")
    print(f"  검사 ⑶ 「{DISCLAIMER}」 {marks}회 (최소 1) · 절 {screens}개")
    return screens, marks


def main() -> None:
    intro_dir = Path(__file__).resolve().parent
    repo = intro_dir.parent

    demo_html = build_demo(repo)
    src = (intro_dir / "intro.html").read_text(encoding="utf-8")

    hits = src.count(SLOT)
    if hits != 1:
        raise RuntimeError(f"자리 {SLOT} 이 {hits}번 나온다 — 1번이어야 한다")

    html_out = src.replace(SLOT, escape_srcdoc(demo_html))
    verify(html_out, demo_html)

    out_path = repo / "dist" / OUT_NAME
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(html_out, encoding="utf-8", newline="\n")

    size = out_path.stat().st_size
    print(f"{out_path.relative_to(repo)}  {size:,} B ({size / 1024:,.1f} KB)")


if __name__ == "__main__":
    main()
