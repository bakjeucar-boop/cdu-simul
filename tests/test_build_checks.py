"""빌드 검사가 실제로 막는지 — 소개자료 · 시연 빌드 스크립트 (세션 7.122 · #117).

입력은 이 파일 안에서 만든 작은 문자열이다. 저장소의 `demo/pfd.html` ·
`demo/demo_steady.json` · `intro/intro.html` 과 `results/` · `dist/` 를 읽지도
쓰지도 않는다 — 없으면 건너뛰는 시험을 만들지 않기 위해서다.
검사마다 온전한 입력에서 통과하는지와 표기·조건을 뺀 입력에서 막히는지를 본다.

`intro/` · `demo/` 는 `__init__.py` 가 없는 디렉터리라 이름공간 패키지로 부른다
(`pyproject.toml` 의 `pythonpath = ["."]`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from demo import make_single_html as demo_build
from intro import make_intro_html as intro_build

# ── 시연 빌드 (demo/make_single_html.py) ────────────────────────────────

MARK = demo_build.DISCLAIMER
FOOTER = f"<footer>{MARK}</footer>"
DEMO_HTML = (
    "<html><head><style>footer{opacity:.8}</style></head><body>\n"
    f"{FOOTER}\n"
    '<script>fetch("demo_steady.json").then(r => r.json())</script>\n'
    "</body></html>\n"
)
DATA = {"cases": [{"i": i} for i in range(demo_build.EXPECTED_CASE_COUNT)]}


def _demo_out(html: str = DEMO_HTML, data: object = DATA) -> str:
    return demo_build.build(html, json.dumps(data, ensure_ascii=False))


def test_demo_intact_passes() -> None:
    demo_build.verify(_demo_out(), DATA)


def test_demo_check1_blocks_changed_data() -> None:
    with pytest.raises(RuntimeError, match="검사 ⑴"):
        demo_build.verify(_demo_out(), {**DATA, "extra": 1})


def test_demo_check1_blocks_wrong_case_count() -> None:
    short = {"cases": DATA["cases"][:-1]}
    with pytest.raises(RuntimeError, match="검사 ⑴"):
        demo_build.verify(_demo_out(data=short), short)


def test_demo_check2_blocks_leftover_fetch() -> None:
    html = DEMO_HTML.replace("</script>", "\nfetch('demo_steady.json')</script>")
    with pytest.raises(RuntimeError, match="검사 ⑵"):
        demo_build.verify(_demo_out(html), DATA)


_NO_FOOTER = DEMO_HTML.replace(FOOTER, "<footer></footer>")


@pytest.mark.parametrize(
    ("html", "data"),
    [
        pytest.param(_NO_FOOTER, DATA, id="visible-removed"),
        pytest.param(
            _NO_FOOTER.replace("<body>", f"<body><!-- {MARK} -->"),
            DATA,
            id="html-comment",
        ),
        pytest.param(
            _NO_FOOTER.replace("<script>", f"<script>// {MARK}\n"),
            DATA,
            id="js-comment",
        ),
        pytest.param(_NO_FOOTER, {**DATA, "note": MARK}, id="data-string"),
    ],
)
def test_demo_check3_blocks_mark_not_visible(html: str, data: object) -> None:
    """#116 — 보이는 표기가 빠지면 주석·데이터 속 표기로는 통과하지 않는다."""
    with pytest.raises(RuntimeError, match="검사 ⑶"):
        demo_build.verify(_demo_out(html, data), data)


@pytest.mark.parametrize("hits", [0, 2])
def test_demo_build_guard_blocks_fetch_not_once(hits: int) -> None:
    html = DEMO_HTML.replace(demo_build.FETCH_CALL, demo_build.FETCH_CALL * hits)
    with pytest.raises(RuntimeError, match="치환 대상"):
        demo_build.build(html, json.dumps(DATA))


def test_demo_embed_payload_escapes_unsafe_chars() -> None:
    """통과 쪽만 본다 — 이스케이프 잔존 가드는 막는 쪽에 닿는 입력이 지금 코드에 없다."""
    unsafe = "</script>" + chr(0x2028) + chr(0x2029)
    text = json.dumps({"s": unsafe}, ensure_ascii=False)
    payload = demo_build.embed_payload(text)
    assert not any(ch in payload for ch in demo_build.UNSAFE_CHARS)
    assert json.loads(payload) == json.loads(text)


# ── 소개자료 빌드 (intro/make_intro_html.py) ─────────────────────────────

IMARK = intro_build.DISCLAIMER
DEMO_PAGE = '<p class="x">시연 & "화면"</p>'
IFRAME = (
    '<!--DEMO-BEGIN--><iframe srcdoc="'
    + intro_build.SLOT
    + '"></iframe><!--DEMO-END-->'
)


def _intro_src(cover: str = f"<p>{IMARK}</p>", other: str = "") -> str:
    return (
        f'<section class="screen" id="s01">{cover}</section>\n'
        f'<section class="screen" id="s02">{other}\n{IFRAME}\n</section>\n'
    )


def _intro_out(src: str, demo: str = DEMO_PAGE) -> str:
    return src.replace(intro_build.SLOT, intro_build.escape_srcdoc(demo))


def test_intro_intact_passes() -> None:
    assert intro_build.verify(_intro_out(_intro_src()), DEMO_PAGE) == (2, 1)


def test_intro_check1_blocks_changed_demo() -> None:
    with pytest.raises(RuntimeError, match="검사 ⑴"):
        intro_build.verify(_intro_out(_intro_src()), DEMO_PAGE + " ")


@pytest.mark.parametrize("word", list(intro_build.BANNED))
def test_intro_check2_blocks_banned_word(word: str) -> None:
    with pytest.raises(RuntimeError, match="검사 ⑵"):
        intro_build.verify(_intro_out(_intro_src(other=f"<p>{word}</p>")), DEMO_PAGE)


def test_intro_check2_ignores_banned_word_inside_iframe() -> None:
    demo = f"<p>{next(iter(intro_build.BANNED))}</p>"
    intro_build.verify(_intro_out(_intro_src(), demo), demo)


@pytest.mark.parametrize(
    ("src", "demo"),
    [
        pytest.param(_intro_src(cover=""), DEMO_PAGE, id="cover-removed"),
        pytest.param(
            _intro_src(cover=f"<!-- {IMARK} -->"), DEMO_PAGE, id="cover-comment"
        ),
        pytest.param(_intro_src(cover="", other=IMARK), DEMO_PAGE, id="other-section"),
        pytest.param(_intro_src(cover=""), IMARK, id="iframe-only"),
    ],
)
def test_intro_check3_blocks_mark_not_on_cover(src: str, demo: str) -> None:
    """#107 — 표지에 보이는 표기가 없으면 다른 자리의 표기로 통과하지 않는다."""
    with pytest.raises(RuntimeError, match="검사 ⑶"):
        intro_build.verify(_intro_out(src, demo), demo)


def _run_intro_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, src: str) -> None:
    """`main` 을 임시 디렉터리 기준으로 돌린다 — 시연 빌드(하위 프로세스)는 대신한다."""
    (tmp_path / "intro").mkdir()
    (tmp_path / "intro" / "intro.html").write_text(src, encoding="utf-8")
    monkeypatch.setattr(
        intro_build, "__file__", str(tmp_path / "intro" / "make_intro_html.py")
    )
    monkeypatch.setattr(intro_build, "build_demo", lambda repo: DEMO_PAGE)
    intro_build.main()


def test_intro_main_intact_writes_to_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_intro_main(tmp_path, monkeypatch, _intro_src())
    assert (tmp_path / "dist" / intro_build.OUT_NAME).is_file()


@pytest.mark.parametrize("hits", [0, 2])
def test_intro_main_guard_blocks_slot_not_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hits: int
) -> None:
    src = _intro_src().replace(intro_build.SLOT, intro_build.SLOT * hits)
    with pytest.raises(RuntimeError, match="자리"):
        _run_intro_main(tmp_path, monkeypatch, src)
    assert not (tmp_path / "dist").exists()
