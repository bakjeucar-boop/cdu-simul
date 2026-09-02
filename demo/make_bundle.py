"""시연 화면을 저장소 없는 PC 로 옮기기 위한 zip 꾸러미를 만든다.

표준 라이브러리만 쓴다(절대 규칙 12).
경로는 이 파일 위치 기준 상대경로만 쓴다(절대 규칙 15).
꾸러미 안은 평평한 구조다 — 푼 자리에서 start.bat / start.sh 가 바로 돈다.

    .venv/Scripts/python.exe demo/make_bundle.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

# 꾸러미에 담는 것 — 화면 하나(pfd.html) + 그 데이터 + 실행 안내·스크립트
BUNDLE_FILES = (
    "pfd.html",
    "demo_steady.json",
    "start.bat",
    "start.sh",
    "README.txt",
)

OUT_NAME = "cdu-demo.zip"


def main() -> None:
    demo_dir = Path(__file__).resolve().parent
    out_path = demo_dir.parent / "dist" / OUT_NAME
    out_path.parent.mkdir(exist_ok=True)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in BUNDLE_FILES:
            bundle.write(demo_dir / name, arcname=name)

    print(f"{out_path.relative_to(demo_dir.parent)}  {out_path.stat().st_size:,} B")
    for info in zipfile.ZipFile(out_path).infolist():
        print(f"  {info.filename:<18} {info.file_size:>9,} B")


if __name__ == "__main__":
    main()
