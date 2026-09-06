"""펌프 수력동력 계측 — 세션 7.59 D3·D4 표를 다시 낸다 [세션 7.61 에 저장소로 옮김].

세션 7.59 는 이 계측을 스크래치패드에 두고 지웠다. 같은 표를 다시 낼 명령이
저장소에 없으면 그 판의 수치를 재현할 수 없으므로 여기 남긴다.

**세션 7.59 와 값이 같기를 기대하지 않는다.** 7.59 는 펌프 일이 모델 **밖**에
있을 때 쟀고, 세션 7.61 이 그것을 열과 balance 양쪽에 넣었다 — 온도 해가
움직였으므로 P_hyd 도 잔차도 그때와 다르다. 나온 값을 그대로 읽는다.

돌리는 법 (저장소 루트에서)::

    .venv/Scripts/python.exe -m scripts.measure_pump_power
    .venv/Scripts/python.exe scripts/measure_pump_power.py

`results/cdu_dataset.csv` 가 있어야 한다(`python -m cdu_simul.dataset` 이 만든다).
CSV 는 **pandas 로 읽는다** — `caveat` 열의 따옴표 안 쉼표 때문에 `awk -F,` 로
가르면 「샘」 행에서 필드가 밀린다(세션 7.49 D5).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from cdu_simul.assumptions import ASSUMPTION_TAG, PASCAL_PER_MAQ, SCENARIO

#: 저장소 루트 기준 상대경로만 쓴다 (절대 규칙 15).
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = REPO_ROOT / "results" / "cdu_dataset.csv"

#: L/s → m^3/s. mAq × Pa/mAq × m^3/s = W (절대 규칙 9).
_M3S_PER_LPS: float = 1.0e-3
_W_PER_KW: float = 1.0e3
_PERCENT: float = 1.0e-2

#: 6장 ① energy balance 임계 [%].
BALANCE_THRESHOLD_PERCENT: float = 0.1

_COLUMNS = [
    "anomaly_mechanism",
    "regime",
    "load_percent",
    "pump_head_mAq",
    "total_flow_Lps",
    "return_flow_Lps",
    "pump_sees_supply_flow",
    "energy_balance_residual_percent",
]


def pump_flow_Lps(frame: pd.DataFrame) -> pd.Series:
    """펌프를 지나는 유량 [L/s] — 「샘」 구조 자유도 ⓑ 를 그대로 따른다.

    `pump_sees_supply_flow` 가 참이면 공급유량(`total_flow_Lps`), 거짓이면
    환수유량이다 [5-1 「「샘」 구조 자유도 셋」]. 「막힘」·정상 행은 밀폐루프라
    둘이 같다.
    """
    sees_supply = frame["pump_sees_supply_flow"].astype("boolean").fillna(True)
    return frame["total_flow_Lps"].where(sees_supply, frame["return_flow_Lps"])


def rack_load_kW(frame: pd.DataFrame) -> pd.Series:
    """CDU 전체 랙 발열량 [kW] — 5장 랙당 발열량 × 랙 수 × 부하율."""
    return (
        SCENARIO.rack_it_load_kW
        * SCENARIO.racks_per_cdu
        * frame["load_percent"]
        * _PERCENT
    )


def _span(values: pd.Series, unit: str, digits: int = 6) -> str:
    if values.dropna().empty:
        return "(빈칸)"
    return (
        f"{values.min():.{digits}f} ~ {values.max():.{digits}f} {unit}"
        f"  (평균 {values.mean():.{digits}f})"
    )


def format_power_table(frame: pd.DataFrame) -> str:
    """세션 7.59 D3 에 해당하는 표 — 전 행의 펌프 수력동력."""
    lines = [
        "── D3 재현: 펌프 수력동력 P_hyd = H(Q_pump) × Q_pump ──",
        f"전 행 {len(frame):,}건 · {_span(frame['P_hyd_kW'], 'kW')}",
        "",
        f"{'기구':<10}{'국면':<12}{'건수':>8}{'P_hyd 최소':>14}"
        f"{'P_hyd 최대':>14}{'P/Q_rack 최대':>16}",
    ]
    for (mechanism, regime), group in frame.groupby(
        ["anomaly_mechanism", "regime"], dropna=False
    ):
        ratio = group["P_hyd_kW"] / group["rack_load_kW"]
        ratio_text = (
            "(부하 0)" if not (group["rack_load_kW"] > 0).any()
            else f"{ratio[group['rack_load_kW'] > 0].max():.6f}"
        )
        lines.append(
            f"{str(mechanism):<10}{str(regime):<12}{len(group):>8,}"
            f"{group['P_hyd_kW'].min():>14.6f}{group['P_hyd_kW'].max():>14.6f}"
            f"{ratio_text:>16}"
        )
    return "\n".join(lines)


def format_balance_table(frame: pd.DataFrame) -> str:
    """세션 7.59 D4 에 해당하는 표 — energy balance 잔차.

    **세션 7.61 이후로는 잔차가 이미 펌프 항을 품고 있다.** 7.59 는 열에는 넣지
    않고 balance 에만 넣어 −0.476136 ~ −3.588382 % 를 봤다. 여기 값이 그것과
    다른 것이 정상이다.
    """
    residual = frame["energy_balance_residual_percent"]
    over = residual.abs() > BALANCE_THRESHOLD_PERCENT
    lines = [
        "── D4 재현: energy balance 잔차 (6장 ① · 임계 0.1 %) ──",
        f"전 행 {len(frame):,}건 · {_span(residual, '%')}",
        f"|최대| {residual.abs().max():.6f} %"
        f" · 임계 초과 {int(over.sum()):,}건"
        f" · 빈칸 {int(residual.isna().sum()):,}건",
        "",
        f"{'기구':<10}{'국면':<12}{'건수':>8}{'잔차 최소':>14}"
        f"{'잔차 최대':>14}{'초과':>8}{'빈칸':>8}",
    ]
    for (mechanism, regime), group in frame.groupby(
        ["anomaly_mechanism", "regime"], dropna=False
    ):
        res = group["energy_balance_residual_percent"]
        lines.append(
            f"{str(mechanism):<10}{str(regime):<12}{len(group):>8,}"
            f"{res.min():>14.6f}{res.max():>14.6f}"
            f"{int((res.abs() > BALANCE_THRESHOLD_PERCENT).sum()):>8,}"
            f"{int(res.isna().sum()):>8,}"
        )
    return "\n".join(lines)


def load_frame(path: Path = DATASET_CSV) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=_COLUMNS)
    frame["P_hyd_kW"] = (
        frame["pump_head_mAq"]
        * pump_flow_Lps(frame)
        * PASCAL_PER_MAQ
        * _M3S_PER_LPS
        / _W_PER_KW
    )
    frame["rack_load_kW"] = rack_load_kW(frame)
    return frame


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔 기본 인코딩 대비
        sys.stdout.reconfigure(encoding="utf-8")
    if not DATASET_CSV.exists():
        print(
            f"{DATASET_CSV.relative_to(REPO_ROOT)} 가 없다 — "
            "먼저 `python -m cdu_simul.dataset` 을 돌린다.",
            file=sys.stderr,
        )
        return 1
    frame = load_frame()
    print(format_power_table(frame))
    print()
    print(format_balance_table(frame))
    print()
    print(f"※ {ASSUMPTION_TAG} (절대 규칙 11)")
    print(
        "※ 잰 것은 **유체에 들어가는 수력동력**이지 모터 입력전력이 아니다 — "
        "펌프 효율·모터 방열은 5장에 없어 모델에 없다"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
