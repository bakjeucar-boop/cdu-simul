"""계통 방열 크기 어림 — 세션 7.62 (재는 판 · 넣지 않는다).

**이 스크립트가 내는 수는 전부 어림값이다. 「잰 값」이 아니다.**
모델에 방열 항은 없고(세션 7.62 C1 확인), 이 판은 그것을 넣지 않는다.
크기가 6장 ① 임계(0.1 %) 대비 어디쯤인지, 세션 7.61 이 넣은 펌프 수력동력과
얼마나 상쇄되는지만 낸다. 넣을지 말지는 사람이 정한다.

**5장 밖 숫자는 `OUT5` 하나에 모아 둔다** — `assumptions.py` 나 정본 5-1 에
넣지 않는다(세션 7.62 범위). 이 판 한정이며, 사람이 채택하기 전까지 다른
코드가 여기서 읽어 가면 안 된다.

돌리는 법 (저장소 루트에서)::

    .venv/Scripts/python.exe scripts/measure_ambient_heat_loss.py

`results/cdu_dataset.csv` 가 있어야 한다(`python -m cdu_simul.dataset` 이 만든다).
CSV 는 **pandas 로 읽는다** — `caveat` 열의 따옴표 안 쉼표 때문에 `awk -F,` 로
가르면 「샘」 행에서 필드가 밀린다(세션 7.49 D5).
"""

from __future__ import annotations

import math
import sys
from itertools import product
from pathlib import Path

import pandas as pd

#: 저장소 루트 기준 상대경로만 쓴다 (절대 규칙 15).
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = REPO_ROOT / "results" / "cdu_dataset.csv"

# `scripts` 를 패키지로 잡아 세션 7.61 의 펌프유량 규칙을 그대로 재사용한다
# (같은 식을 두 곳이 따로 쓰지 않는다 — collaboration.md ③).
sys.path.insert(0, str(REPO_ROOT))

from cdu_simul.assumptions import (  # noqa: E402
    ASSUMPTION_TAG,
    PASCAL_PER_MAQ,
    PIPING,
    SCENARIO,
)
from scripts.measure_pump_power import pump_flow_Lps  # noqa: E402

_M_PER_MM: float = 1.0e-3
_M3S_PER_LPS: float = 1.0e-3
_W_PER_KW: float = 1.0e3

#: 6장 ① energy balance 임계 [%].
BALANCE_THRESHOLD_PERCENT: float = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# C3. 5장 밖 숫자 — **이 판 한정 어림값**. 전수가 여기 있고, 개수는 len(OUT5) 다.
# ─────────────────────────────────────────────────────────────────────────────
#: 이름 → (값, 단위, 출처). 세는 방법: `len(OUT5)` (스크립트가 표 끝에 찍는다).
OUT5: dict[str, tuple[float, str, str]] = {
    "outer_diameter_25A_mm": (
        33.4,
        "mm",
        "규격 정의값 — ASME B36.10M NPS 1 외경. 5-1 「배관 내경」 행이 같은 "
        "규격의 내경만 적었다. 외경도 같은 규격의 정의값이지 가정치가 아니나, "
        "5-1 에 적혀 있지 않으므로 여기서 센다",
    ),
    "outer_diameter_80A_mm": (
        88.9,
        "mm",
        "규격 정의값 — ASME B36.10M NPS 3 외경. 위와 같다",
    ),
    "T_ambient_low_C": (
        18.0,
        "℃",
        "업계 통상값 — ASHRAE TC9.9 데이터센터 권장 환경범위(18~27℃)의 하한. "
        "실제 기계실 온도는 5장에 없다",
    ),
    "T_ambient_high_C": (
        27.0,
        "℃",
        "업계 통상값 — 위 권장 범위의 상한",
    ),
    "h_bare_low_Wm2K": (
        8.0,
        "W/m²K",
        "물리 유도 — 수평 원관 자연대류 h≈1.32·(ΔT/D)^0.25 [ΔT 10~20 K · "
        "D 0.033~0.089 m → 4.3~6.5] + 복사 h_rad=εσ(Ts²+Ta²)(Ts+Ta) "
        "[ε 0.3 무광 강관 → 1.9] 의 합을 내림한 값",
    ),
    "h_bare_high_Wm2K": (
        12.0,
        "W/m²K",
        "물리 유도 — 위 대류 상한 6.5 + 복사 ε 0.9(도장·산화 표면) → 5.7 "
        "의 합을 올림한 값",
    ),
    "insulation_thickness_mm": (
        25.0,
        "mm",
        "업계 통상값 — 냉수배관 결로방지 보온 두께로 흔히 쓰는 25 mm. "
        "5장에 보온 유무·사양이 없다",
    ),
    "insulation_k_WmK": (
        0.036,
        "W/mK",
        "업계 통상값 — 엘라스토머 폼 보온재 열전도율 전형치. 위와 같은 이유",
    ),
}


def _v(name: str) -> float:
    return OUT5[name][0]


#: (라벨, 호칭경 외경 mm, 등가길이 m) — 5-1 「계통 보유수량 M」 방침 B 그대로.
#: 전부 25A + 등가길이 하한 / 전부 80A + 등가길이 상한 두 극단. 중간값은 없다.
GEOMETRY_BOUNDS: tuple[tuple[str, float, float], ...] = (
    (
        "하한 (전부 25A · 등가길이 하한 · 8랙)",
        _v("outer_diameter_25A_mm"),
        PIPING.equivalent_length_m.low,
    ),
    (
        "상한 (전부 80A · 등가길이 상한 · 8랙)",
        _v("outer_diameter_80A_mm"),
        PIPING.equivalent_length_m.high,
    ),
)


def surface_area_m2(outer_diameter_mm: float, equivalent_length_m: float) -> float:
    """계통 배관 외표면적 [m²] — 등가길이를 물리길이로 읽는다.

    **등가길이 ≥ 물리길이이므로 이 어림은 과대 방향이다** (등가길이는 부속류
    저항을 직관 길이로 환산한 개념이라 실제 배관보다 길다). 5장에 물리길이가
    없어 다른 읽을 방법이 없다.

    랙 수 8 은 `SCENARIO.racks_per_cdu` 에서 읽는다 — `dynamics.holdup_bounds()`
    가 M 을 8배로 읽는 것과 같은 근거다(PROCEED.md 세션 5.5-D 마무리 ⑷).
    """
    return (
        math.pi
        * outer_diameter_mm
        * _M_PER_MM
        * equivalent_length_m
        * SCENARIO.racks_per_cdu
    )


def overall_U_Wm2K(
    outer_diameter_mm: float, h_bare_Wm2K: float, insulated: bool
) -> float:
    """겉면적(무보온 배관 외표면) 기준 총괄 열전달계수 [W/m²K].

    무보온이면 그대로 `h`. 보온이면 원통 전도저항 ln(r2/r1)/(2πk) 과 보온재
    바깥면 외기저항을 직렬로 두고, **무보온 외표면적 기준**으로 환산한다.

    금속관 벽 전도저항과 관내 대류저항은 빼고 본다 — 보온재·외기 저항보다 두
    자리 작다. 보온재 바깥면 h 는 무보온 h 를 그대로 쓴다(면적이 커지고 ΔT 가
    작아지는 몫을 반영하지 않는다 — **과대 방향**).
    """
    if not insulated:
        return h_bare_Wm2K
    r1_m = outer_diameter_mm * _M_PER_MM / 2.0
    r2_m = r1_m + _v("insulation_thickness_mm") * _M_PER_MM
    r_conduction = math.log(r2_m / r1_m) / (2.0 * math.pi * _v("insulation_k_WmK"))
    r_film = 1.0 / (h_bare_Wm2K * 2.0 * math.pi * r2_m)
    return 1.0 / ((r_conduction + r_film) * 2.0 * math.pi * r1_m)


def load_operating_points() -> pd.DataFrame:
    """부하별 운전점 — 정상 행에서만 읽는다 [부하 %, T_벌크평균, Q_rack, P_hyd].

    T_유체는 5-1 벌크평균 규약 그대로 (T_supply + T_return)/2 다 — 새 규칙이
    아니다. P_hyd 는 세션 7.61 이 모델에 넣은 펌프 수력동력이고, 펌프를 지나는
    유량은 `scripts.measure_pump_power.pump_flow_Lps` 규칙을 그대로 쓴다.
    """
    frame = pd.read_csv(DATASET_CSV)
    normal = frame[frame["scenario_kind"] == "정상"].copy()
    normal["T_bulk_C"] = (normal["T_supply_C"] + normal["T_return_C"]) / 2.0
    normal["P_hyd_kW"] = (
        normal["pump_head_mAq"]
        * PASCAL_PER_MAQ
        * pump_flow_Lps(normal)
        * _M3S_PER_LPS
        / _W_PER_KW
    )
    # ε·C_총 [kW/K] — 세션 7.61 이 쓴 관계 T_ret = T_2차공급 + (Q_총+P_hyd)/(ε·C_총)
    # 를 뒤집어 읽는다. 새 계수를 만들지 않는다 (세션 7.62 C6).
    normal["eps_C_total_kWK"] = (
        normal["hx_duty_kW"] / (normal["T_return_C"] - normal["T_secondary_supply_C"])
    )
    grouped = normal.groupby("load_percent").agg(
        T_bulk_C=("T_bulk_C", "mean"),
        P_hyd_kW=("P_hyd_kW", "mean"),
        eps_C_total_kWK=("eps_C_total_kWK", "mean"),
        rows=("T_bulk_C", "size"),
    )
    grouped["Q_rack_kW"] = (
        SCENARIO.cdu_total_load_kW * grouped.index.to_series() / 100.0
    )
    return grouped.reset_index()


def build_combinations(points: pd.DataFrame) -> pd.DataFrame:
    """C2 표면적 × C3 조합 × 부하 전수."""
    rows = []
    for (geo_label, d_mm, length_m), T_amb_C, h_Wm2K, insulated, point in product(
        GEOMETRY_BOUNDS,
        (_v("T_ambient_low_C"), _v("T_ambient_high_C")),
        (_v("h_bare_low_Wm2K"), _v("h_bare_high_Wm2K")),
        (False, True),
        points.itertuples(index=False),
    ):
        area_m2 = surface_area_m2(d_mm, length_m)
        U_Wm2K = overall_U_Wm2K(d_mm, h_Wm2K, insulated)
        Q_loss_kW = U_Wm2K * area_m2 * (point.T_bulk_C - T_amb_C) / _W_PER_KW
        net_kW = point.P_hyd_kW - Q_loss_kW
        rows.append(
            {
                "geometry": geo_label.split(" ")[0],
                "A_m2": area_m2,
                "T_amb_C": T_amb_C,
                "h_Wm2K": h_Wm2K,
                "보온": "보온" if insulated else "무보온",
                "U_Wm2K": U_Wm2K,
                "load_percent": point.load_percent,
                "Q_loss_kW": Q_loss_kW,
                "Q_loss_over_Qrack_percent": Q_loss_kW / point.Q_rack_kW * 100.0,
                "임계배수": (
                    Q_loss_kW / point.Q_rack_kW * 100.0 / BALANCE_THRESHOLD_PERCENT
                ),
                "Q_loss_over_P_hyd": Q_loss_kW / point.P_hyd_kW,
                "net_kW": net_kW,
                "net_over_Qrack_percent": net_kW / point.Q_rack_kW * 100.0,
                "net_임계배수": abs(net_kW) / point.Q_rack_kW * 100.0
                / BALANCE_THRESHOLD_PERCENT,
                "dT_return_walkback_K": Q_loss_kW / point.eps_C_total_kWK,
            }
        )
    return pd.DataFrame(rows)


#: 세션 7.61 C8 실측 이동 [K] — 이 판이 다시 재지 않고 그대로 인용한다.
SESSION761_T_RETURN_SHIFT_K: tuple[float, float] = (0.065232, 0.110178)
SESSION761_T_SUPPLY_SHIFT_K: tuple[float, float] = (0.031035, 0.067582)


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

    print("── C2. 배관 외표면적 (등가길이를 물리길이로 읽음 · 과대 방향) ──")
    for label, d_mm, length_m in GEOMETRY_BOUNDS:
        area = surface_area_m2(d_mm, length_m)
        print(
            f"  {label:<38} A = π × {d_mm:.1f} mm × {length_m:.0f} m × "
            f"{SCENARIO.racks_per_cdu} = {area:8.3f} m²"
        )

    print()
    print("── C3. 5장 밖 숫자 (이 판 한정 어림값) ──")
    for name, (value, unit, source) in OUT5.items():
        print(f"  {name:<26} {value:>8.4g} {unit:<7} {source.splitlines()[0][:62]}")
    print(f"  → 전수 {len(OUT5)} 개 (세는 방법: `len(OUT5)`)")

    points = load_operating_points()
    print()
    print("── 운전점 (정상 행 · 벌크평균 규약) ──")
    print(points.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    table = build_combinations(points)
    print()
    print("── C4. 계통 방열 크기 (32조합) ──")
    c4 = table[
        [
            "geometry", "보온", "T_amb_C", "h_Wm2K", "U_Wm2K", "load_percent",
            "Q_loss_kW", "Q_loss_over_Qrack_percent", "임계배수",
        ]
    ]
    print(c4.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    worst = table.loc[table["Q_loss_over_Qrack_percent"].idxmax()]
    best = table.loc[table["Q_loss_over_Qrack_percent"].idxmin()]
    print(
        f"  최대: {worst['Q_loss_over_Qrack_percent']:.4f} % "
        f"({worst['임계배수']:.1f}× 임계) — {worst['geometry']}·{worst['보온']}·"
        f"T_amb {worst['T_amb_C']:.0f}℃·h {worst['h_Wm2K']:.0f}·"
        f"부하 {worst['load_percent']:.0f}%"
    )
    print(
        f"  최소: {best['Q_loss_over_Qrack_percent']:.4f} % "
        f"({best['임계배수']:.1f}× 임계) — {best['geometry']}·{best['보온']}·"
        f"T_amb {best['T_amb_C']:.0f}℃·h {best['h_Wm2K']:.0f}·"
        f"부하 {best['load_percent']:.0f}%"
    )

    print()
    print("── C5. 펌프 수력동력과의 상쇄 (지금 해 위의 덧셈 — 다시 풀지 않았다) ──")
    c5 = table[
        [
            "geometry", "보온", "T_amb_C", "h_Wm2K", "load_percent",
            "Q_loss_over_P_hyd", "net_kW", "net_over_Qrack_percent", "net_임계배수",
        ]
    ]
    print(c5.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        f"  방열/P_hyd 비 {table['Q_loss_over_P_hyd'].min():.3f} ~ "
        f"{table['Q_loss_over_P_hyd'].max():.3f}"
    )
    print(
        f"  순 합(P_hyd − 방열) {table['net_kW'].min():+.4f} ~ "
        f"{table['net_kW'].max():+.4f} kW · "
        f"양(+) {int((table['net_kW'] > 0).sum())}조합 / "
        f"음(−) {int((table['net_kW'] < 0).sum())}조합"
    )
    print(
        "  ※ 두 항을 다 넣고 **다시 풀면** 이 값에서 더 움직인다 — "
        "여기 적은 것은 현재 해 위의 덧셈이다 (세션 7.59 D7 과 같은 한계)"
    )

    print()
    print("── C6. 온도 해가 되돌아오는 폭 (T_return · ε·C_총 은 7.61 관계에서 읽음) ──")
    lo, hi = SESSION761_T_RETURN_SHIFT_K
    c6 = table[
        ["geometry", "보온", "T_amb_C", "h_Wm2K", "load_percent",
         "dT_return_walkback_K"]
    ].copy()
    c6["7.61이동_하한대비_%"] = c6["dT_return_walkback_K"] / lo * 100.0
    c6["7.61이동_상한대비_%"] = c6["dT_return_walkback_K"] / hi * 100.0
    print(c6.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        f"  7.61 T_return 이동 {lo:.6f}~{hi:.6f} K · "
        f"T_supply 이동 {SESSION761_T_SUPPLY_SHIFT_K[0]:.6f}~"
        f"{SESSION761_T_SUPPLY_SHIFT_K[1]:.6f} K (다시 재지 않고 인용)"
    )
    print(
        "  ※ T_supply 되돌아오는 폭은 내지 않는다 — 방열의 노드 배분이 "
        "정해져 있지 않다(펌프 일의 5-1 배분 규칙은 방열에 적용되지 않는다)"
    )

    print()
    print(f"※ {ASSUMPTION_TAG} (절대 규칙 11)")
    print(
        "※ **이 판의 수는 전부 어림값이다 — 「잰 값」이 아니다.** 모델에 방열 항은 "
        "없고 이 스크립트는 아무것도 넣지 않는다"
    )
    print(
        "※ 어림이 과대 방향인 몫 셋: 등가길이 ≥ 물리길이 · 보온재 바깥면 h 를 "
        "무보온 h 로 씀 · 관벽/관내 저항 무시"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
