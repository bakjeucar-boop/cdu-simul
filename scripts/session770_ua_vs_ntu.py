"""세션 7.70 — NTU 상수 대신 UA 고정이면 무엇이 얼마나 움직이는지 **잰다**.

**이 스크립트는 모델에 아무것도 넣지 않는다.** `src/` 를 한 줄도 고치지 않고,
데이터셋을 다시 만들지 않으며, 미해결 #70 을 닫지도 않는다. 이미 있는
`results/cdu_dataset.csv` 를 읽어 **그 해 위에서** 다음 셋을 낼 뿐이다.

  ⑴ 지금 쓰는 상수 NTU (데이터셋 `ntu` 열)
  ⑵ UA 를 정격점에서 역산해 고정했을 때 나왔을 NTU = UA / C_min
  ⑶ 둘의 비, 그로부터 나오는 ε 변화와 온도 이동의 **어림**

**⑶ 은 어림이지 잰 값이 아니다.** 지금 해 위의 덧셈이다 — ε 가 바뀌면 온도가
바뀌고, 온도가 바뀌면 물성(ρ·cp)이 바뀌어 C 가 바뀌고, 그러면 ε 가 또 바뀐다.
그 되먹임을 이 스크립트는 **풀지 않는다.** 다시 풀면 더 움직인다
(같은 한계를 세션 7.59 D7 · 7.62 가 적었다).

UA 역산에 쓰는 값과 그 출처
---------------------------
  · 정격 1차측 부피유량 15.5 L/s — 5장 「펌프 정격유량 약 15.5 L/s」
    (`assumptions.PumpAssumptions.rated_flow_Lps`)
  · 정격 2차측 부피유량 = 유량비 1:1 × 위 값 — 5장 · 5-1 「2차측 유체」
    (`assumptions.HeatExchangerAssumptions.secondary_flow_Lps`)
  · 정격 1차측 물성온도 37℃ — 이미 있는 `hydraulics.rated_property_temperature_C()`
    (5장 공급 32℃ · 환수 42℃ 의 벌크평균). 5-1 「계통 잔여저항의 물성 의존」이
    Kv·K·잔여저항 역산의 기준점으로 정한 「정격 물성(37℃)」을 **그대로 빌려 온다.**
    **빌려 온 것임을 밝힌다** — 5-1 은 그 문장을 잔여저항에 대해 적었지 열교환기
    UA 에 대해 적지 않았다. 새 숫자는 아니지만 5-1 이 HX 에 대해 정한 것도 아니다
  · 2차측 물성온도 = 2차측 **공급온도** — 5-1 「2차측 공급온도」 ·
    `model.hx_capacity_terms` 가 적은 그대로(선택이 아니라 강제)
  · 정격 NTU = 그 행 자신의 `ntu` 축값(2 또는 3) — 5장 「NTU 2~3」

  **새 숫자 0개다.** 위 다섯은 전부 5장·5-1 에 있는 값이거나 거기서 유도된다.

**모든 수치는 가정값 기반이며 실측이 아니다.**
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cdu_simul.assumptions import ASSUMPTION_TAG, HEAT_EXCHANGER, PUMP
from cdu_simul.fluid import coolant_cp_Jkg_K, coolant_density_kgm3
from cdu_simul.hydraulics import rated_property_temperature_C
from cdu_simul.model import hx_effectiveness_counterflow

#: 저장소 루트 — 이 파일은 `<root>/scripts/` 에 있다 (절대 규칙 15).
_ROOT = Path(__file__).resolve().parents[1]
_DATASET = _ROOT / "results" / "cdu_dataset.csv"
_OUT = _ROOT / "results" / "session770_ua_vs_ntu.txt"

#: 1 L = 1e-3 m^3.
_M3_PER_LITRE = 1.0e-3

#: 정격 1차측 물성온도 [℃]. **리터럴을 적지 않는다** — 이미 있는
#: `hydraulics.rated_property_temperature_C()`(5장 공급 32℃ · 환수 42℃ 의
#: 벌크평균 = 37℃)를 그대로 쓴다. 그 함수가 Kv·K·잔여저항 역산의 기준점이고,
#: UA 역산도 같은 성격의 역산이라 기준점을 새로 고르지 않는다.
_RATED_PROPERTY_T_C = rated_property_temperature_C()

#: 이 판이 읽어야 하는 열만 읽는다 — 59열 15,104행을 통째로 올리지 않는다.
_COLUMNS = [
    "scenario_kind",
    "regime",
    "cdu_config",
    "cdu_index",
    "pump_head_rated_mAq",
    "branch_dp_rated_mAq",
    "valve_dp_rated_mAq",
    "anomaly_mechanism",
    "blockage_level_percent",
    "massloss_size_fraction",
    "load_percent",
    "ntu",
    "T_secondary_supply_C",
    "total_flow_Lps",
    "return_flow_Lps",
    "secondary_share_Lps",
    "T_supply_C",
    "T_return_C",
    "hx_duty_kW",
]


def _capacity_rate_W_K(volume_flow_Lps: float, T_property_C: float) -> float:
    """체적유량과 물성온도에서 열용량유량 C = V̇·ρ·cp [W/K] 를 낸다 (순수 함수)."""
    return (
        volume_flow_Lps
        * _M3_PER_LITRE
        * coolant_density_kgm3(T_property_C)
        * coolant_cp_Jkg_K(T_property_C)
    )


def _rated_c_min_W_K(T_secondary_supply_C: float) -> float:
    """정격점의 C_min [W/K] — UA 역산의 기준점.

    정격점은 1차측·2차측 모두 5장 정격 부피유량 15.5 L/s 이고, 1차측 물성은
    37℃ · 2차측 물성은 공급온도에서 평가한다(위 모듈 docstring 의 출처 목록).
    """
    return min(
        _capacity_rate_W_K(PUMP.rated_flow_Lps, _RATED_PROPERTY_T_C),
        _capacity_rate_W_K(HEAT_EXCHANGER.secondary_flow_Lps, T_secondary_supply_C),
    )


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    """행마다 ⑴ 지금 NTU ⑵ UA 고정 NTU ⑶ 비·ε·온도 이동 어림을 붙인다.

    **순수 함수다** — 입력 프레임을 고치지 않고 새 프레임을 돌려준다.
    """
    out = frame.copy()

    # 1차측에서 HX 를 지나는 것은 「샘」에서 **환수유량**이다
    # (`massloss_thermal.solve_massloss_steady` 가 `C_return_W_K` 를 넘긴다).
    # 「막힘」·정상은 환수유량 = 총유량이라 같은 열을 써도 된다.
    primary_flow_Lps = out["return_flow_Lps"].fillna(out["total_flow_Lps"])

    # 물성 평가온도 = 벌크평균 [5-1 「cp·ρ 평가 온도」 · `model` 기본 규칙].
    T_bulk_C = 0.5 * (out["T_supply_C"] + out["T_return_C"])

    out["C_primary_W_K"] = [
        _capacity_rate_W_K(q, t) for q, t in zip(primary_flow_Lps, T_bulk_C, strict=True)
    ]
    out["C_secondary_W_K"] = [
        _capacity_rate_W_K(q, t)
        for q, t in zip(
            out["secondary_share_Lps"], out["T_secondary_supply_C"], strict=True
        )
    ]
    out["C_min_W_K"] = out[["C_primary_W_K", "C_secondary_W_K"]].min(axis=1)
    out["C_max_W_K"] = out[["C_primary_W_K", "C_secondary_W_K"]].max(axis=1)
    out["primary_is_C_min"] = out["C_primary_W_K"] < out["C_secondary_W_K"]

    # ⑴ 지금 — 상수 NTU. ⑵ UA 고정 — UA = NTU_정격 × C_min,정격.
    out["UA_W_K"] = [
        ntu * _rated_c_min_W_K(t2)
        for ntu, t2 in zip(out["ntu"], out["T_secondary_supply_C"], strict=True)
    ]
    out["ntu_now"] = out["ntu"]
    out["ntu_ua_fixed"] = out["UA_W_K"] / out["C_min_W_K"]
    out["ntu_ratio"] = out["ntu_ua_fixed"] / out["ntu_now"]

    # ⑶ ε 는 지금 코드와 **같은 함수**로 낸다 — 여기서 식을 다시 적지 않는다.
    cr = out["C_min_W_K"] / out["C_max_W_K"]
    out["eps_now"] = [
        hx_effectiveness_counterflow(n, c)
        for n, c in zip(out["ntu_now"], cr, strict=True)
    ]
    out["eps_ua_fixed"] = [
        hx_effectiveness_counterflow(n, c)
        for n, c in zip(out["ntu_ua_fixed"], cr, strict=True)
    ]
    out["eps_ratio"] = out["eps_ua_fixed"] / out["eps_now"]

    # 온도 이동 **어림**. 모델은 T_return = T_2차 + Q_총/(ε·C_min) 이고
    # T_supply = T_return − (Q+P_환수)/C_총 이므로, ε 만 바뀌면 둘은 **같은 양**
    # 만큼 평행이동한다. Q_총 은 데이터셋의 `hx_duty_kW` 로 대신한다
    # (모델 정의상 Q_hx = ε·C_min·(T_return − T_2차) 이므로 분자가 바로 그것이다).
    q_total_W = out["hx_duty_kW"] * 1.0e3
    out["dT_return_K"] = (q_total_W / out["C_min_W_K"]) * (
        1.0 / out["eps_ua_fixed"] - 1.0 / out["eps_now"]
    )
    return out


#: 정상 기준행을 이상행에 붙일 때 맞추는 축 — 「막힘」·「샘」 양쪽에 다 있는 열만.
_BASELINE_KEYS = [
    "cdu_config",
    "cdu_index",
    "pump_head_rated_mAq",
    "branch_dp_rated_mAq",
    "valve_dp_rated_mAq",
    "ntu",
    "T_secondary_supply_C",
    "load_percent",
]


def with_signal_shift(frame: pd.DataFrame) -> pd.DataFrame:
    """이상행의 온도 이동에서 **같은 축 정상행의 이동을 뺀다**.

    UA 고정이 낳는 온도 이동은 정상행에도 이상행에도 걸리므로, 그 **공통분**은
    신호(이상 − 정상)에서 상쇄된다. 신호에 실제로 걸리는 것은 차이분뿐이다.
    이 함수가 내는 `dT_signal_K` 가 그 차이분이다.

    **이것도 어림이다** — ⑴ 지금 해 위의 덧셈이라는 한계를 그대로 물려받고,
    ⑵ 정상 기준행은 `blockage_level_percent == 0` 인 「막힘」 축의 0 수준
    행이다(「샘」 전용 정상행은 데이터셋에 따로 없다).
    """
    baseline = frame.loc[
        (frame["anomaly_mechanism"] == "K_approx")
        & (frame["blockage_level_percent"] == 0.0),
        [*_BASELINE_KEYS, "dT_return_K", "T_return_C"],
    ].rename(
        columns={"dT_return_K": "dT_baseline_K", "T_return_C": "T_return_baseline_C"}
    )
    merged = frame.merge(baseline, on=_BASELINE_KEYS, how="left", validate="many_to_one")
    merged["dT_signal_K"] = merged["dT_return_K"] - merged["dT_baseline_K"]
    # 지금 데이터셋이 내고 있는 신호 자체 — 이상행 T_return − 같은 축 정상행 T_return.
    merged["signal_K"] = merged["T_return_C"] - merged["T_return_baseline_C"]
    merged["signal_share"] = merged["dT_signal_K"].abs() / merged["signal_K"].abs()
    return merged


def _state_label(row: pd.Series) -> str:
    """네 상태 — 정상 · 유휴 · 막힘 · 샘 (`docs/session749-state-table.md` 축).

    데이터셋의 기구 열 값은 「막힘」이 `K_approx` · 「샘」이 `massloss` 이고,
    `K_approx` 중 `blockage_level_percent == 0` 인 행이 **정상**이다
    (정상 경로를 「막힘」 축의 0 수준으로 낸다 — `dataset.py`).
    """
    if str(row["anomaly_mechanism"]) == "massloss":
        return "샘"
    if float(row["blockage_level_percent"]) > 0.0:
        return "막힘"
    return "유휴" if float(row["load_percent"]) <= 20.0 else "정상"


def summarize(enriched: pd.DataFrame) -> str:
    """상태 × 부하 표와 가장 어긋나는 자리를 글로 낸다 (순수 함수)."""
    frame = enriched.copy()
    frame["state"] = frame.apply(_state_label, axis=1)

    lines: list[str] = [
        "세션 7.70 — NTU 상수 vs UA 고정 (재는 판 · 모델·데이터셋 무수정)",
        f"{ASSUMPTION_TAG}",
        "",
        f"읽은 것: {_DATASET.relative_to(_ROOT).as_posix()} · {len(frame):,}행",
        f"UA 역산 기준: 1차 {PUMP.rated_flow_Lps} L/s @ {_RATED_PROPERTY_T_C}℃ · "
        f"2차 {HEAT_EXCHANGER.secondary_flow_Lps} L/s @ 공급온도 · NTU 는 행의 축값",
        "",
        "── C3. 상태 × 부하 별 NTU (중앙값 / 최소 / 최대) ──",
        "",
        f"{'상태':<6}{'부하%':>7}{'행수':>8}{'NTU 지금':>10}"
        f"{'UA고정 NTU 중앙':>16}{'최소':>9}{'최대':>9}{'비 최대':>10}",
    ]
    for (state, load, _), group in frame.groupby(
        ["state", "load_percent", "ntu"], sort=True
    ):
        lines.append(
            f"{state:<6}{load:>7.0f}{len(group):>8,}"
            f"{group['ntu_now'].median():>10.3f}"
            f"{group['ntu_ua_fixed'].median():>16.4f}"
            f"{group['ntu_ua_fixed'].min():>9.4f}"
            f"{group['ntu_ua_fixed'].max():>9.4f}"
            f"{group['ntu_ratio'].max():>10.4f}"
        )

    worst = frame.loc[frame["ntu_ratio"].idxmax()]
    lines += [
        "",
        "── 가장 어긋나는 자리 (NTU 비 최대) ──",
        f"  상태={_state_label(worst)} · 부하={worst['load_percent']:.0f}% · "
        f"기구={worst['anomaly_mechanism']} · NTU축={worst['ntu_now']:.0f} · "
        f"T2nd={worst['T_secondary_supply_C']:.0f}℃",
        f"  1차 C={worst['C_primary_W_K']:,.1f} W/K"
        f" · 2차 C={worst['C_secondary_W_K']:,.1f} W/K"
        f" · 1차가 C_min 인가={bool(worst['primary_is_C_min'])}",
        f"  NTU {worst['ntu_now']:.4f} → {worst['ntu_ua_fixed']:.4f}"
        f" (비 {worst['ntu_ratio']:.4f})",
        f"  ε {worst['eps_now']:.6f} → {worst['eps_ua_fixed']:.6f}"
        f" (비 {worst['eps_ratio']:.6f})",
        "",
        "── C4. 온도 이동 **어림** (잰 값이 아니다 · 다시 풀면 더 움직인다) ──",
        "",
        f"{'상태':<6}{'부하%':>7}{'ΔT 중앙[K]':>13}{'ΔT 최소[K]':>13}{'ΔT 최대[K]':>13}"
        f"{'|ΔT| 최대[K]':>14}",
    ]
    for (state, load, _), group in frame.groupby(
        ["state", "load_percent", "ntu"], sort=True
    ):
        lines.append(
            f"{state:<6}{load:>7.0f}"
            f"{group['dT_return_K'].median():>13.5f}"
            f"{group['dT_return_K'].min():>13.5f}"
            f"{group['dT_return_K'].max():>13.5f}"
            f"{group['dT_return_K'].abs().max():>14.5f}"
        )

    lines += [
        "",
        "── C4-B. 신호에 실제로 걸리는 몫 (정상행 이동을 뺀 차이분) ──",
        "",
        f"{'상태':<6}{'부하%':>7}{'차이분 중앙[K]':>16}{'|차이분| 최대[K]':>18}"
        f"{'신호 중앙[K]':>15}{'몫 중앙':>10}{'몫 최대':>10}",
    ]
    for (state, load, _), group in frame.groupby(
        ["state", "load_percent", "ntu"], sort=True
    ):
        if state in ("정상", "유휴"):
            continue
        lines.append(
            f"{state:<6}{load:>7.0f}"
            f"{group['dT_signal_K'].median():>16.5f}"
            f"{group['dT_signal_K'].abs().max():>18.5f}"
            f"{group['signal_K'].median():>15.5f}"
            f"{group['signal_share'].median():>10.3f}"
            f"{group['signal_share'].max():>10.3f}"
        )
    lines.append(
        "  「몫」 = |차이분| / |신호| — 1 을 넘으면 이동이 신호보다 크다는 뜻이다."
    )

    # 7.45 D3 이 기준 C 에서 남긴 **출구온도 12짝**의 조건 — 부하 20 % ·
    # branch ΔP 2.0 · valve ΔP 5.0 · 다중 CDU · 「샘」 크기 0.25 · |Δ| 9.43e-04
    # ~9.59e-04 K · 임계 1e-3 K (PROCEED.md 미해결 표 · 7.47 D4-(다) 재확인).
    #
    # **그 |Δ| 는 세션 7.45 시점의 값이다.** 같은 양을 7.61(펌프 수력동력)이
    # 8.997e-04~9.414e-04 로, 7.72(UA 고정)가 2.213e-03~2.899e-03 으로 옮겼다
    # (수는 그 두 판의 `PROCEED.md` 기록에서 옮긴 것이고 이 스크립트가 다시
    # 잰 것이 아니다 · #86 · 세션 7.75 에 닫혔다). **지금 값으로 바꿔 적지
    # 않는다** — 12짝은 축이 아니라 그때의 실패 건수로만 정의돼 있어 그 집합이
    # 지금 되살아나지 않는다(#84 · 세션 7.77 이 짝마다 식별자와 여유를
    # `docs/session777-massloss-pairs-ac.csv`·`-b.csv` 로 남겨 닫았다 —
    # 앞으로의 판 사이 대조는 서지만 7.45 시점의 12짝은 되살아나지 않는다).
    # 7.72 가 본 무리를 세션 7.73 D3 이 세었더니 32짝이었고, 그 32 가 7.45 의
    # 12 와 같은 집합인지는 서지 않는다(정본 `CLAUDE.md` 게이트 표 4단계).
    # 랙 출구온도도 T_supply 와 **같은 양**만큼 평행이동하므로(ε 는 T_return
    # 식에만 들어간다) 여기 차이분이 그 12짝에 그대로 얹힌다.
    boundary = frame[
        (frame["anomaly_mechanism"] == "massloss")
        & (frame["massloss_size_fraction"] == 0.25)
        & (frame["load_percent"] == 20.0)
        & (frame["branch_dp_rated_mAq"] == 2.0)
        & (frame["valve_dp_rated_mAq"] == 5.0)
        & (frame["cdu_config"] != "single")
    ]
    lines += [
        "",
        "── C4-C. 7.45 D3 기준 C 경계 12짝의 조건에서 (어림) ──",
        f"  해당 행 {len(boundary):,}"
        f" · 차이분 |중앙| {boundary['dT_signal_K'].abs().median():.3e} K"
        f" · |최대| {boundary['dT_signal_K'].abs().max():.3e} K",
        "  견줄 것(세션 7.45 시점): 그 12짝의 |Δ| 는 9.43e-04 ~ 9.59e-04 K 이고"
        " 임계는 1.0e-03 K — 모자란 몫이 4.1e-05 ~ 5.7e-05 K 였다.",
        "  같은 양을 7.61 이 8.997e-04~9.414e-04, 7.72(UA 고정)가"
        " 2.213e-03~2.899e-03 으로 옮겼다 — 위 수를 지금 값으로 읽지 않는다"
        " (#86 · 세션 7.75 에 닫혔다 · 12짝 집합은 지금 되살아나지 않는다 ·"
        " #84 · 세션 7.77 에 닫혔다).",
    ]

    share = frame["primary_is_C_min"].mean()
    lines += [
        "",
        "── C_min 이 어느 쪽인가 (UA 고정이 걸리는지를 가른다) ──",
        f"  1차측이 C_min 인 행: {frame['primary_is_C_min'].sum():,} / {len(frame):,}"
        f" ({share:.1%})",
        "  2차측이 C_min 인 행에서는 C_min 이 고정이므로 **NTU 비가 정확히 1** 이다"
        " — UA 고정으로 바꿔도 그 행은 안 움직인다.",
        "",
        "T_supply 와 T_return 은 **같은 양**만큼 평행이동한다"
        " (ε 가 T_return 식에만 들어가고 T_supply−T_return 은 ε 와 무관하다).",
        "이 표는 지금 해 위의 덧셈이다 — 물성 되먹임을 풀지 않았다.",
    ]
    return "\n".join(lines)


def main() -> int:
    frame = pd.read_csv(_DATASET, usecols=_COLUMNS)
    report = summarize(with_signal_shift(enrich(frame)))
    _OUT.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
