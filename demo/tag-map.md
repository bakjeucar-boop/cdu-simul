# PFD 태그 ↔ 데이터 열 매핑 (시연용)

**세션 7.0 · 2026-08-31 · 가정값 기반 — 실측 아님**

PFD 화면에 붙일 스트림 태그가 **어느 열에서 오는지** 정한 표다. 화면(HTML)은
이 문서가 정한 태그만 그린다.

판정 규칙 — 태그마다 셋 중 하나다.

| 구분 | 뜻 | 처리 |
|---|---|---|
| ⑴ | 52열본에 **열로 있다** | 시연 실행이 같은 계산 경로로 낸다 |
| ⑵ | 열은 없지만 **기존 공개 함수로 계산된다** | 그 함수로 채운다 |
| ⑶ | 둘 다 아니다 | **뺀다.** 새 숫자를 만들지 않는다(절대 규칙 1) |

**52열본을 읽어 쓰는 것이 아니다.** 시연 데이터는 별도 실행이 낸 별도 파일이며,
이 표는 「같은 물리량이 데이터셋의 어느 열에 대응하는가」를 적은 것이다
(52열본은 부하 2수준뿐이라 시연의 부하 축을 담지 못한다 — 세션 7.0 §3⑴).

---

## 1. 채운 태그

| PFD 태그 | 구분 | 데이터셋 열 | 시연 실행의 출처 |
|---|---|---|---|
| 랙별 유량 [L/s] | ⑴ | `rack0..7_flow_Lps` | `FlowDistributionResult.rack_flows_Lps` |
| 랙별 출구온도 [℃] | ⑴ | `rack0..7_outlet_C` | `SteadyStateResult.rack_return_temps_C` |
| CDU 1차측 공급온도 [℃] | ⑴ | `T_supply_C` | `SteadyStateResult.T_supply_C` |
| CDU 1차측 환수온도 [℃] | ⑴ | `T_return_C` | `SteadyStateResult.T_return_C` |
| CDU 1차측 총유량 [L/s] | ⑴ | `total_flow_Lps` | `FlowDistributionResult.total_flow_Lps` |
| 2차측 공급온도 [℃] | ⑴ | `T_secondary_supply_C` | 케이스 입력(고정 경계조건 · 절대 규칙 7) |
| 2차측 배분유량 [L/s] | ⑴ | `secondary_share_Lps` | `PlantSteadyStateResult.secondary_shares_Lps`<br>단일 CDU 는 `HEAT_EXCHANGER.secondary_flow_Lps` 고정 |
| 열교환기 duty [kW] | ⑴ | `hx_duty_kW` | `SteadyStateResult.hx_duty_kW` |
| CDU 부하 [%] | ⑴ | `load_percent` | 케이스 입력 |
| CDU 부하 [kW] | ⑵ | 없음 | `CduCase.rack_load_kW × SCENARIO.racks_per_cdu` |
| 랙별 질량유량 [kg/h] | ⑵ | 없음 | `rack_flows_Lps × ρ × 3.6` (`rack_flows_kgph[0..7]`) |
| CDU 1차측 총질량유량 [kg/h] | ⑵ | 없음 | `total_flow_Lps × ρ × 3.6` (`total_flow_kgph`) |
| 2차측 배분 질량유량 [kg/h] | ⑵ | 없음 | `secondary_share_Lps × ρ × 3.6` (`secondary_share_kgph`) |

⑵ 는 **곱셈 하나씩**이고 새 숫자가 없다 — 부하 [kW] 는 5장 랙당 발열량 × 부하율
× 랙 수, 질량유량 3종은 부피유량 × ρ × 3.6(1 L = 1e-3 m³ · 1 h = 3600 s 환산).
**ρ 는 5-1 「수력 계산의 물성 평가 온도」 그대로 1차측 벌크평균온도에서 CoolProp
으로 얻으며, `secondary_share_kgph` 도 같은 ρ 를 쓴다** — 2차측 스트림의 실제
밀도와는 다르다(2차측 물성 평가 온도를 5-1 이 정한 바 없어 새 규칙을 만들지
않았다 · 세션 7.2). 시연 JSON 의 `meta.mass_flow_density_rule` 에 같은 문장이
실린다.

**부피유량 태그를 대체하지 않는다** — 화면은 L/s 태그를 그대로 쓰고 m³/h 가
필요하면 L/s × 3.6 으로 환산한다. 질량유량 태그는 별도로 얹는 것이다.

**solver 플래그도 함께 싣는다**(절대 규칙 5). 실패 케이스를 버리지 않는다.

| 플래그 | 데이터셋 열 | 시연 실행의 출처 |
|---|---|---|
| 수력 `fsolve` `ier` | `hydraulic_solver_ier` | `FlowDistributionResult.solver_ier` |
| 열 고정점 수렴 | `thermal_solver_converged` | `SteadyStateResult.solver_converged` |
| CDU 바깥 고정점 수렴 | 없음 | `CduSteadyStateResult.outer_solver_converged` |
| 플랜트 상위 `fsolve` `ier` | `plant_solver_ier` | `PlantSteadyStateResult.top_level_solver_ier` |

---

## 2. 뺀 태그 — ⑶ 에 해당한다

### 2차측 환수온도 [℃] — **뺀다**

**모델에 없는 양이다.** 열 없음(⑴ 아님), 그것을 내는 공개 함수 없음(⑵ 아님).

- 2차측은 **고정 경계조건(블랙박스)** 이다 — 절대 규칙 7 · 5-1 「2차측 공급온도」.
  냉각탑·드라이쿨러를 모델링하지 않는다
- `model.hx_capacity_terms` 는 그래서 2차측 물성을 **공급온도에서** 평가한다.
  「2차측 출구온도는 모델에 없으므로 벌크평균을 만들 수 없다 — 선택이 아니라
  강제다」(`model.py`)
- ε-NTU duty 로 역산하면 숫자는 나온다. **그러나 그것은 새 모델이다** — 다중 CDU
  에서는 두 CDU 의 환수가 공유 2차측에서 섞이므로 혼합까지 세워야 하고, 그것이
  「2차측 동특성을 임의로 모델링」이다(절대 규칙 7 · 하지 말 것)

**이 시연에서 뺀다.** 화면에 2차측 환수 스트림을 그리지 않는다.

---

## 3. 있지만 쓰지 않는 태그 — 사람이 뺀 것

세션 7.0 §3⑵ 의 결정이다. **없어서가 아니라 근거가 없어서 안 쓴다.**

| 태그 | 있는가 | 쓰지 않는 이유 |
|---|---|---|
| 펌프 양정 [mAq] | ⑴ `pump_head_mAq` | 아래 |
| 랙 분기 ΔP · 밸브 ΔP [mAq] | ⑵ `FlowDistributionResult.rack_branch_dp_mAq` · `rack_valve_dp_mAq` | 아래 |
| 잔여저항 ΔP [mAq] | ⑵ `FlowDistributionResult.residual_dp_mAq` | 아래 |
| 지점별 정압 [kPa] | ⑶ | 아래 |

이유 셋:

- **계통 정압이 모델에 없다.** 팽창탱크를 모델링하지 않았다 — 정압을 주려면 5장
  밖 숫자가 둘 필요하다
- **잔여저항이 집중저항 하나다**(미해결 #24). CDU 내부를 구간으로 나눈 적이 없어
  「어느 지점의 압력」이라고 붙일 자리가 없다
- 양정·ΔP 는 값 자체는 있으나, 정압이 없는 화면에서 스트림 태그로 걸면 **계통
  압력을 표시한 것처럼 읽힌다**

---

## 4. 보고만 한다 — 이 판에서 고치지 않았다

**52열본의 `heat_capacity_ratio` 열에는 Cr 이 아니라 ε(열교환기 유효도)가 들어
있다.**

- `dataset_plan.PROVENANCE_COLUMNS` 는 그 열을 「Cr — 유도값이다(세션 5-B)」로
  적어 두었다
- `dataset.steady_rows` 는 그 자리에 `result.thermal.hx_effectiveness` 를 넣는다
- 실측: 같은 케이스에서 ε = 0.66705, Cr = 0.99827 로 **다른 값**이다
  (Cr 은 `PlantSteadyStateResult.heat_capacity_ratios()` 가 낸다)

Cr 은 이 시연의 태그가 아니므로 시연 데이터에는 영향이 없다. **다음 판의 화면이
52열본에서 Cr 을 읽으려 하면 안 된다.** 데이터셋 수정은 이 판의 범위 밖이다
(절대 규칙 13 — 목록으로만 보고한다).
