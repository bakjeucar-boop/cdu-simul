# 코드 「leak」 이름 ↔ 기구 대응표 (세션 7.32)

> **[가정값: 업계 공개자료 전형범위 · 프로젝트정리 5장] 실측 아님.**
> 이 문서는 정본 문서 4종이 아니다. 낱말 규약의 **정본은 `CLAUDE.md` 절대 규칙 8**
> 이고, 이 문서는 그 규약을 코드 이름 층에 적용한 결과를 적는다.

## 왜 이 문서가 있는가

`CLAUDE.md` 절대 규칙 8 「낱말 규약」(세션 7.27)은 **「누출」을 상위 개념 전용**으로
쓰고 모델 안의 기구는 언제나 **「막힘」·「샘」**으로 적게 한다. 정본 문서 넷은
세션 7.27~7.30 에 그 규약대로 닫혔다. **코드는 닫지 않았다** — 세션 7.32 가
방안 **(다)+(라)** 를 골랐다:

- **(다)** 두 기구를 **겸하던 이름만** 가른다.
- **(라)** 남는 「leak」 이름은 규약의 **명시적 예외**로 선언하고 이 대응표를 남긴다
  (`intro/` 가 반대 방향의 예외인 것과 같은 취급이다 — 그쪽은 빌드가 「누출」을
  막는다).

### 이것이 최종이 아니다

(가)안(**기구별로 이름을 전부 가른다**)이 틀려서 (다)를 고른 것이 **아니다.**
지금 (가)를 하지 않는 이유는 시점 하나다:

- **데이터셋 열 이름 4개가 이미 세 벌로 나갔다** — `leak_level_percent` ·
  `leak_rack_index` · `leak_cdu_index` · `leak_model`. 48열본 CSV · 52열본 ·
  `results/` 산출물이 그 이름으로 유통됐다.
- **(가)가 강제되는 조건은 하나뿐이다: 「샘」이 데이터셋 생성 경로에 들어가는 날.**
  그때는 두 기구가 같은 표에 서므로 열 이름이 기구를 구분해야 한다.
- 그날은 **스키마 판 다음**이다. 지금 열 이름을 바꾸면 아직 정해지지 않은 스키마에
  맞춰 바꾸는 것이 되어 **두 번 바꾸게 된다.**

즉 이 대응표는 **(가)가 열릴 때까지의 다리**다. (가)를 열 때 변경 목록이 되는
것은 **표 2**(그대로 둔 이름)이지 표 1 이 아니다 — 표 1 은 이미 적용된 것을
적는다(세션 7.51 D1 (가)).

**세션 7.52 가 (가)를 데이터셋 열 셋에 대해 열었다** — 「샘」이 데이터셋 생성
경로에 들어가 두 기구가 같은 표에 서게 됐다(위 두 번째 조건). 표 2 의 첫 행
셋이 표 1 로 옮겨 갔고, `leak_model` 만 표 2 에 남았다.

## 규칙 (세션 7.32 가 적용한 것)

1. **「샘」 전용 이름에서 `leak` 를 뺀다** — 「샘」의 정의어인 **질량손실
   (`massloss`)** 만 남긴다. 파일명도 같다.
2. **두 기구를 겸하던 이름은 중립어로 가른다** — 주입 랙을 가리키는 이름은
   `injection_rack_*` 로 적는다(5-1 이 두 항목 모두 「주입 지점」이라고 부른다).
   **단 `injection_` 은 「빌려 쓰는 자리」를 가르는 데만 쓰고 데이터셋 열에는
   쓰지 않는다** — `assumptions.py` 에 이미 `LEAK.injection_rack_index` 가 있어
   같은 철자가 열과 가정치 둘을 가리키게 된다(세션 7.51 D2 (다)). 그래서 세션
   7.52 는 데이터셋 열에 **`anomaly_`** 를 골랐다: 두 기구를 함께 덮는 중립어이고
   저장소에 같은 철자가 0회다.
3. ~~**「막힘」 이름은 건드리지 않는다**~~ — **세션 7.52 가 (가)안을 열어 이
   규칙이 끝났다.** 근거였던 「데이터셋 열·산출물에 닿아 있다」는 (가)를 여는
   순간 무효가 되도록 쓰인 것이다. 데이터셋 열 셋은 표 1 로 옮겼고, 나머지
   「막힘」 이름(표 2 의 남은 행들)은 여전히 그대로 둔다.

## 표 1 — 바꾼 이름 (old → new)

세션 7.32 가 아래 표의 대부분을 바꿨고, **세션 7.52 가 데이터셋 열 이름 셋을
얹었다**(표 아래 첫 세 행 · 표 2 의 첫 행에서 옮겨 온 것이다).

| 옛 이름 | 새 이름 | 기구 |
|---|---|---|
| `leak_level_percent` | `blockage_level_percent` | 「막힘」 **(7.52)** |
| `leak_rack_index` | `anomaly_rack_index` | **겸함 → 중립 (7.52)** |
| `leak_cdu_index` | `anomaly_cdu_index` | **겸함 → 중립 (7.52)** |
| `leak_massloss.py` | `massloss.py` | 「샘」 |
| `leak_massloss_thermal.py` | `massloss_thermal.py` | 「샘」 |
| `LeakTopology` | `MassLossTopology` | 「샘」 |
| `leak_topologies()` | `massloss_topologies()` | 「샘」 |
| `leak_flow_Lps` | `massloss_flow_Lps` | 「샘」 |
| `leak_flow_bound_Lps()` | `massloss_flow_bound_Lps()` | 「샘」 |
| `leak_bounds_Lps` | `massloss_bounds_Lps` | 「샘」 |
| `leak_sizes_Lps()` | `massloss_sizes_Lps()` | 「샘」 |
| `leak_enthalpy_kW` | `massloss_enthalpy_kW` | 「샘」 |
| `m_leak_kgs` | `m_massloss_kgs` | 「샘」 |
| `Q_leak` · `ṁ_leak` (기호) | `Q_massloss` · `ṁ_massloss` | 「샘」 |
| `balance_residual_with[out]_leak_percent` | `…_with[out]_massloss_percent` | 「샘」 |
| `balance_with[out]_leak_percent` | `balance_with[out]_massloss_percent` | 「샘」 |
| `leak_rack_flow_change_Lps` | `injection_rack_flow_change_Lps` | **겸함 → 중립** |
| `leak_rack_outlet_temp_C` | `injection_rack_outlet_temp_C` | **겸함 → 중립** |
| `leak_rack_outlet_C` | `injection_rack_outlet_C` | **겸함 → 중립** |
| 지역변수 `leaked` | `perturbed` | **겸함 → 중립** |
| 지역변수 `leaks` · `leak` | `massloss_flows` · `loss` | 「샘」 |
| `test_leak_zero_reproduces_closed_loop` | `test_massloss_zero_…` | 「샘」 |
| `test_balance_closes_only_with_leak_enthalpy` | `…_with_massloss_enthalpy` | 「샘」 |

**두 「겸함」의 정체**: `leak_rack_*` 접두는 `leak.py`·`dynamics.py`·
`transport_lag.py` 에서는 **「막힘」**을, `massloss*.py` 에서는 **「샘」**을 가리켰다.
형제처럼 보이는 `leak_rack_flow_change_percent`(막힘)와
`leak_rack_flow_change_Lps`(샘)는 **부호가 정반대**다(미해결 #36). 「샘」 쪽만
중립어로 갈랐고 「막힘」 쪽 이름은 그대로 두었다.

## 표 2 — 그대로 둔 「leak」 이름 (규약의 명시적 예외)

| 이름 계열 | 어디에 | 가리키는 기구 | 왜 안 바꿨나 |
|---|---|---|---|
| `LEAK` · `LeakScenarioAssumptions` · `k_multiplier_levels` · `k_increase_percent_*` | `assumptions.py` | 「막힘」 | 5장 표 항목 「누출 시나리오(「막힘」)」의 전사다. 열 값(`"누출 +50%"`)이 데이터셋에 나갔다 |
| `apply_leak_to_rack()` · `injection_rack_index` | `hydraulics.py`·`assumptions.py` | 「막힘」 | 「샘」 모듈이 **빌려 쓴다** — 아래 「빌려 쓰는 자리」 |
| `leak_case` · `leak_signal` · `LeakSteadySignal` · `leak_rack_*_percent` | `leak.py` | 「막힘」 | 「막힘」 전용이라 오독이 없다. (가)에서 함께 간다 |
| `LeakStepCase` · `integrate_leak_step` · `LeakTransientResult` · `STIMULUS_LEAK_STEP` | `dynamics.py`·`dataset.py` | 「막힘」 | 같음. `STIMULUS_LEAK_STEP` 은 **열 값**이다 |
| `LEAK_MODEL_K_APPROX` · `leak_model` | `dataset.py` · 데이터셋 열 (58열본) | **상위(누출)** | 어느 기구로 모사했는지 밝히는 태그라 상위가 맞다 — 규약 위반이 아니다. 세션 7.52 가 나머지 세 열을 바꿀 때 **이것만 남겼다**: 시연 화면 셋에 411자리로 박혀 있어(7.51 D2 (라)) 시연 화면 판으로 미뤘다 |
| 시험 이름 `test_*leak*` (9파일) | `tests/` | 「막힘」 | 전부 「막힘」 경로다. 이름을 바꾸면 수집 시험 603건의 ID 가 바뀐다 |

## 빌려 쓰는 자리 — 끊지 않았다

「샘」 모듈 둘이 **「막힘」의 이름을 빌려 쓴다**: `apply_leak_to_rack` ·
`LEAK.injection_rack_index` · `LEAK.k_multiplier_levels`.

**공유를 끊지 않는다** — 끊으면 동작이 바뀐다. 그리고 끊을 이유도 없다: 이 모듈들이
하는 일 자체가 **「막힘」 해와 「샘」 해를 같은 조건에서 나란히 놓는 것**이라 「막힘」
경로를 같은 코드로 돌아야 한다(세션 4 C2 와 같은 취지).

다만 **문서 쪽 주입 지점은 세션 7.27 에 이미 기구별로 갈렸다** — 5-1 「「막힘」 주입
지점」과 「「샘」 주입 지점」이다. 랙 번호가 같을 뿐 **성립 이유가 다르다**:

- 「막힘」에서 랙 번호가 결과에 영향하지 않는 것은 **8랙 대칭** 때문이고,
- 「샘」에서는 **애초에 랙에 국소화되지 않기** 때문이다(랙 간 비대칭 2.220e-16 L/s ·
  세션 5.6). 「랙별 유량 하락으로 막힘 랙을 특정」하는 판독은 「샘」에 대응물이 없다.

이 사실을 코드 세 자리에 주석으로 밝혔다: `massloss.py` 머리(빌려 쓰는 이름 블록) ·
`massloss_thermal.injection_rack_outlet_temp_C` · `hydraulics.apply_leak_to_rack`.

## 아직 안 닫힌 자리 (사람이 정한다)

1. ~~**정본 5-1 의 코드 인용 넷**~~ — **세션 7.33 이 닫았다.** 사람의 지시로
   128행 `massloss.massloss_flow_bound_Lps()` · `SWEEP_FRACTIONS`(안 바뀐 이름) ·
   `massloss_thermal.massloss_sizes_Lps()`, 129행 `massloss.massloss_topologies()`
   로 이었다. **이름만 이었고 근거란의 논증·수치는 건드리지 않았다.**
   문서를 고쳤으므로 **프로젝트 지식 재업로드가 필요하다.**
2. ~~**「누출랙」 계열 낱말**~~ — **세션 7.52 가 닫았다.** 붙임 「누출랙」 43 ·
   띄움 「누출 랙」 30 = 73자리를 자리마다 기구를 확인해 갈랐다(정본 3종 10자리
   포함). 세션 7.32 가 적은 자리 인용(「정리 126·127·128·129·134 ·
   `project-overview.md` 268」)은 7.51 재측에서 **정리 126·127·128·134**(129 는 0) ·
   `project-overview.md` **296** 으로 어긋나 있었고, 7.52 가 그 자리를 실제로
   고쳤다. `tests/` 13자리(전부 띄움 철자)도 함께 갔다.
3. **`massloss*.py` 두 모듈의 「누출」** — **세션 7.52 가 표시 문언 18자리만
   닫았다.** docstring 32 · 주석 6 = **56자리 중 나머지 38 은 열려 있다**
   (범위를 사람이 「표시 문언」으로 정했다 — 7.51 D8-8). 미해결 **#59** 에 남는다.
4. **`resistance_proxy`** (`demo_steady.py` 의 JSON 키) — 새 규칙 8 은 「막힘」이
   「샘」의 대용(proxy)이 **아니라고** 한다. 값 문언은 고쳤지만 **키는 그대로 두었다**
   — `demo/`·`dist/` HTML 이 그 키로 읽는다. **세션 7.33 이 산출물을 재생성해
   고친 값 문언이 `demo/demo_steady.json` · `dist/` 에 실렸다 — 키는 여전히 열려
   있다.**

**남은 것은 3 의 잔여(docstring·주석 38자리)와 4 다.** 4 는 `demo/`·`dist/`
재빌드가 걸리므로 `leak_model` 411자리와 함께 **시연 화면 판**에서 본다.

## 관련

- `CLAUDE.md` 절대 규칙 8 (낱말 규약의 정본)
- `AI서버_CDU_디지털트윈_프로젝트정리.md` 5-1 「「막힘」 주입 지점」·「「샘」 주입
  지점」·「「샘」(질량손실) 크기 수준」
- `PROCEED.md` 미해결 **#59**(낱말 규약 잔여) · **#36**(두 기구의 부호가 갈린다)
