# PROCEED.md — 진행 상태

> 이 파일이 "오늘 상태"의 정본이다. `project-overview.md` 「지금 어디인가」 절과
> 어긋나면 이 파일이 맞다. 대화 시작 시 `daily_brief.py` 출력을 첫 메시지에 붙인다.
> **모든 수치는 가정값 기반이며 실측이 아니다.**

## 현재 상태

| 세션 | 단계 | 게이트 통과 여부 | 다음 세션 첫 작업 |
|---|---|---|---|
| 1-A | 저장소 clone · 가상환경 · 저장소 골격 · 5장 가정치 전사 · CoolProp 래퍼 · `PROCEED.md`·`daily_brief.py` 초기 구성 · 문서 4종 커밋 | **해당 없음 (물리 모델 없음)** — 환경 스모크 테스트 5건 통과했으나 이는 feasibility 판정이 아니다 | 1-B: 단일 랙 + 단일 CDU, 2차측 고정온도, 정상상태 모델 구현 → energy balance 오차 <0.1% 검증 |

## 세션 로그

### 세션 1-A · 2026-08-27 · 저장소 clone · 개발환경 · 가정치 전사

**완료한 항목**

- 저장소 clone (`https://github.com/bakjeucar-boop/cdu-simul`, 빈 저장소 · 기본 브랜치 `main`)
- 정본 문서 4종을 저장소 루트에 복사 (`CLAUDE.md` 포함 — Claude Code 자동 인식 위치)
- Python 3.12.10 확인 → 저장소 루트에 `.venv` 생성
- 저장소 골격: `pyproject.toml` · `.gitignore` · `README.md` · `src/cdu_simul/` · `tests/`
- 승인 패키지 설치 (numpy 2.5.2 · scipy 1.18.1 · CoolProp 8.0.0 · pandas 3.0.5 · pytest 9.1.1)
- `src/cdu_simul/assumptions.py` — 프로젝트정리 5장 표 전사 (출처 태그 포함)
- `src/cdu_simul/fluid.py` — CoolProp 물성 래퍼 (밀도·비열)
- `tests/test_environment.py` — 환경 스모크 테스트 5건, 전부 통과
- `PROCEED.md` · `daily_brief.py` 초기 구성
- `CLAUDE.md` 「명령어」 절을 확정된 실제 값으로 채움

**만든/바꾼 파일**

| 파일 | 성격 |
|---|---|
| `pyproject.toml` · `.gitignore` · `README.md` · `src/cdu_simul/__init__.py` | 신규 (골격) |
| `AI서버_CDU_디지털트윈_프로젝트정리.md` · `CLAUDE.md` · `collaboration.md` · `project-overview.md` | 신규 (문서 4종 복사) |
| `CLAUDE.md` 「명령어」 절 | 수정 — **웹 프로젝트 지식 재업로드 필요** |
| `src/cdu_simul/assumptions.py` | 신규 (5장 전사) |
| `src/cdu_simul/fluid.py` | 신규 (CoolProp 래퍼) |
| `tests/test_environment.py` | 신규 (환경 스모크) |
| `PROCEED.md` · `daily_brief.py` | 신규 |

**돌린 것 / 안 돌린 것**

- 돌렸음: `pytest` (5 passed) · `daily_brief.py` · `pip list` · CoolProp 물성 조회
- **안 돌렸음**: 물리 모델 일체. 이 세션에는 `CDUModel`·압력평형(`fsolve`)·ε-NTU·
  `solve_ivp`·유량분배가 없다. energy balance 오차 · T_return 방향성 · 수렴시간 ·
  극단 케이스 발산 — **6장 네 기준 중 어느 것도 이 세션에서 보지 않았다.**

**미해결 항목** — 아래 「미해결 목록」 절 참조.

**다음 세션이 알아야 할 결정**

1. **CoolProp 유체 문자열 위치**: `assumptions.py` 의 `SCENARIO.coolant_coolprop_id`
   에 두고 `fluid.py` 가 참조한다. PG25 선택 자체가 5장 가정치이므로 교체 지점이
   assumptions 한 곳이어야 한다는 판단(절대 규칙 2 · collaboration.md ④).
2. **가정치 파일 형식은 `.py`** (yaml 아님) — 타입힌트·frozen dataclass 에 맞고
   의존성이 늘지 않는다.
3. **범위값은 `Range(low, high, unit)` 로 범위 그대로 보존.** 중앙값 등 대표값을
   임의로 고르지 않았다. `Range` 에 대표값 추출 메서드를 일부러 두지 않았다.
4. **5장 표 원단위를 그대로 보존**(mAq · L/s · ℃ · kW · %). 단위 변환은 하지 않았다.
   유일한 변환은 `fluid.py` 의 ℃→K (CoolProp 호출 경계 전용).
5. **파생값은 만들지 않았다** — 밸브 Cv 역산, 펌프 곡선 계수 H0·a·b, 배관 K값.
   5장 표에 수치가 없다. 1-B에서 필요해지면 사람에게 먼저 확인한다.
6. **Python 3.12.10 사용** (시스템에 3.13·3.12 존재, 3.11 없음). CoolProp 8.0.0 은
   cp312 휠로 설치됐다.
7. **패키지 임포트는 편집설치**(`pip install -e ".[dev]"`)로 해결했다. `PYTHONPATH`
   조작이나 `sys.path` 삽입을 코드에 넣지 않았다.

## 미해결 목록

| # | 한 줄 | 크기 | 영향 |
|---|---|---|---|
| 1 | 밸브 Cv · 배관 K값이 5장 표에 없다 — "ΔP 조건에서 역산"이라고만 적혀 있어 역산 규칙(어느 ΔP·어느 유량 기준)을 사람이 정해야 한다 | M | 1-B 압력평형(`fsolve`)을 못 세운다 — 1-B 착수 전 필요 |
| 2 | 펌프 특성곡선 계수 H0·a·b 가 5장 표에 없다 (형태 `H=H0-aQ-bQ²` 만 있음) | M | 1-B 펌프 모델을 못 세운다 — 1-B 착수 전 필요 |
| 3 | 범위로 적힌 값(정격양정 20~30 mAq, NTU 2~3, 2차측 27~30℃, 헤더 65~80A, 등가길이 20~30m, 랙당 ΔP 2~3 mAq, 밸브 ΔP 3~5 mAq)의 대표값이 미정 | M | 1-B에서 단일 케이스를 돌리려면 대표값 또는 범위 스윕 방침이 필요 |
| 4 | 5장 표의 "CDU 정격 ~750 kW"가 근사 표기 — 열교환기 정격 750 kW 와 같은 값인지 별개 값인지 문서가 명시하지 않음 | S | 정격 대비 부하율 계산 기준이 갈릴 수 있음 |
| 5 | 5장 표 값들의 자체 정합성 미확인 — 랙당 1.94 L/s × 8 = 15.52 L/s 로 펌프 정격 15.5 L/s 와 근사 일치하나, 이 일치가 의도된 것인지 문서에 없음. 또한 15.5 L/s ≠ 930 L/min(= 15.5 L/s 로 일치) 는 정합 | S | 문서 확인만 필요, 코드 영향 없음 |
| 6 | 한글 파일명(`AI서버_...md`)이 `git status`·`git ls-files` 에서 8진 이스케이프로 표기됨(`"AIì..."`). `git -c core.quotepath=false` 로 읽으면 정상 표시되므로 **저장된 파일명 바이트는 정상 UTF-8이고 표시 문제일 뿐**임을 확인했다. 세션 1-A에서는 관측만 하고 이름·git 설정 모두 바꾸지 않았다 (상호참조가 문서 4종에 흩어져 있어 별도 판 필요) | M | 다른 PC·다른 로케일에서 clone 시 문제 소지는 남아 있으나 **확인하지 않았다**. 지금 이 PC의 동작에는 영향 없음 |
| 7 | 물성 조회 기준 압력을 표준대기압(101325 Pa)으로 두었다 — 물리 상수이지 5장 가정치가 아니며, 실제 계통 압력에서 조회해야 하는지 미확인 | S | 비압축성 유체라 밀도·비열 영향은 작을 것으로 보이나 **확인하지 않았다** |
| 8 | `project-overview.md` 「파일럿 종료 조건」이 미정 상태 — 문서 자체가 세션 1-B 이후 사람이 확정한다고 적고 있다 | L | 세션 6 진입 판단 불가 (문서대로 1-B 이후 확정) |
| 9 | 이 PC의 git 이 `core.autocrlf` 로 LF→CRLF 변환 경고를 낸다 (`LF will be replaced by CRLF`). 세션 1-A에서는 `.gitattributes` 를 추가하지 않았다 — 지시 범위 밖(절대 규칙 13) | S | 여러 PC를 오갈 때 줄바꿈만 바뀐 diff 가 생길 소지. 실제로 발생하는지는 **확인하지 않았다** |
