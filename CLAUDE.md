# Kairos (καιρός)

LLM 추론 서빙 최적화 및 KV cache 오프로딩 연구 프로젝트. "언제 내릴 것인가, 언제 올릴 것인가, 어디로 보낼 것인가"라는 타이밍 문제에 답하는 것을 목표로 한다.

상세 내용은 `docs/kairos-project-plan.md` 참고. 이 문서는 해당 플랜을 기반으로 작업할 때 Claude가 빠르게 맥락을 잡기 위한 요약이다.

## 디렉토리 구조

```
docs/          # 플랜, Deep Trace Document(논문 요약), Pain Point 로그
experiments/   # Phase 0~1 실습 코드 (PyTorch, KV cache 구현, vLLM 벤치 스크립트)
notes/         # 주간 작업 로그, 아이디어 메모
```

Phase 2 진입 시 `kairos-bench/`를 분리 생성(또는 별도 오픈소스 레포로 독립)한다.

## 프로젝트 성격

- **연구 프로젝트**(코드베이스가 아님). 현재 이 폴더에는 플랜 문서 하나만 존재한다.
- 최종 목표: SK하이닉스 선행 AI Software Solution — Platform Software 포지션 지원을 위한 경쟁력 확보.
- 타임라인: 약 18개월, 주당 10–20시간.
- 사용자(주영)는 백엔드 엔지니어 출신으로 spring-grpc / resilience4j / Reactor Netty 등 오픈소스 기여 경험과 K8s/CKAD 역량을 보유. Whisper 기반 LocalSub, 도구 프로젝트 Probe 등 기존 작업에서 이어지는 연장선이다.

## 연구 타겟 레이어

```
[Inference Serving Engine]  vLLM, SGLang, TensorRT-LLM    ← 주 타겟
[Memory Hierarchy]          GPU HBM → DRAM → CXL → SSD   ← 확장 타겟
```

모델 아키텍처/학습/CUDA 커널/리눅스 커널은 **비목표**. 이미 만들어진 모델을 효율적으로 서빙하는 시스템 레이어에 집중.

## 핵심 전략: 도구 중심 접근 (Tool-Centric)

단순히 오프로딩 알고리즘을 제안하는 대신, **Kairos Bench**라는 벤치마킹/프로파일링 도구를 먼저 만든다. 이 도구가 ① vLLM 생태계 기여 ② 연구 실험 ③ 논문을 동시에 해결한다.

Kairos Bench 3개 모듈:
1. **오프로딩 전략 비교 프레임워크** — 동일 조건에서 vLLM 내장/LMCache/layerwise 등 전략 비교 (우선순위 1)
2. **KV cache 접근 패턴 프로파일러** — 레이어별/블록별 핫/콜드 분석 (우선순위 2)
3. **메모리 계층 에뮬레이터** — 대역폭/지연 파라미터 기반 CXL/SSD 시뮬레이션 (Phase 4)

## Phase 로드맵

| Phase | 기간 | 목표 | 산출물 |
|------|------|------|--------|
| 0 | 12–16주 | PyTorch + 트랜스포머 + KV cache 직접 구현, 핵심 논문 8편 리딩 | Deep Trace Documents, KV cache 구현 코드 |
| 1 | 10–14주 | vLLM 코드베이스 이해 + 오픈소스 기여 + Pain Point 수집 | PR merged + Pain Point 10+개 |
| 2 | 14–18주 | Kairos Bench 개발 + 비교 실험 + arXiv | Kairos Bench + arXiv 프리프린트 |
| 3 | 8–12주 | 워크숍 논문 제출 + 공동 연구자 확보 | 워크숍 제출 + 협업 관계 |
| 4 | 16–24주 | CXL 확장 + 정식 학회 풀 페이퍼 | EuroSys/ATC/SoCC 제출 |

현재 날짜는 2026-04-15. 최신 동향: vLLM 0.11.0 오프로딩 커넥터, vLLM RFC #33398 layerwise 오프로딩(2026-01), LMCache 3-tier, llm-d 분산 파일시스템 오프로딩.

## 핵심 기술 개념 (작업 시 전제)

- **KV Cache**: 트랜스포머 추론 시 토큰별 K/V 텐서 캐싱. 시퀀스 길이에 선형 비례하여 VRAM 소비.
- **PagedAttention** (vLLM, 2023): KV cache를 블록 단위 페이징으로 관리. 이미 풀린 문제.
- **Continuous Batching** (Orca, 2022): 이미 풀린 문제.
- **KV Cache Offloading**: 활발히 연구 중인 Kairos 타겟. 핫/콜드 판별, 프리페치 타이밍, 워크로드별 전략, 디바이스 조합이 핵심 질문.
- **Prefill(compute-bound) vs Decode(memory-bound)** 구분이 모든 분석의 출발점.

## 실행 환경

- GPU: NVIDIA RTX 30/40 계열 1장 (VRAM 10–16GB)
- Host DRAM: 32GB+ (오프로딩 목적지)
- OS: Ubuntu 22.04+ (WSL2 가능)
- 모델: Llama-2-7B, Qwen2-7B (4-bit quantization)
- 벤치마크: ShareGPT, LongBench, RULER
- 프로파일링: torch.profiler, nvidia-smi, Nsight Systems
- CXL 에뮬레이션: emucxl, CXLMemSim, 또는 Kairos Bench 모듈 3

## 작업 시 주의사항

- **리스크 회피보다 도구 중심 해결**: 연구 질문을 못 찾는 리스크는 Pain Point 기록 → 도구 설계로 해결한다.
- **Top-Tier 학회는 첫 논문 타겟이 아니다**: 워크숍(MLSys/ASPLOS/ISCA/EuroMLSys) + arXiv + 오픈소스 기여 조합으로 접근. EuroSys/ATC/SoCC는 Phase 4.
- **단일 GPU 스케일 한계 수용**: 7B 모델에 집중, 일반화는 공동 연구자 확보 후.
- **에뮬레이션 기반 실험 정당화 가능**: 실물 CXL 대비 최대 26% 지연 차이는 논문에 명시하되 "에뮬에서도 효과 있으니 실물은 더 좋을 것" 스토리로 활용.
- **PR 전략**: 문서/테스트 → 버그 픽스 → 기능 개선 순서로 vLLM 커뮤니티 신뢰 확보.

## SK하이닉스 JD 매핑 (작업 방향 결정 시 참고)

- Platform Software: vLLM/SGLang 최적화, 이종 메모리 AI 시스템 — Kairos로 거의 풀커버.
- AI Model Research: 메모리 접근 패턴/병목 분석 — 부분 커버.
- System Software: KV Cache Offloading, GPU Runtime 메모리 — 부분 커버 (커널/컴파일러는 범위 밖).
