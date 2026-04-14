# Kairos (καιρός)

> **"적절한 때에 적절한 곳으로"**
> LLM 추론 서빙 최적화 및 KV cache 오프로딩 연구 프로젝트

---

## 1. 프로젝트 개요

### 1.1 프로젝트 정의

Kairos는 LLM 추론 서빙 시스템에서 KV cache의 메모리 계층 간 이동(오프로딩/로딩)을 최적화하는 연구 프로젝트다. "언제 내릴 것인가, 언제 올릴 것인가, 어디로 보낼 것인가"라는 세 가지 타이밍 질문에 답하는 것이 핵심이다.

### 1.2 이름의 의미

Kairos는 그리스 철학에서 "적절한 때"를 뜻한다. 크로노스(Chronos, 물리적 시간)와 대비되는 개념으로, 단순한 시간의 흐름이 아니라 "결정적 순간"을 의미한다. KV cache 오프로딩 연구의 본질이 정확히 이것이다.

- 프로파일링으로 측정하는 것 = 크로노스 (지연 ns, 대역폭 GB/s)
- 그 데이터를 바탕으로 최적의 순간을 판단하는 것 = 카이로스

### 1.3 배경: 왜 이 연구가 필요한가

LLM 추론 시 모든 토큰의 Key/Value 텐서를 캐싱하여 재계산을 피하는 것이 KV cache다. 모델이 커지고 컨텍스트 윈도우가 길어지면서(32K → 128K → 1M+) KV cache가 GPU VRAM(HBM)을 초과하는 상황이 빈번해지고 있다.

이 문제는 업계 전체의 과제다.

- 클라우드 업체: 같은 GPU에서 더 많은 요청 처리 → GPU당 수익 증가
- AI 서비스 업체: 추론 비용 절감 → 토큰당 단가 경쟁력
- 엔터프라이즈: 사내 LLM 서빙 시 제한된 GPU로 최대 효율 확보
- 오픈소스 커뮤니티: 개인/소규모 팀이 큰 모델을 돌릴 수 있는 민주화

SK하이닉스 관점에서는 HBM, DRAM, CXL, NAND를 모두 만드는 회사로서, "멀티 티어 메모리 구조가 최적"이라는 결론이 나오면 전 제품 라인의 시장이 확대된다. 오프로딩 소프트웨어가 이 구조의 실용성을 증명하는 핵심이다.

---

## 2. 목표

### 2.1 최종 목표

- SK하이닉스 선행 AI Software Solution — Platform Software 포지션에 경쟁력 있는 지원자가 되는 것
- 해당 JD의 핵심 요구사항: "vLLM, SGLang, Nvidia Dynamo 등 추론 시스템 최적화 연구 및 개발", "이종 메모리(HBM, DRAM, CXL 등) 기반 AI 시스템 설계 및 개발"

### 2.2 단계별 목표

| 단계 | 목표 | 산출물 |
|------|------|--------|
| Phase 0 | 추론 서빙 시스템의 기반 지식 구축 | Deep Trace Documents, KV cache 구현 코드 |
| Phase 1 | vLLM 코드베이스 이해 + 오픈소스 기여 + 연구자 Pain Point 수집 | PR merged + Pain Point 목록 |
| Phase 2 | 오프로딩 벤치마킹/프로파일링 도구 개발 + 비교 실험 + arXiv | Kairos Bench + arXiv 논문 1편 |
| Phase 3 | 워크숍 논문 제출 + 공동 연구자 확보 | 워크숍 제출 1건 + 협업 관계 구축 |
| Phase 4 | CXL 확장 + 정식 학회 풀 페이퍼 | EuroSys/ATC/SoCC 제출 1건 |

### 2.3 비목표 (하지 않는 것)

- LLM 모델 아키텍처 자체를 설계하거나 학습하는 것
- 새로운 attention 메커니즘을 제안하는 것
- CUDA 커널을 밑바닥부터 작성하는 것
- 리눅스 커널 메모리 관리 코드를 직접 수정하는 것

---

## 3. 연구 대상: LLM 추론 서빙 스택

### 3.1 전체 스택 구조

```
[Applications]              ChatGPT, Claude, 사내 LLM 서비스
       ↓
[Cluster Orchestration]     NVIDIA Dynamo, llm-d, aibrix
       ↓
[Inference Serving Engine]  vLLM, SGLang, TensorRT-LLM    ← Kairos 핵심 타겟
       ↓
[Compute Framework]         PyTorch, FlashAttention, CUDA
       ↓
[Memory Hierarchy]          GPU HBM → DRAM → CXL → SSD   ← Kairos 확장 타겟
```

### 3.2 Kairos가 집중하는 레이어

Inference Serving Engine 레이어를 주 연구 대상으로 하며, Memory Hierarchy 레이어로 확장한다. 모델 자체를 만드는 것이 아니라, 이미 만들어진 모델이 가장 효율적으로 서빙되는 시스템을 연구한다.

### 3.3 핵심 기술 개념

**KV Cache란:** 트랜스포머 모델이 추론할 때, 각 토큰의 Key/Value 텐서를 캐싱하여 다음 토큰 생성 시 재계산을 피하는 기법이다. 시퀀스 길이에 비례하여 선형으로 증가하며, GPU VRAM의 주요 소비자다.

**Continuous Batching (이미 풀린 문제):** 여러 요청의 토큰을 하나의 배치로 묶어 GPU 활용률을 높이는 기법. Orca 논문(2022)에서 제안.

**PagedAttention (이미 풀린 문제):** KV cache를 OS의 가상 메모리처럼 블록 단위로 관리하여 메모리 단편화를 해결하는 기법. vLLM 논문(2023)에서 제안.

**KV Cache Offloading (활발히 연구 중 — Kairos 타겟):** GPU VRAM에 있는 KV cache를 DRAM/CXL/SSD 등 하위 메모리 계층으로 이동시켜 VRAM 부족을 해결하는 기법. 핵심 연구 질문:
- 어떤 KV cache 블록이 핫(자주 접근)하고 콜드(드물게 접근)한가?
- 언제 오프로딩하고 언제 프리페치(미리 올려놓기)할 것인가?
- 워크로드 종류에 따라 최적 전략이 어떻게 달라지는가?
- 어떤 디바이스 조합(HBM+DRAM, HBM+DRAM+SSD, HBM+CXL+SSD)이 최적인가?

### 3.4 메모리 계층 특성

```
디바이스       대역폭       지연       용량         역할
GPU HBM      수 TB/s     매우 낮음   80-192GB    핫 데이터 (활성 KV cache)
Host DRAM    ~100 GB/s   낮음       128-2TB     웜 데이터 (swap 대상)
CXL 메모리    ~80 GB/s    중간       TB 단위      웜/콜드 데이터 (확장 메모리)
NVMe SSD     ~7 GB/s     높음       TB 단위      콜드 데이터 (아카이브)
```

### 3.5 vLLM 내부 구조 (연구 대상)

vLLM은 LLM 모델을 내부에 로드하고, 스케줄러 + 블록 매니저 + KV cache 관리 시스템으로 감싸서 서빙하는 PyTorch 애플리케이션이다.

```
vLLM 프로세스 (Python/PyTorch)
├── HTTP API 서버 (요청 수신, OpenAI 호환)
├── 스케줄러 (어떤 요청을 언제 처리할지 결정)
│   └── waiting / running / swapped 큐 관리
│   └── preemption 전략 (recompute vs swap)
├── 블록 매니저 (KV cache 메모리 할당/해제)
│   └── 논리 블록 ↔ 물리 블록 매핑 (PagedAttention)
├── 모델 러너
│   └── LLM 모델 (예: Llama-2-7B, PyTorch 모델)
│   └── 커스텀 Attention 커널 (흩어진 블록에서 KV 읽기)
└── KV cache 오프로딩 커넥터
    └── GPU VRAM ↔ Host DRAM 비동기 전송
    └── vLLM 0.11.0부터 도입, 0.12.0에서 대폭 개선
```

### 3.6 경쟁 환경

**추론 서빙 엔진:**
- vLLM: GitHub 74.9K stars, de facto 표준, 가장 큰 커뮤니티. Kairos의 주 타겟.
- SGLang: RadixAttention, 멀티턴 워크로드에서 vLLM 대비 ~29% 높은 throughput. 비교 연구 대상.
- TensorRT-LLM: NVIDIA GPU 전용, 최고 성능이나 커뮤니티 약함.
- LMDeploy: C++ TurboMind 엔진, 양자화 모델 서빙에 강점.

**클러스터 오케스트레이션 (Phase 4 이후 확장 가능):**
- NVIDIA Dynamo: 2026년 1.0 릴리즈, 가장 강한 모멘텀. Disaggregated serving.
- llm-d: K8s 네이티브, Red Hat/Google/IBM/NVIDIA 공동 참여. K8s 경험과 직접 연결.
- aibrix: ByteDance 출신, vLLM 산하. LoRA + 분산 KV cache.

---

## 4. 실행 환경

### 4.1 하드웨어

- GPU: NVIDIA RTX 30/40 계열 1장 (VRAM 10-16GB)
- Host DRAM: 32GB 이상 권장 (64GB 이상 이상적, 오프로딩 목적지)
- NVMe SSD: 3-tier 오프로딩 실험용
- OS: Ubuntu (WSL2 또는 네이티브)

### 4.2 소프트웨어 스택

```
OS: Ubuntu 22.04+ (WSL2 지원)
GPU Driver: NVIDIA Driver
CUDA Toolkit: 시스템 설치
PyTorch: pip install (CUDA 바이너리 자동 매칭)
vLLM: 소스 빌드 (수정 가능한 상태)
모델: Llama-2-7B, Qwen2-7B (4-bit quantization으로 RTX GPU에서 구동)
벤치마크: ShareGPT, LongBench, RULER
프로파일링: torch.profiler, nvidia-smi, Nsight Systems
CXL 에뮬레이션 (Phase 4): emucxl (GitHub 오픈소스) 또는 대역폭 제한 주입
```

### 4.3 시간 투자

- 주당 10-20시간
- 총 타임라인: 약 18개월

---

## 4.5 최신 오프로딩 전략 동향 (2026년 4월 기준)

### 4.5.1 Prefix cache 오프로딩 (현재 프로덕션 주류)

vLLM 0.11.0에서 도입된 오프로딩 커넥터가 대표적. 완료된 요청의 KV cache를 CPU DRAM으로 내려놓고, 같은 프리픽스를 가진 새 요청이 오면 재계산 없이 다시 올린다. LMCache가 이를 NVMe SSD까지 확장하여 3-tier(GPU → DRAM → SSD) 구조를 제공한다.

### 4.5.2 Layerwise 오프로딩 (최신 — 주목할 방향)

vLLM RFC #33398 (2026년 1월). 기존이 "요청 단위" 오프로딩이라면, 이건 "레이어 단위"로 쪼갠다. 모델의 일부 레이어 KV cache만 GPU에 두고 나머지는 CPU로 내리며, 각 레이어 forward pass 직전에 비동기로 올리고 끝나면 다시 내리는 파이프라인 방식이다. DeepSeek-V3 같은 sparse attention 모델에서 특히 효과적이다.

### 4.5.3 분산 파일시스템 기반 오프로딩 (클러스터 레벨)

llm-d가 제안. KV 블록을 공유 파일시스템에 파일로 저장하여, 여러 vLLM 인스턴스가 캐싱된 프리픽스를 공유한다. 서버 재시작 시에도 캐시가 유지되며, 새 노드 추가 시 기존 캐시를 즉시 활용할 수 있다.

### 4.5.4 Sparse attention + 오프로딩 결합 (열린 연구 영역)

2026년 4월 논문에서 기존 오프로딩 기법들이 context-intensive task(입력에서 대량 정보를 추출하는 작업)에서 심각한 성능 저하를 보임을 밝혔다. 워크로드별 adaptive하게 로드 양을 조절하는 것이 future work로 제시됨.

### 4.5.5 Kairos 관점 매핑

```
이미 구현됨 (이해 대상):     Prefix cache 오프로딩 (vLLM + LMCache)
활발히 개발 중 (기여 대상):   Layerwise 오프로딩, 3-tier 오프로딩
열린 연구 질문 (논문 대상):   워크로드별 adaptive 전략, context-intensive task 정확도, CXL 티어 최적화
```

---

## 4.6 진입 전략: 도구 중심 접근법 (Tool-Centric Approach)

### 4.6.1 전략 개요

단순히 오프로딩 알고리즘을 제안하는 것이 아니라, 오프로딩 전략을 테스트하고 비교하기 쉽게 만드는 도구를 먼저 만든다. 이 도구가 vLLM 생태계 기여, 연구 실험, 논문의 세 가지를 동시에 해결한다.

### 4.6.2 현재 연구자들의 Pain Point

- 벤치마킹이 제각각: 논문마다 다른 모델, 워크로드, 메트릭으로 실험하여 기법 간 직접 비교가 불가능
- 메모리 계층 시뮬레이션 비표준: CXL 대역폭 테스트 시 각자 알아서 지연 주입하거나 에뮬레이션 환경 구축. 재현 불가
- 프로파일링 파편화: KV cache 접근 패턴 분석 시 vLLM 코드를 직접 수정해 로그를 심어야 함

### 4.6.3 Kairos Bench (가칭) 구상

```
모듈 1: 오프로딩 전략 비교 프레임워크
  - 동일 워크로드/모델/메트릭으로 여러 전략을 한 번에 비교
  - vLLM 내장 오프로딩, LMCache, layerwise 등을 플러그인으로 교체
  - 결과를 표준화된 포맷으로 출력 (표, 그래프 자동 생성)

모듈 2: 메모리 계층 에뮬레이터
  - 대역폭/지연 파라미터만 넣으면 CXL, 느린 SSD 등을 시뮬레이션
  - vLLM의 오프로딩 커넥터에 끼워넣을 수 있는 인터페이스
  - "대역폭 X, 지연 Y일 때 throughput은?" sweep 실험 자동화

모듈 3: KV cache 접근 패턴 프로파일러
  - 어떤 레이어의 어떤 블록이 언제 접근되는지 트레이싱
  - 워크로드별 핫/콜드 분포 시각화
  - 오프로딩 전략 설계에 직접 쓸 수 있는 분석 데이터 제공
```

### 4.6.4 도구 중심 접근의 장점

1. 도구 자체가 논문이 된다: 시스템 학회에서 벤치마킹 프레임워크/분석 도구 논문이 잘 받아들여짐. "N개 전략을 통일 조건에서 비교한 결과 + 도구 공개"가 논문 1편.
2. 다른 연구자가 쓰면 인용이 쌓인다: "Kairos Bench를 사용하여 실험했다"라는 인용이 커뮤니티 영향력으로 이어짐.
3. 공동 연구자가 알아서 찾아온다: "이 도구로 내 전략 테스트해봤는데 결과 공유하고 싶다"는 역학.
4. 백엔드 엔지니어의 강점을 최대 활용: 도구 설계, API 설계, 재현 가능한 환경 구축은 정확히 주영님의 강점 영역.

### 4.6.5 Phase 흐름 변화

```
Phase 1:
  vLLM 코드 읽기 + 버그 픽스/기능 개선 PR (커뮤니티 신뢰 확보)
                                          ↘
                                동시에 "뭐가 불편한가" Pain Point 기록
                                          ↓
Phase 2:
  Kairos Bench 설계/구현 + 버그 픽스/기능 개선 PR 계속
       ↓
  도구로 비교 실험 수행 → arXiv 프리프린트
       ↓
Phase 3:
  도구 + 비교 실험 결과를 워크숍 논문으로 제출
  도구 사용자로부터 공동 연구자 확보
```

---

## 5. Phase 0: 기반 구축 (12-16주)

### 5.1 목적

트랜스포머 추론 과정에서 KV cache의 역할을 코드 레벨로 체감하고, 핵심 논문을 읽어 연구 지형을 파악한다.

### 5.2 Step 0-1: PyTorch 기초 (2주)

**학습 범위:**
- 텐서 연산, broadcasting, GPU 메모리 관리 (torch.cuda.memory_allocated, torch.cuda.max_memory_allocated)
- torch.profiler를 이용한 연산별 시간/메모리 프로파일링
- torch.nn.Module 구조, forward/backward 흐름

**실습 과제:**
- 2-layer MLP 직접 작성 → GPU 학습 → torch.profiler로 메모리 측정
- 배치 사이즈 변경에 따른 GPU 메모리 패턴 관찰

**자료:**
- PyTorch 공식 "Deep Learning with PyTorch: A 60 Minute Blitz"
- PyTorch Profiler 공식 문서

**완료 기준:**
- torch.profiler로 레이어별 메모리 사용량을 측정하고 해석할 수 있다
- GPU 메모리 할당/해제 패턴을 코드에서 추적할 수 있다

### 5.3 Step 0-2: 트랜스포머 + KV cache 직접 구현 (3-4주)

전체 커리큘럼에서 가장 중요한 단계다. 추론 서빙 최적화 연구의 모든 것이 여기서 시작된다.

**학습 범위:**
- Scaled Dot-Product Attention 수식과 구현
- Multi-Head Attention에서 Q, K, V 텐서의 shape 변환
- Positional Encoding (RoPE 포함)
- Prefill (전체 프롬프트 처리, compute-bound) vs Decode (토큰별 생성, memory-bound) 차이

**핵심 구현 과제:**
1. 소규모 GPT (6-layer, 8-head, d_model=512) 직접 구현
2. KV cache 없이 추론하는 naive 버전 작성
3. KV cache를 레이어별로 저장/재사용하는 버전 작성
4. 두 버전의 메모리 사용량, 추론 시간 비교 측정
5. 시퀀스 길이를 128 → 512 → 2048로 늘려가며 KV cache 메모리 증가 패턴 관찰

**자료:**
- Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out" (YouTube)
- Jay Alammar, "The Illustrated Transformer"
- 논문: "Attention Is All You Need" (Vaswani et al., 2017)

**완료 기준 (아래 질문에 코드를 보며 답할 수 있어야 한다):**
- KV cache 텐서의 shape (batch, num_heads, seq_len, head_dim)에서 각 차원이 메모리에 미치는 영향은?
- 시퀀스 길이가 2배 되면 KV cache 메모리는 얼마나 증가하는가? 왜?
- 배치 내 시퀀스 길이가 다를 때 padding이 왜 메모리 낭비인가?
- Prefill과 Decode 단계에서 GPU 연산 패턴이 어떻게 다른가?

### 5.4 Step 0-3: 핵심 논문 리딩 (4-6주)

매주 1-2편, 아래 순서대로 읽는다. 각 논문이 어떤 문제를 풀려고 했는지가 순서대로 이해된다.

**필수 — 추론 서빙 기초:**

| # | 논문 | 핵심 개념 | 읽는 이유 |
|---|------|-----------|-----------|
| 1 | Orca (Yu et al., 2022) | Continuous Batching | 배치 처리의 근본 문제와 해결 |
| 2 | vLLM / PagedAttention (Kwon et al., 2023) | 페이징 기반 KV cache 관리 | 현재 de facto 표준 |
| 3 | SGLang / RadixAttention (Zheng et al., 2024) | Prefix KV cache 재활용 | vLLM 이후의 진화 방향 |
| 4 | SpecInfer (Miao et al., 2023) | Draft-then-verify | Decode 단계 최적화 핵심 기법 |

**필수 — KV cache 최적화:**

| # | 논문 | 핵심 개념 | 읽는 이유 |
|---|------|-----------|-----------|
| 5 | FlexGen (Sheng et al., 2023) | GPU-CPU-SSD 3-tier 오프로딩 | 오프로딩 기초 프레임워크 |
| 6 | ShadowKV (Sun et al., 2024) | Sparse attention + offloading | 최신 오프로딩 접근법 |
| 7 | InfiniGen (Lee et al., 2024) | 부분 KV prefetch | Prefetch 전략 연구 |
| 8 | ScoutAttention (2026, DAC) | GPU-CPU 협업 attention | 최신 연구 동향 |

**선택 — 확장 읽기:**
- FlashAttention (Dao et al., 2022): GPU 메모리 계층 최적화의 교과서
- Splitwise (Patel et al., 2024): Disaggregated serving
- LMCache (2025): Nvidia Dynamo와 통합된 KV cache 시스템

**논문 정리 형식 (Deep Trace Document):**
1. 어떤 문제를 풀려고 하는가? (1문장)
2. 핵심 아이디어 (1문단)
3. 시스템 아키텍처 다이어그램 (직접 그리기)
4. 실험 셋업: 모델, GPU, 워크로드
5. 핵심 결과 수치 (throughput, latency, memory 절감)
6. 한계점 / 저자가 못 다룬 것

**완료 기준:**
- 8편의 Deep Trace Document 완성
- 각 논문의 핵심 아이디어를 3문장 이내로 설명할 수 있다
- 논문 간 관계(어떤 논문이 어떤 논문의 한계를 해결했는지)를 설명할 수 있다

### 5.5 Step 0-4: GPU 메모리 계층 + 프로파일링 (2주)

**학습 범위:**
- GPU 메모리 계층: HBM ↔ L2 Cache ↔ SRAM ↔ Register
- Compute-bound vs Memory-bound 연산 구분
- HBM bandwidth가 decode의 병목인 이유
- CUDA 기초 개념: kernel, thread block, grid (개념 수준)

**실습 과제:**
- nvidia-smi로 GPU 메모리/utilization 모니터링
- torch.profiler + TensorBoard로 추론 과정 시각화
- Hugging Face 모델(Llama-2-7B 등)로 추론 메모리 프로파일링
- Nsight Systems로 GPU kernel 타임라인 확인 (선택)

**자료:**
- NVIDIA CUDA C++ Programming Guide (Memory Hierarchy 부분)
- FlashAttention 논문 Section 2의 GPU 메모리 계층 설명

**완료 기준:**
- "이 연산이 memory-bound인 이유"를 프로파일링 데이터로 설명할 수 있다
- GPU 메모리 프로파일링 도구를 사용하여 추론 병목 지점을 식별할 수 있다

### 5.6 Phase 0 종합 완료 기준

- [ ] KV cache 있는/없는 추론 코드를 직접 작성하고 메모리 차이를 수치로 설명 가능
- [ ] 필수 논문 8편의 Deep Trace Document 완성
- [ ] GPU 메모리 프로파일링 도구로 추론 병목 지점 식별 가능
- [ ] Prefill vs Decode의 차이를 코드와 프로파일링 데이터로 설명 가능

---

## 6. Phase 1: vLLM 코드베이스 진입 + 오픈소스 기여 (10-14주)

### 6.1 목적

vLLM 내부 동작을 코드 레벨에서 이해하고, 오픈소스 기여를 통해 커뮤니티 인지도를 확보하며, 연구자/개발자로서 오프로딩 테스트의 불편함(Pain Point)을 체감하고 기록한다.

### 6.2 Step 1-1: vLLM 코드 리딩 (4-5주)

**읽기 순서 (의존 관계 반영):**

1주차 — 진입점 + 전체 구조:
- vllm/entrypoints/llm.py (사용자 API)
- vllm/engine/llm_engine.py (엔진 메인 루프)
- 전체 디렉토리 구조 지도 그리기

2주차 — 스케줄러:
- vllm/core/scheduler.py (요청 스케줄링)
- waiting / running / swapped 큐 관리
- preemption 전략 (recompute vs swap)

3주차 — 메모리 관리 (핵심):
- vllm/core/block_manager.py (블록 할당/해제)
- PagedAttention 실제 구현
- 물리 블록 ↔ 논리 블록 매핑

4주차 — KV cache + Attention 연산:
- vllm/attention/ (attention backend 구현체)
- vllm/worker/ (GPU worker 모델 실행 흐름)
- KV cache가 어떤 텐서에 어떻게 저장되는지

5주차 — 오프로딩 커넥터 (Phase 2 준비):
- vllm/distributed/kv_transfer/ (KV 오프로딩 인터페이스)
- CPU offloading backend 구현
- 비동기 전송 메커니즘

**완료 기준:**
- 요청이 들어오면 스케줄러가 어떤 과정을 거쳐 GPU에 배치를 올리는지 설명 가능
- 메모리 부족 시 preemption 동작 과정 설명 가능
- KV cache 블록의 생명주기(할당 → 사용 → 해제/swap) 설명 가능

### 6.3 Step 1-2: 로컬 환경 구동 + 실험 (2-3주)

**실습 과제:**
1. vLLM 소스 빌드 → RTX GPU에서 실행 (Llama-2-7B 또는 Qwen2-7B, 4-bit quantization)
2. ShareGPT 데이터셋으로 벤치마크 환경 구성
3. 핵심 메트릭 측정: throughput (req/s), TTFT, TPOT
4. 동시 요청 수 변경 실험
5. 스케줄러 파라미터 튜닝 실험 (max_num_seqs, max_num_batched_tokens)
6. GPU memory utilization 변화 관찰
7. preemption 발생 빈도와 성능 영향 분석

**완료 기준:**
- vLLM을 로컬 GPU에서 소스 빌드하고 벤치마크 실행 가능
- 스케줄러 파라미터 변경이 throughput/latency에 미치는 영향을 수치로 설명 가능

### 6.4 Step 1-3: 오픈소스 기여 (4-6주)

**기여 전략 (기존 spring-grpc, resilience4j 기여 경험 적용):**

1단계 — Good First Issue / 버그 픽스 (1-2주):
- GitHub Issues에서 "good first issue" 라벨 확인
- 문서 개선, 에러 메시지 개선 등 저위험 PR
- 코드 리뷰 과정을 통해 프로젝트 컨벤션 학습

2단계 — 테스트/벤치마크 기여 (1-2주):
- 특정 워크로드 벤치마크 추가
- 엣지 케이스 테스트 추가
- 내부 동작의 예상치 못한 행동 발견 가능

3단계 — 기능 개선 PR (2주):
- Phase 1-2에서 발견한 비효율에 대한 개선 PR
- 예: 스케줄링 정책 개선, 메모리 관리 최적화, 프로파일링 도구 추가

### 6.5 Step 1-4: Pain Point 기록 (Phase 1 전체에 걸쳐 병행)

Phase 1의 모든 과정에서 "연구자/개발자로서 뭐가 불편한가"를 의식적으로 기록한다. 이것이 Phase 2에서 Kairos Bench의 요구사항이 된다.

**기록 관점:**
- 오프로딩 전략 A와 B를 같은 조건에서 비교하고 싶을 때 불편한 점
- 메모리 대역폭/지연을 바꿔가며 실험할 때 필요한 수작업
- KV cache 접근 패턴을 확인하려 할 때 vLLM 코드를 어떻게 수정해야 했는지
- 벤치마크 결과를 정리하고 시각화하는 데 필요한 반복 작업
- 다른 논문의 실험을 재현하려 할 때 겪는 어려움

**기록 형식:** 마크다운 문서에 날짜별로 누적. 각 항목에 "현재 어떻게 해결하고 있는가"와 "도구가 있다면 어떻게 해결할 수 있을까"를 함께 적는다.

**완료 기준:**
- [ ] vLLM에 최소 1개 PR merged (또는 의미 있는 리뷰를 받은 PR open)
- [ ] vLLM 커뮤니티(Discord/GitHub)에서 교류 시작
- [ ] Pain Point 목록 최소 10개 이상 수집
- [ ] Pain Point 중 Kairos Bench로 해결할 상위 3개 선정

---

## 7. Phase 2: Kairos Bench 개발 + 비교 실험 + arXiv (14-18주)

### 7.1 목적

Phase 1에서 수집한 Pain Point를 해결하는 벤치마킹/프로파일링 도구(Kairos Bench)를 개발하고, 이 도구를 사용하여 기존 오프로딩 전략들을 통일된 조건에서 비교 실험한 뒤, 도구 + 실험 결과를 arXiv 프리프린트로 발표한다.

### 7.2 Step 2-1: Kairos Bench 설계 (2-3주)

Phase 1에서 수집한 Pain Point를 바탕으로 도구의 범위와 인터페이스를 결정한다.

**설계 원칙:**
- vLLM 생태계에 자연스럽게 통합되는 구조 (독립 프로젝트가 아니라 vLLM과 함께 쓰는 도구)
- 최소 기능으로 시작하여 점진적 확장 (3개 모듈 중 가장 가치 높은 것부터)
- 재현 가능성 최우선 (실험 설정 → 실행 → 결과 수집 → 시각화가 한 커맨드로)

**모듈 우선순위 결정:**

```
우선순위 1 — 오프로딩 전략 비교 프레임워크:
  연구자 입장에서 가장 즉각적 가치. 동일 조건에서 전략 간 비교가 불가능한 것이
  이 분야의 가장 큰 Pain Point.
  - 전략 플러그인 인터페이스 설계
  - 표준 워크로드 프로파일 정의
  - 결과 출력 포맷 정의

우선순위 2 — KV cache 접근 패턴 프로파일러:
  비교 실험의 "왜?"를 설명하는 데 필수.
  - vLLM에 최소 침습적으로 트레이싱 코드 삽입
  - 레이어별/블록별 접근 빈도, 타이밍 수집
  - 핫/콜드 분포 시각화

우선순위 3 — 메모리 계층 에뮬레이터:
  CXL 실험에 필요하지만 Phase 4에서 본격 사용.
  - 대역폭/지연 파라미터 기반 throttling
  - vLLM 오프로딩 커넥터와 호환되는 백엔드 인터페이스
```

**완료 기준:**
- 도구의 아키텍처 문서 작성 완료
- 모듈 1(전략 비교)의 인터페이스 설계 완료
- 지원할 오프로딩 전략 목록 확정 (vLLM 내장, LMCache, layerwise 등)

### 7.3 Step 2-2: Kairos Bench 구현 (6-8주)

**모듈 1 구현 (4-5주):**
- 전략 플러그인 시스템: 오프로딩 전략을 설정 파일로 교체 가능하게
- 워크로드 러너: ShareGPT, LongBench, RULER 등을 표준화된 방식으로 실행
- 메트릭 수집기: throughput, TTFT, TPOT, GPU memory utilization, PCIe 전송량
- 결과 리포터: 전략별 비교 표/그래프 자동 생성
- 재현 스크립트: 실험 설정을 YAML로 정의, 한 커맨드로 전체 실험 실행

**모듈 2 구현 (2-3주):**
- vLLM의 attention/scheduler 경로에 트레이싱 hook 삽입
- 레이어별/블록별 접근 기록을 시계열 데이터로 수집
- 핫/콜드 히트맵 시각화 (matplotlib 또는 간단한 웹 UI)

**실험 인프라:**
- GPU: RTX (VRAM 10-16GB)
- DRAM: 32GB+ (오프로딩 목적지)
- NVMe SSD: 3-tier 실험
- 소프트웨어: vLLM 소스 빌드 (수정 가능)
- 모델: Llama-2-7B, Qwen2-7B (4-bit quantization)

**병행:** vLLM 버그 픽스/기능 개선 PR 계속 제출하여 커뮤니티 신뢰 유지

**완료 기준:**
- [ ] Kairos Bench 모듈 1(전략 비교)이 동작하여 최소 2개 전략을 비교 가능
- [ ] Kairos Bench 모듈 2(프로파일러)가 동작하여 KV cache 접근 패턴 시각화 가능
- [ ] GitHub에 오픈소스로 공개

### 7.4 Step 2-3: Kairos Bench를 사용한 비교 실험 (3-4주)

Kairos Bench로 기존 오프로딩 전략들을 통일된 조건에서 비교한다.

**실험 매트릭스:**

```
전략 축:
  - vLLM 내장 CPU 오프로딩
  - LMCache (CPU + SSD)
  - Layerwise 오프로딩 (vLLM RFC #33398 기반)
  - Baseline (오프로딩 없음)

워크로드 축:
  - 짧은 멀티턴 대화 (ShareGPT)
  - 긴 문맥 QA (LongBench)
  - Context-intensive task (코드 생성, 번역, RAG)

메트릭:
  - Throughput, TTFT, TPOT
  - GPU memory utilization
  - Accuracy degradation
  - KV cache 접근 패턴 (프로파일러 데이터)
```

**핵심 질문:**
- 어떤 워크로드에서 어떤 전략이 유리한가?
- KV cache 접근 패턴이 전략 선택에 어떤 시사점을 주는가?
- 메모리 대역폭이 병목인 지점은 어디인가?

**실험 체크리스트:**
- [ ] 모든 전략에 대해 동일 워크로드/모델로 baseline 측정
- [ ] 최소 3개 워크로드 패턴에서 전체 전략 비교
- [ ] 프로파일러 데이터로 "왜 이 결과가 나왔는가" 분석
- [ ] 재현 가능한 실험 스크립트 + 로그 정리

### 7.5 Step 2-4: arXiv 프리프린트 작성 (4-5주)

**논문 구조 (도구 + 비교 실험 논문):**
1. Introduction (1.5p): 오프로딩 전략 비교의 어려움, Kairos Bench 제안
2. Background & Motivation (1.5p): KV cache 오프로딩 기법 분류, 기존 비교의 한계
3. Kairos Bench Design (2-3p): 도구 아키텍처, 전략 플러그인, 프로파일러
4. Experimental Study (2-3p): N개 전략의 통일 조건 비교, 워크로드별 분석
5. Insights & Discussion (1p): 접근 패턴 분석에서 도출된 시사점
6. Related Work (0.5p)
7. Conclusion (0.5p)

**작성 도구:** LaTeX + Overleaf (학회 템플릿 사용)

**완료 기준:**
- [ ] arXiv에 프리프린트 업로드 완료
- [ ] Kairos Bench 코드 GitHub 공개 + README 정비
- [ ] 실험 재현 스크립트 포함

---

## 8. Phase 3: 워크숍 논문 제출 + 공동 연구자 확보 (8-12주)

### 8.1 목적

arXiv 프리프린트와 Kairos Bench를 기반으로 워크숍 논문을 제출하고, 도구 사용자/커뮤니티로부터 공동 연구자를 확보한다.

### 8.2 타겟 워크숍/학회

**워크숍 (첫 논문 타겟, 4-6페이지):**
- MLSys 워크숍 (매년 5월, 제출 마감 2-3월)
- ASPLOS 워크숍 (매년 3월, 제출 마감 12-1월)
- ISCA 워크숍 (매년 6월, 제출 마감 3-4월)
- EuroMLSys (EuroSys 공동 워크숍, 매년 4월)

**정식 학회 (Phase 4 타겟, 10-12페이지):**
- EuroSys (매년 3-4월, 제출 마감 전년 10월)
- USENIX ATC (매년 7월, 제출 마감 1월)
- SoCC (매년 11월, 제출 마감 6월)
- ACM DAC (매년 6-7월)

참고: Top-Tier 학회(MLSys, OSDI, SOSP, ASPLOS 메인)는 채택률 15-20%로 첫 논문으로 비현실적. SK하이닉스 JD에서 Top-Tier 논문은 우대사항이지 필수가 아님. 워크숍 논문 + arXiv + 오픈소스 기여 조합이면 충분히 어필 가능.

### 8.3 공동 연구자 탐색 경로

**경로 0 — Kairos Bench 사용자 (도구 중심 접근의 핵심):**
Kairos Bench를 GitHub에 공개하면, 자신의 오프로딩 전략을 테스트하고 싶은 연구자가 도구를 사용하게 된다. "이 도구로 내 전략 벤치마킹했는데 결과를 공유하고 싶다", "이런 기능이 추가되면 좋겠다"는 피드백이 협업의 시작점이 된다. 빈손으로 컨택하는 것과 완전히 다른 역학이다.

**경로 1 — vLLM/SGLang 커뮤니티:**
Phase 1 기여 과정에서 교류한 메인테이너/기여자. 관련 연구를 하는 사람을 GitHub/Discord에서 식별.

**경로 2 — 국내 대학원 교수 컨택:**
KAIST, SNU, POSTECH 시스템 연구실. arXiv 프리프린트 + Kairos Bench + vLLM 기여 이력을 첨부하여 이메일. 산학 공동 연구 또는 비학위 연구 참여 제안. 컨택 전 반드시 arXiv에 1저자로 먼저 올려서 선행 기록을 확보한다.

**경로 3 — 학회/워크숍 현장:**
워크숍 제출 시 (채택 여부 무관) 학회 참석하여 같은 세션 발표자, 포스터 세션 참가자와 교류.

**경로 4 — SK하이닉스 리서치 협업:**
논문 실적 + 오픈소스 기여 + 도구가 있는 상태에서 접근하면 인턴 또는 공동 연구 가능성.

### 8.4 완료 기준

- [ ] 워크숍에 논문 제출 완료
- [ ] 공동 연구자 또는 멘토 1인 이상 확보
- [ ] 리뷰어 피드백 수용하여 논문 보강

---

## 9. Phase 4: CXL 확장 + 정식 학회 (16-24주)

### 9.1 목적

Phase 2-3의 DRAM/SSD 오프로딩 연구를 이종 메모리(CXL) 관점으로 확장하여 정식 학회 풀 페이퍼를 작성한다.

### 9.2 실험 확장

**Kairos Bench 모듈 3(메모리 계층 에뮬레이터) 구현:**
Phase 2에서 우선순위 3으로 미뤘던 모듈을 이 단계에서 본격 구현한다. 대역폭/지연 파라미터를 설정하면 CXL, 느린 SSD 등을 시뮬레이션하는 백엔드를 vLLM 오프로딩 커넥터에 통합한다.

**3-tier 오프로딩 실험:**
- GPU VRAM → Host DRAM → NVMe SSD 구현 및 최적 배치 전략 설계
- 각 tier 간 전송 대역폭 측정
- Kairos Bench 모듈 1로 전략 비교, 모듈 2로 접근 패턴 분석, 모듈 3로 CXL 에뮬레이션

**CXL 에뮬레이션:**
- 방법 1 — Kairos Bench 모듈 3: 대역폭/지연 파라미터 기반 throttling. "대역폭 X, 지연 Y일 때 throughput은?" sweep 실험 자동화.
- 방법 2 — emucxl: GitHub 오픈소스 (https://github.com/cloudarxiv/emucxl). VM 기반 NUMA 리소스 설정. Ubuntu 22.04, 커널 5.15 이상.
- 방법 3 — CXLMemSim: gem5 대비 15배 빠른 소프트웨어 시뮬레이션.
- 방법 4 — 실물 CXL 하드웨어 (공동 연구를 통해 접근).

참고: 에뮬레이션 기반 논문도 학회에서 인정받음. 다만 실제 CXL 메모리가 에뮬레이션 대비 최대 26% 낮은 지연을 보일 수 있다는 한계를 논문에 명시해야 함. 이 한계가 오히려 "에뮬레이션에서도 이만큼 효과 있으니 실물에서는 더 좋을 것"이라는 스토리로 활용 가능.

**연구 깊이 확장:**
- 워크로드별 adaptive tiering 정책
- Prefetch 정확도 vs 대역폭 소비 트레이드오프
- 다양한 모델 크기/아키텍처에서의 일반화 검증
- "어떤 대역폭/지연 특성의 메모리가 있으면 오프로딩이 실용적인가" → 하드웨어 설계 가이드라인 제시

### 9.3 풀 페이퍼 작성 + 학회 제출

- 워크숍 피드백 + 확장 실험 결과 통합
- 10-12페이지 풀 페이퍼로 EuroSys, ATC, SoCC 타겟
- 동시에 arXiv 업데이트

### 9.4 완료 기준

- [ ] 이종 메모리 (DRAM + SSD + CXL 에뮬레이션) 오프로딩 실험 결과 확보
- [ ] 정식 학회 (EuroSys/ATC/SoCC 중 1곳)에 풀 페이퍼 제출
- [ ] arXiv 프리프린트 업데이트

---

## 10. 주간 시간 배분 가이드

주당 15시간 기준:

| Phase | 논문 읽기 | 코드 작업 | 글쓰기/정리 |
|-------|-----------|-----------|-------------|
| Phase 0 | 6h | 7h | 2h |
| Phase 1 | 3h | 9h | 3h (Pain Point 기록 포함) |
| Phase 2 | 2h | 10h (Kairos Bench 개발 + 실험) | 3h |
| Phase 3 | 2h | 5h | 8h |
| Phase 4 | 2h | 7h | 6h |

---

## 11. SK하이닉스 JD 커버리지

Kairos를 통해 달성 가능한 JD 커버리지:

**Platform Software (거의 풀커버):**
- [x] vLLM, SGLang 등 추론 시스템 최적화 연구 및 개발
- [x] 이종 메모리(HBM, DRAM, CXL 등) 기반 AI 시스템 설계 및 개발
- [x] Cloud/On-premise 환경에서 대규모 AI 서비스 설계 및 구축 (기존 백엔드 경력)
- [x] 최신 AI 플랫폼 소프트웨어의 내부 동작 및 메모리 관리 방식 분석

**AI Model Research (부분 커버):**
- [x] 메모리 접근 패턴 및 병목 특성 분석
- [x] 메모리 용량, 대역폭, 지연 요구사항 도출
- [ ] 새로운 AI 모델 아키텍처 제안 (별도 역량, Kairos 범위 밖)

**System Software (부분 커버):**
- [x] AI 추론 시스템의 KV Cache Offloading 설계 및 개발
- [x] GPU Runtime 메모리 사용 분석
- [ ] 리눅스 커널 메모리 관리 (Kairos 범위 밖)
- [ ] 컴파일러 설계 (Kairos 범위 밖)

**우대사항 (기존 역량 + Kairos):**
- [x] 오픈소스 프로젝트 기여 경험 (기존: Reactor Netty, resilience4j, Spring Cloud Gateway + Kairos: vLLM + Kairos Bench)
- [x] 연구 결과 공개 경험 (Kairos Bench 오픈소스 도구 + arXiv 프리프린트)
- [x] 분산 시스템, 컨테이너, Kubernetes 활용 경험 (CKAD 인증)
- [x] 이종 메모리, 가속기 기반 시스템에서의 모델 최적화 경험 (Kairos Phase 2-4)
- [ ] Top-Tier 논문 게재 (워크숍/A등급 학회로 접근 가능)

---

## 12. 기존 역량 및 프로젝트와의 시너지

### 12.1 백엔드 엔지니어링 경험 전이

- 스케줄러 설계 → vLLM 스케줄러 이해 및 개선
- 메모리 풀 관리 (HikariCP 내부) → KV cache 블록 매니저
- 동시성 제어 → Continuous Batching, preemption 전략
- 성능 프로파일링 → GPU 메모리/연산 병목 분석

### 12.2 LocalSub과의 연결

- Whisper 모델의 로컬 GPU 추론 → 추론 파이프라인 이해
- GPU 메모리 관리 (모델 로딩, 배치 처리) → 프로파일링 감각
- Tauri/Rust + Python 서빙 구조 → 시스템 소프트웨어 설계 경험
- LocalSub에서 추론 서빙 최적화 적용 가능 (Whisper KV cache 관리, 배치 처리 최적화)

### 12.3 오픈소스 기여 방법론 전이

- spring-grpc, resilience4j 기여 경험을 vLLM 기여에 그대로 적용
- Deep Trace Document 방법론으로 코드베이스 분석

### 12.4 Probe와의 사고방식 연결

- Probe(워크플로우 검증 도구) 설계 경험 → Kairos Bench(벤치마킹/프로파일링 도구) 설계에 직접 전이
- 연구자의 워크플로우를 분석하고 반복 작업을 자동화하는 사고방식이 동일
- 도구 API 설계, 플러그인 아키텍처, 재현 가능한 환경 구축은 백엔드 엔지니어의 강점 영역

### 12.5 llm-d 확장 가능성 (Phase 4 이후)

- K8s/CKAD 경험 → llm-d (K8s 네이티브 분산 추론) 기여
- vLLM 기여 경험 + K8s 경험 조합 → 희소한 포지셔닝

---

## 13. 리스크 및 완화 전략

| 리스크 | 완화 전략 |
|--------|-----------|
| Phase 0에서 PyTorch/GPU 학습곡선이 예상보다 가파름 | 기간을 16주까지 유연하게 확장, Karpathy 튜토리얼부터 시작 |
| vLLM 기여가 받아들여지지 않음 | 문서/테스트 기여부터 시작, 커뮤니티 컨벤션 충분히 학습 후 기능 PR |
| 연구 질문을 못 찾음 | 도구 중심 접근으로 완화: Pain Point 기록 → 도구 설계로 자연스럽게 연구 방향 결정 |
| Kairos Bench를 아무도 안 씀 | vLLM 생태계에 자연스럽게 통합되는 구조로 설계, README/문서 품질 확보 |
| RTX 단일 GPU의 실험 스케일 한계 | 소규모 모델(7B)에 집중, 연구의 일반화는 공동 연구자 확보 후 |
| CXL 실물 하드웨어 접근 불가 | Kairos Bench 모듈 3(메모리 계층 에뮬레이터)으로 파라미터 기반 실험, 한계 명시 후 스토리 활용 |
| 학회 제출 기한 놓침 | 여러 학회 병렬 추적, arXiv 먼저 올리고 학회는 다음 사이클 |
| 논문 작성 경험 부재 | 도구 + 비교 실험 논문은 구조가 명확하여 첫 논문으로 적합. 공동 연구자 확보 시 리뷰 가능 |
