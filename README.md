# 세무·노무 실무 가이드 및 최신 법령 검색 RAG 시스템

근로소득 비과세 수당(생산직 야간근로수당, 식대, 자가운전보조금, 출산·보육수당)과 4대보험 보수월액·경정청구 실무 문서를 기반으로, **하이브리드 검색(TF-IDF + BM25)**, **적용연도 메타데이터 필터링**, **할루시네이션 차단 가드레일**을 제공하는 Python RAG 프로토타입입니다.

---

## 핵심 기능

1. **문서 파싱 및 메타데이터 자동 추출**
   - `.md`, `.txt`, `.pdf`(`pypdf`) 문서 지원 및 웹 업로드(`bytes`) 실시간 인덱싱
   - 문서 본문 또는 파일명에서 `적용연도`, `기관`, `카테고리` 자동 추출
2. **조항(`## 제N조`) 단위 스마트 청킹**
   - `##` 헤더 기준으로 법령·실무 조항 문맥을 보존하며, 600자를 초과하는 긴 조항은 100자 overlap으로 분할
3. **외부 형태소 분석기 없는 하이브리드 검색**
   - 한국어(3글자 이상 앞 2글자 접두 토큰 보강)·영문·숫자 통합 토크나이저
   - `TfidfVectorizer` 코사인 유사도(**0.6**) + `BM25Okapi` 정규화 점수(**0.4**) 결합
   - `year_filter` (`2026`, `2025`, `전체`) 지원으로 구버전/신버전 가이드 혼선 방지
4. **할루시네이션 차단 가드레일 (Hallucination Guardrail)**
   - 최고 유사도가 임계값(`0.08`) 미만이면 답변 생성을 차단하고 근거 부족 안내 반환 (`grounded=False`, `mode="guardrail_blocked"`)
5. **이중 생성 모드 (Dual Mode)**
   - `GEMINI_API_KEY` 설정 시 `google-genai` SDK(`gemini-2.5-flash`) 기반 출처 필수 표기 답변 생성
   - API 키 미설정 또는 호출 실패 시 **로컬 근거 데모 모드**로 자동 폴백(외부 키 없이 즉시 시연 가능)

---

## 시스템 아키텍처

```text
+-------------------------------------------------------------------+
|                   User Interface (CLI / Streamlit)                |
+---------------------------------+---------------------------------+
                                  | Query + year_filter + Top-K
                                  v
+-------------------------------------------------------------------+
|                     TaxLaborRAG Core Engine                       |
|                                                                   |
|  [1. Loader & Chunker]                                            |
|   data/*.md, .txt, .pdf --> Metadata(연도/카테고리) + ## 조항 청킹   |
|                                                                   |
|  [2. Hybrid Retriever]                                            |
|   Year Filter --> TF-IDF Cosine(0.6) + BM25Okapi Norm(0.4)        |
|                                                                   |
|  [3. Hallucination Guardrail]                                     |
|   max_score < 0.08 ?                                              |
|     |-- YES --> guardrail_blocked (근거 부족 안내, 추론 차단)        |
|     '-- NO  --> [4. Dual Generator]                               |
|                    |-- GEMINI_API_KEY O --> Gemini 2.5 Flash      |
|                    '-- GEMINI_API_KEY X --> Local Evidence Demo   |
+-------------------------------------------------------------------+
```

---

## 프로젝트 폴더 구조

```text
.
├── data/
│   ├── 2026_근로소득_비과세_수당_실무가이드.md
│   ├── 2026_4대보험_보수월액_및_경정청구_가이드.md
│   └── 2025_근로소득_비과세_수당_실무가이드.md
├── rag_engine.py
├── cli_app.py
├── streamlit_app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 설치 및 실행 방법

### 1. 저장소 클론 및 패키지 설치
```bash
git clone https://github.com/hyun1216/RAGproject.git
cd RAGproject
pip install -r requirements.txt
```

### 2. (선택) Gemini API 키 설정
API 키 없이도 **로컬 근거 데모 모드**로 모든 기능을 시연할 수 있습니다. Gemini 2.5 Flash 답변 생성을 사용하려면 `.env` 파일을 생성하세요.
```bash
cp .env.example .env
# .env 파일 내 GEMINI_API_KEY 입력
```

### 3. CLI 실행
```bash
# 사전 정의된 3가지 검증 시나리오 자동 실행
python cli_app.py --demo

# 대화형 질의응답 모드 (/year 2026, /year 2025, /year 전체, exit)
python cli_app.py
```

### 4. Streamlit 웹 UI 실행
```bash
streamlit run streamlit_app.py
```

---

## 테스트 질문 시나리오

| 번호 | 테스트 질문 | 연도 필터 | 기대 결과 |
| :---: | :--- | :---: | :--- |
| 1 | 생산직 야간근로수당 비과세 요건과 한도가 어떻게 되나요? | `2026` | 2026년 가이드 제1조 검색 (월정액 210만 원 이하, 직전 총급여 3,000만 원 이하, 연 240만 원 한도) |
| 2 | 건강보험 경정청구 소멸시효와 EDI 반려 사유 알려줘 | `2026` | 2026년 4대보험 가이드 제1·2조 검색 (국세 5년 vs 건강·고용·산재 3년, 포괄임금·210만 원 초과·3년 도과) |
| 3 | 비트코인 양도소득세 계산 방법 알려줘 | `2026` | 임계값(0.08) 미만으로 `guardrail_blocked` 반환 (할루시네이션 차단) |
