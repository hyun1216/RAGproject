from __future__ import annotations

import streamlit as st

from rag_engine import TaxLaborRAG

st.set_page_config(
    page_title="세무·노무 실무 가이드 RAG",
    layout="wide",
)

EXAMPLE_QUESTIONS = [
    ("비과세 요건", "생산직 야간근로수당 비과세 요건과 한도가 어떻게 되나요?"),
    ("경정청구 반려 사유", "건강보험 경정청구 소멸시효와 EDI 반려 사유 알려줘"),
    ("환각 차단 테스트", "비트코인 양도소득세 계산 방법 알려줘"),
]


def get_rag_engine() -> TaxLaborRAG:
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = TaxLaborRAG(data_dir="data")
    return st.session_state.rag_engine


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_uploads" not in st.session_state:
        st.session_state.indexed_uploads = set()
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


init_state()
rag = get_rag_engine()

with st.sidebar:
    st.title("RAG 시스템 설정")
    api_key_input = st.text_input(
        "Gemini API Key (선택 사항)",
        type="password",
        placeholder="미입력 시 로컬 근거 데모 모드 동작",
        help="GEMINI_API_KEY를 입력하면 gemini-2.5-flash로 자연어 답변을 생성합니다.",
    )

    year_filter = st.selectbox(
        "적용 연도 필터",
        options=["전체", "2026", "2025"],
        index=1,
    )

    top_k = st.slider(
        "검색 문서 수 (Top-K)",
        min_value=1,
        max_value=5,
        value=3,
    )

    st.divider()
    st.subheader("실시간 문서 업로드")
    uploaded_files = st.file_uploader(
        "가이드 문서 추가 (.md, .txt, .pdf)",
        type=["md", "txt", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        newly_added = 0
        for uf in uploaded_files:
            file_sig = f"{uf.name}:{uf.size}"
            if file_sig not in st.session_state.indexed_uploads:
                chunks_added = rag.add_document_from_bytes(uf.name, uf.getvalue())
                st.session_state.indexed_uploads.add(file_sig)
                newly_added += chunks_added
        if newly_added > 0:
            st.success(f"신규 {newly_added}개 청크가 인덱스에 반영되었습니다.")

    st.metric("현재 적재된 청크 수", f"{rag.chunk_count}개 (문서 {rag.document_count}건)")

    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("세무·노무 실무 가이드 및 최신 법령 검색 RAG")
st.caption(
    "TF-IDF(0.6) + BM25(0.4) 하이브리드 검색 · 연도별 메타데이터 필터링 · 할루시네이션 차단 가드레일"
)

cols = st.columns(3)
for col, (label, question_text) in zip(cols, EXAMPLE_QUESTIONS):
    if col.button(label, use_container_width=True):
        st.session_state.pending_prompt = question_text

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("results"):
            mode_label = {
                "gemini": "Gemini 2.5 Flash",
                "local_demo": "로컬 근거 데모",
                "guardrail_blocked": "할루시네이션 차단됨",
            }.get(msg.get("mode", ""), msg.get("mode", ""))
            with st.expander(f"참조 근거 및 유사도 점수 ({mode_label} | 연도: {msg.get('year_filter', '전체')})"):
                for idx, res in enumerate(msg["results"], start=1):
                    st.markdown(
                        f"**[{idx}] {res['source']} ({res['year']}년) — {res['article']}**\n"
                        f"- 종합 유사도: `{res['hybrid_score']:.4f}` | "
                        f"벡터(TF-IDF): `{res['vector_score']:.4f}` | "
                        f"BM25: `{res['bm25_score']:.4f}`"
                    )
                    st.text(res["content"])

prompt = st.chat_input("세무·노무 관련 질문을 입력하세요 (예: 식대 비과세 한도는 얼마인가요?)")
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = rag.query(
        question=prompt,
        top_k=top_k,
        year_filter=year_filter,
        api_key=api_key_input or None,
    )

    serialized_results = [
        {
            "source": r.chunk.source,
            "year": r.chunk.year,
            "article": r.chunk.article,
            "content": r.chunk.content,
            "hybrid_score": r.hybrid_score,
            "vector_score": r.vector_score,
            "bm25_score": r.bm25_score,
        }
        for r in response.results
    ]

    with st.chat_message("assistant"):
        st.markdown(response.answer)
        if serialized_results:
            with st.expander(f"참조 근거 및 유사도 점수 (모드: {response.mode} | 연도: {response.year_filter})"):
                for idx, res in enumerate(serialized_results, start=1):
                    st.markdown(
                        f"**[{idx}] {res['source']} ({res['year']}년) — {res['article']}**\n"
                        f"- 종합 유사도: `{res['hybrid_score']:.4f}` | "
                        f"벡터(TF-IDF): `{res['vector_score']:.4f}` | "
                        f"BM25: `{res['bm25_score']:.4f}`"
                    )
                    st.text(res["content"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "mode": response.mode,
            "year_filter": response.year_filter,
            "results": serialized_results,
        }
    )
