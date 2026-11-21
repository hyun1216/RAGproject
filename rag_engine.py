from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}
PLACEHOLDER_API_KEYS = {"", "your_gemini_api_key_here"}


def tokenize_korean(text: str) -> list[str]:
    """별도 형태소 분석기 없이 동작하는 한국어·영문·숫자 토크나이저.

    3글자 이상의 한글 단어는 앞 2글자 접두 토큰을 함께 생성해 복합어 검색 재현율을 높인다.
    """
    raw_tokens = re.findall(r"[가-힣]+|[a-zA-Z0-9]+", text)
    tokens: list[str] = []
    for tok in raw_tokens:
        if "가" <= tok[0] <= "힣":
            tokens.append(tok)
            if len(tok) >= 3:
                tokens.append(tok[:2])
        else:
            tokens.append(tok.lower())
    return tokens


@dataclass
class DocumentChunk:
    chunk_id: str
    source: str
    year: str
    agency: str
    category: str
    article: str
    content: str


@dataclass
class SearchResult:
    chunk: DocumentChunk
    hybrid_score: float
    vector_score: float
    bm25_score: float


@dataclass
class RAGResponse:
    query: str
    answer: str
    grounded: bool
    mode: str  # "gemini" | "local_demo" | "guardrail_blocked"
    year_filter: str
    results: list[SearchResult] = field(default_factory=list)


class TaxLaborRAG:
    """세무·노무 실무 가이드 및 최신 법령 검색 하이브리드 RAG 엔진."""

    def __init__(
        self,
        data_dir: str | Path | None = "data",
        min_score_threshold: float = 0.08,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        api_key: str | None = None,
    ) -> None:
        self.min_score_threshold = min_score_threshold
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.api_key = api_key

        self.chunks: list[DocumentChunk] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._bm25: Optional[BM25Okapi] = None

        if data_dir is not None and Path(data_dir).exists():
            self.load_directory(data_dir)

    @property
    def document_count(self) -> int:
        return len({c.source for c in self.chunks})

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def load_directory(self, data_dir: str | Path) -> int:
        """지정된 디렉터리의 .md, .txt, .pdf 문서를 모두 로드하고 인덱스를 구축한다."""
        directory = Path(data_dir)
        if not directory.exists():
            raise FileNotFoundError(f"데이터 디렉터리를 찾을 수 없습니다: {directory}")

        added = 0
        for file_path in sorted(directory.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                added += self.add_document_from_bytes(
                    filename=file_path.name,
                    file_bytes=file_path.read_bytes(),
                    rebuild_index=False,
                )

        self._build_index()
        return added

    def add_document_from_bytes(
        self,
        filename: str,
        file_bytes: bytes,
        rebuild_index: bool = True,
    ) -> int:
        """업로드된 파일(bytes)을 파싱·청킹하여 인덱스에 즉시 반영한다."""
        text = self._parse_bytes(filename, file_bytes)
        if not text.strip():
            return 0

        metadata = self._extract_metadata(text, filename)
        new_chunks = self._chunk_document(text, filename, metadata)

        # 동일 파일명 재업로드 시 기존 청크 교체
        self.chunks = [c for c in self.chunks if c.source != filename]
        self.chunks.extend(new_chunks)

        if rebuild_index:
            self._build_index()

        return len(new_chunks)

    @staticmethod
    def _parse_bytes(filename: str, file_bytes: bytes) -> str:
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")

        if ext == ".pdf":
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)

        for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
            try:
                return file_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        return file_bytes.decode("utf-8", errors="replace")

    @staticmethod
    def _extract_metadata(text: str, filename: str) -> dict[str, str]:
        year_match = re.search(r"적용연도\s*:\s*(\d{4})", text)
        if not year_match:
            year_match = re.search(r"(20\d{2})", filename)
        year = year_match.group(1) if year_match else "미지정"

        agency_match = re.search(r"기관\s*:\s*(.+)", text)
        agency = agency_match.group(1).strip() if agency_match else "실무 가이드"

        category_match = re.search(r"카테고리\s*:\s*(.+)", text)
        if category_match:
            category = category_match.group(1).strip()
        elif "비과세" in filename or "수당" in filename:
            category = "근로소득 비과세 수당"
        elif "4대보험" in filename or "경정청구" in filename:
            category = "4대보험 및 경정청구"
        else:
            category = "세무·노무 실무"

        return {"year": year, "agency": agency, "category": category}

    def _chunk_document(
        self,
        text: str,
        filename: str,
        metadata: dict[str, str],
    ) -> list[DocumentChunk]:
        """'##' 헤더 기준으로 조항을 우선 보존하고, 600자 초과 시 100자 overlap으로 분할한다."""
        normalized = text.replace("\r\n", "\n").strip()
        sections = re.split(r"(?=^##\s+)", normalized, flags=re.MULTILINE)

        chunks: list[DocumentChunk] = []
        chunk_idx = 0

        has_h2 = any(sec.lstrip().startswith("## ") for sec in sections)
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue

            if sec.startswith("## "):
                first_line, *rest = sec.split("\n", 1)
                article = first_line.lstrip("#").strip()
                body = sec
            else:
                # 문서 상단 메타데이터/제목만 있는 서문은 ## 섹션이 존재할 경우 제외
                if has_h2:
                    continue
                article = "본문"
                body = sec

            for part in self._split_with_overlap(body):
                chunk_idx += 1
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{filename}#{chunk_idx}",
                        source=filename,
                        year=metadata["year"],
                        agency=metadata["agency"],
                        category=metadata["category"],
                        article=article,
                        content=part,
                    )
                )

        return chunks

    def _split_with_overlap(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        parts: list[str] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            parts.append(text[start:end].strip())
            if end >= len(text):
                break
            start += step
        return parts

    def _build_index(self) -> None:
        if not self.chunks:
            self._vectorizer = None
            self._tfidf_matrix = None
            self._bm25 = None
            return

        corpus = [
            f"{c.year} {c.category} {c.article}\n{c.content}" for c in self.chunks
        ]
        self._vectorizer = TfidfVectorizer(
            tokenizer=tokenize_korean,
            token_pattern=None,
            lowercase=False,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)

        tokenized_corpus = [tokenize_korean(doc) for doc in corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 3,
        year_filter: str | None = None,
    ) -> list[SearchResult]:
        """TF-IDF(0.6) + BM25Okapi(0.4) 하이브리드 검색 및 연도 메타데이터 필터링."""
        if not self.chunks or self._vectorizer is None or self._bm25 is None:
            return []

        if year_filter and year_filter != "전체":
            candidate_indices = [
                i for i, c in enumerate(self.chunks) if c.year == str(year_filter)
            ]
        else:
            candidate_indices = list(range(len(self.chunks)))

        if not candidate_indices:
            return []

        query_vec = self._vectorizer.transform([query])
        vector_scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        query_tokens = tokenize_korean(query)
        unique_query_tokens = set(query_tokens)
        raw_bm25 = np.array(self._bm25.get_scores(query_tokens), dtype=float)
        clipped_bm25 = np.clip(raw_bm25, 0.0, None)
        max_bm25 = float(clipped_bm25[candidate_indices].max())
        norm_bm25 = clipped_bm25 / max_bm25 if max_bm25 > 0 else clipped_bm25

        results: list[SearchResult] = []
        for idx in candidate_indices:
            v_score = float(vector_scores[idx])
            doc_tokens = set(
                tokenize_korean(
                    f"{self.chunks[idx].year} {self.chunks[idx].category} "
                    f"{self.chunks[idx].article}\n{self.chunks[idx].content}"
                )
            )
            coverage = (
                len(unique_query_tokens & doc_tokens) / len(unique_query_tokens)
                if unique_query_tokens
                else 0.0
            )
            b_score = float(norm_bm25[idx]) * (coverage ** 2)
            # 키워드/벡터가 전혀 매칭되지 않은 경우 노이즈 점수를 주지 않음
            if v_score == 0.0 and raw_bm25[idx] <= 0.0:
                h_score = 0.0
            else:
                h_score = self.vector_weight * v_score + self.bm25_weight * b_score

            results.append(
                SearchResult(
                    chunk=self.chunks[idx],
                    hybrid_score=round(h_score, 4),
                    vector_score=round(v_score, 4),
                    bm25_score=round(b_score, 4),
                )
            )

        results.sort(key=lambda r: r.hybrid_score, reverse=True)
        return results[:top_k]

    def query(
        self,
        question: str,
        top_k: int = 3,
        year_filter: str | None = "전체",
        api_key: str | None = None,
    ) -> RAGResponse:
        """하이브리드 검색, 할루시네이션 가드레일 검사, 이중 모드(Gemini/로컬 데모) 답변 생성."""
        effective_year = year_filter or "전체"
        results = self.search(question, top_k=top_k, year_filter=effective_year)

        if not results or results[0].hybrid_score < self.min_score_threshold:
            top_score = results[0].hybrid_score if results else 0.0
            blocked_msg = (
                f"**근거 부족으로 답변을 생성하지 않았습니다 (할루시네이션 차단 가드레일 작동).**\n\n"
                f"- 적용 연도 필터: `{effective_year}`\n"
                f"- 최고 검색 유사도: `{top_score:.4f}` (임계값 `{self.min_score_threshold:.2f}` 미만)\n"
                f"- 안내: 적재된 세무·노무 실무 가이드 문서에서 질문과 관련된 조항을 찾을 수 없습니다. "
                f"문서 외 영역에 대한 추론 답변은 제공하지 않습니다."
            )
            return RAGResponse(
                query=question,
                answer=blocked_msg,
                grounded=False,
                mode="guardrail_blocked",
                year_filter=effective_year,
                results=results,
            )

        resolved_key = (
            api_key or self.api_key or os.getenv("GEMINI_API_KEY", "")
        ).strip()

        if resolved_key and resolved_key not in PLACEHOLDER_API_KEYS:
            gemini_answer = self._generate_with_gemini(
                question=question,
                results=results,
                api_key=resolved_key,
            )
            if gemini_answer:
                return RAGResponse(
                    query=question,
                    answer=gemini_answer,
                    grounded=True,
                    mode="gemini",
                    year_filter=effective_year,
                    results=results,
                )

        demo_answer = self._generate_local_demo_answer(question, results)
        return RAGResponse(
            query=question,
            answer=demo_answer,
            grounded=True,
            mode="local_demo",
            year_filter=effective_year,
            results=results,
        )

    @staticmethod
    def _generate_with_gemini(
        question: str,
        results: list[SearchResult],
        api_key: str,
    ) -> str | None:
        try:
            from google import genai

            context_blocks = []
            for idx, res in enumerate(results, start=1):
                c = res.chunk
                context_blocks.append(
                    f"[근거 {idx}] 출처: {c.source} | 적용연도: {c.year} | 조항: {c.article} (유사도: {res.hybrid_score:.4f})\n"
                    f"{c.content}"
                )
            context_text = "\n\n---\n\n".join(context_blocks)

            prompt = (
                "당신은 대한민국 세무·노무 실무 전문 AI 어시스턴트입니다.\n"
                "반드시 아래 [참고 문서]에 포함된 내용만을 근거로 답변하세요.\n"
                "규칙:\n"
                "1. 제공된 문서 외의 추론, 일반 상식 보충, 법령 임의 해석을 절대 금지합니다.\n"
                "2. 문서에 없는 내용이면 '제공된 문서에서 확인할 수 없습니다'라고 답하세요.\n"
                "3. 답변 본문 또는 마지막에 반드시 인용한 출처(파일명, 적용연도, 제N조)를 명시하세요.\n\n"
                f"[참고 문서]\n{context_text}\n\n"
                f"[질문]\n{question}\n\n"
                "[답변]"
            )

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            text = getattr(response, "text", None)
            return text.strip() if text else None
        except Exception:
            return None

    @staticmethod
    def _generate_local_demo_answer(
        question: str,
        results: list[SearchResult],
    ) -> str:
        lines = [
            f"**[로컬 근거 데모 모드]** 질문하신 내용과 가장 일치하는 실무 가이드 조항을 검색했습니다.",
            "",
        ]
        for idx, res in enumerate(results, start=1):
            c = res.chunk
            lines.append(
                f"### {idx}. [{c.year}년] {c.article} "
                f"(종합 {res.hybrid_score:.4f} | 벡터 {res.vector_score:.4f} | BM25 {res.bm25_score:.4f})"
            )
            lines.append(f"- **출처**: `{c.source}` ({c.agency})")
            lines.append("")
            lines.append(c.content)
            lines.append("")

        sources_summary = ", ".join(
            f"`{r.chunk.source} ({r.chunk.article})`" for r in results
        )
        lines.append(f"---\n**참조 출처 요약**: {sources_summary}")
        return "\n".join(lines)
