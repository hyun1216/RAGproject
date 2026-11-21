from __future__ import annotations

import argparse
import sys

from rag_engine import TaxLaborRAG

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DEMO_SCENARIOS = [
    ("생산직 야간근로수당 비과세 요건과 한도가 어떻게 되나요?", "2026"),
    ("건강보험 경정청구 소멸시효와 EDI 반려 사유 알려줘", "2026"),
    ("비트코인 양도소득세 계산 방법 알려줘", "2026"),
]


def print_response(response) -> None:
    print("=" * 72)
    print(f"질문       : {response.query}")
    print(f"연도 필터  : {response.year_filter}")
    print(f"동작 모드  : {response.mode}")
    print(f"근거 여부  : {'근거 기반(grounded=True)' if response.grounded else '근거 부족 차단(grounded=False)'}")
    if response.results:
        top = response.results[0]
        print(
            f"최고 유사도: {top.hybrid_score:.4f} "
            f"(벡터 {top.vector_score:.4f} / BM25 {top.bm25_score:.4f})"
        )
    print("-" * 72)
    print(response.answer)
    print("=" * 72)
    print()


def run_demo(rag: TaxLaborRAG) -> None:
    print("\n[세무·노무 실무 가이드 RAG 시스템 - 자동 데모 시연]")
    print(f"적재 문서 수: {rag.document_count}개 | 적재 청크 수: {rag.chunk_count}개\n")
    for idx, (question, year) in enumerate(DEMO_SCENARIOS, start=1):
        print(f">>> 시나리오 {idx}")
        res = rag.query(question, top_k=3, year_filter=year)
        print_response(res)


def run_interactive(rag: TaxLaborRAG) -> None:
    current_year = "전체"
    print("\n[세무·노무 실무 가이드 RAG 시스템 - 대화형 CLI]")
    print(f"적재 문서 수: {rag.document_count}개 | 적재 청크 수: {rag.chunk_count}개")
    print("명령어: /year 2026 | /year 2025 | /year 전체 | exit\n")

    while True:
        try:
            user_input = input(f"[연도:{current_year}] 질문 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("종료합니다.")
            break
        if user_input.startswith("/year"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2 and parts[1] in {"2026", "2025", "전체"}:
                current_year = parts[1]
                print(f"연도 필터가 '{current_year}'(으)로 변경되었습니다.\n")
            else:
                print("사용법: /year 2026 | /year 2025 | /year 전체\n")
            continue

        res = rag.query(user_input, top_k=3, year_filter=current_year)
        print_response(res)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="세무·노무 실무 가이드 및 최신 법령 검색 RAG CLI"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="사전 정의된 3가지 검증 시나리오를 자동 실행합니다.",
    )
    args = parser.parse_args()

    rag = TaxLaborRAG(data_dir="data")
    if args.demo:
        run_demo(rag)
    else:
        run_interactive(rag)


if __name__ == "__main__":
    main()
