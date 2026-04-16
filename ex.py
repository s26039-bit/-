"""
GSM 길잡이 AI 챗봇 — RAG Pipeline (Gemini 버전)
====================================
CSV 데이터 → 청크 생성 → Gemini 임베딩 → FAISS 벡터스토어 → LangChain RAG QA

사용 라이브러리:
  pip install langchain langchain-google-genai langchain-community faiss-cpu pandas python-dotenv
"""
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── 환경 변수 ──────────────────────────────────────────────────────────────────
load_dotenv()  # .env 파일에서 GOOGLE_API_KEY 로드

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

CSV_PATH        = Path(__file__).parent / "data" / "gsm_survey.csv"
VECTORSTORE_DIR = Path(__file__).parent / "vectorstore"

# ── 질문 컬럼 매핑 ─────────────────────────────────────────────────────────────
# (카테고리 레이블, CSV 컬럼 검색 키워드)
QUESTION_MAP = [
    ("선배 종합 조언",           "후배들에게 가장 해주고"),
    ("입학 후 가장 먼저 할 것",  "학교에 들어오고 나서"),
    ("힘든 경험 극복 방법",      "가장 힘들었던"),
    ("프로젝트 경험",            "프로젝트 경험"),
    ("공기업 준비 방법",         "공기업을 가기"),
    ("학교생활 꿀팁",            "학교 생활 꿀팁"),
    ("IT 네트워크 기능반 소개",  "IT 네트워크가"),
    ("공부 방법 조언",           "공부에 대한"),
    ("취업 프로세스 경험",       "취업 프로세스"),
    ("기타 조언",                "기타 조언"),
    ("프로젝트 조언",            "프로젝트에 대한"),
    ("학교생활 조언",            "학교생활에 대한"),
    ("진로 조언",                "진로에 대한"),
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드 & 청크(Document) 생성
# ══════════════════════════════════════════════════════════════════════════════

def find_column(df: pd.DataFrame, keyword: str) -> str | None:
    """키워드가 포함된 컬럼명을 반환합니다."""
    for col in df.columns:
        if keyword in col:
            return col
    return None


def build_documents(csv_path: Path) -> list[Document]:
    """
    CSV → LangChain Document 리스트 변환.

    각 Document는 하나의 RAG 청크를 나타냅니다.
    메타데이터에 기수/상태/전공/카테고리를 포함시켜
    필터링 검색(metadata filtering)이 가능하도록 합니다.
    """
    df = pd.read_csv(csv_path)
    documents: list[Document] = []

    for idx, row in df.iterrows():
        grade  = str(row.get("기수 / 학년", "")).strip()
        status = str(row.get("현재 상태", "")).strip()
        major  = str(row.get("전공", "")).strip()

        if grade == "nan":  grade = "기타"
        if status == "nan": status = "기타"
        if major == "nan":  major = ""

        respondent_label = f"{grade} / {status}" + (f" / {major}" if major else "")

        for category, keyword in QUESTION_MAP:
            col = find_column(df, keyword)
            if col is None:
                continue

            value = row.get(col)
            if pd.isna(value) or not str(value).strip() or len(str(value).strip()) < 5:
                continue

            content = str(value).strip()

            full_text = (
                f"[카테고리] {category}\n"
                f"[출처] {respondent_label}\n"
                f"[내용]\n{content}"
            )

            documents.append(Document(
                page_content=full_text,
                metadata={
                    "category":    category,
                    "grade":       grade,
                    "status":      status,
                    "major":       major,
                    "respondent":  respondent_label,
                    "source":      f"GSM 설문 ({respondent_label})",
                    "raw_content": content,
                },
            ))

    print(f"✅ 총 {len(documents)}개 Document 생성 완료")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    긴 답변을 청크로 분할합니다.
    chunk_size=500, chunk_overlap=50으로 문맥 연속성 보장.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    split_docs = splitter.split_documents(documents)
    print(f"✅ 분할 후 총 {len(split_docs)}개 청크")
    return split_docs


# ══════════════════════════════════════════════════════════════════════════════
# 2. 벡터스토어 생성 / 로드
# ══════════════════════════════════════════════════════════════════════════════

def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Gemini 임베딩 모델을 반환합니다.
    models/embedding-001 : v1beta API와 호환되는 안정 버전
    task_type="retrieval_document" : RAG 검색에 최적화된 임베딩 생성
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_document",
    )


def build_vectorstore(documents: list[Document], save_dir: Path) -> FAISS:
    """
    Gemini 임베딩으로 FAISS 벡터스토어를 생성하고 로컬에 저장합니다.
    """
    embeddings = _get_embeddings()

    print("⏳ 임베딩 생성 중... (Google Gemini API 호출)")
    vectorstore = FAISS.from_documents(documents, embeddings)

    save_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_dir))
    print(f"✅ 벡터스토어 저장 완료: {save_dir}")
    return vectorstore


def load_vectorstore(save_dir: Path) -> FAISS:
    """저장된 FAISS 벡터스토어를 로드합니다."""
    embeddings = _get_embeddings()
    vectorstore = FAISS.load_local(
        str(save_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"✅ 벡터스토어 로드 완료: {save_dir}")
    return vectorstore


def get_vectorstore(force_rebuild: bool = False) -> FAISS:
    """
    벡터스토어가 이미 존재하면 로드, 없으면 새로 생성합니다.
    force_rebuild=True 시 항상 새로 생성합니다.
    """
    index_file = VECTORSTORE_DIR / "index.faiss"

    if not force_rebuild and index_file.exists():
        return load_vectorstore(VECTORSTORE_DIR)

    print("🔨 벡터스토어 새로 생성 중...")
    raw_docs   = build_documents(CSV_PATH)
    split_docs = split_documents(raw_docs)
    return build_vectorstore(split_docs, VECTORSTORE_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# 3. RAG 프롬프트 & 체인 구성
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """당신은 광주소프트웨어마이스터고등학교(GSM) 후배들을 돕는 AI 챗봇 "GSM 길잡이"입니다.
GSM 선배들의 실제 경험과 조언을 바탕으로 신뢰할 수 있고 따뜻한 답변을 제공합니다.

[답변 규칙]
1. 아래 [참고 자료]에 있는 선배들의 실제 경험과 조언만을 근거로 답변합니다.
2. 참고 자료에 없는 내용은 "선배들의 응답에서 해당 내용을 찾지 못했어요"라고 솔직하게 말합니다.
3. 선배들의 말투와 표현을 최대한 살려서 생생하게 전달합니다.
4. 여러 선배의 의견이 있을 경우 다양한 관점을 함께 소개합니다.
5. 출처(기수, 상태, 전공)를 자연스럽게 언급해 신뢰도를 높입니다.
6. 답변은 친근하고 따뜻한 어투로 작성합니다.

[참고 자료]
{context}
"""

HUMAN_TEMPLATE = "{question}"


def format_context(docs: list[Document]) -> str:
    """검색된 Document들을 프롬프트에 삽입할 텍스트로 변환합니다."""
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        parts.append(
            f"--- 참고 {i} ({meta.get('respondent', '알 수 없음')}) ---\n"
            f"카테고리: {meta.get('category', '')}\n"
            f"{meta.get('raw_content', doc.page_content)}"
        )
    return "\n\n".join(parts)


def build_rag_chain(vectorstore: FAISS):
    """
    LangChain LCEL로 RAG 체인을 구성합니다.

    흐름:
      질문 → Retriever(유사도 검색, k=5) → 프롬프트 조합 → Gemini 1.5 Pro → 답변

    검색 시 task_type을 "retrieval_query"로 설정해
    문서 임베딩(retrieval_document)과 쿼리 임베딩을 구분합니다.
    """
    # 쿼리용 임베딩은 task_type을 retrieval_query로 별도 설정
    query_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_query",
    )
    vectorstore.embedding_function = query_embeddings.embed_query

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  HUMAN_TEMPLATE),
    ])

    # Gemini 1.5 Pro: GPT-4o에 대응하는 고성능 모델
    # temperature=0.3: 사실 기반 답변이므로 낮게 설정
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        convert_system_message_to_human=False,  # Gemini는 system role 지원
    )

    rag_chain = (
        {
            "context":  retriever | format_context,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# ══════════════════════════════════════════════════════════════════════════════
# 4. 카테고리 필터링 검색 (선택적 고급 기능)
# ══════════════════════════════════════════════════════════════════════════════

CATEGORY_KEYWORDS = {
    "공부":     ["공부", "학습", "스터디", "자격증", "공부법"],
    "프로젝트": ["프로젝트", "개발", "팀", "협업", "깃허브", "git"],
    "취업":     ["취업", "취직", "면접", "포트폴리오", "이력서", "자소서"],
    "공기업":   ["공기업", "ncs", "NCS", "자격증", "한국사"],
    "인간관계": ["친구", "선배", "인간관계", "룸메", "기숙사"],
    "마음가짐": ["마음가짐", "멘탈", "번아웃", "슬럼프", "포기"],
    "기능반":   ["기능반", "기능경기", "IT 네트워크", "로보틱스"],
    "입학":     ["입학", "처음", "1학년", "신입생", "먼저"],
}


def detect_category(question: str) -> str | None:
    """질문에서 관련 카테고리를 감지합니다."""
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in question:
                return category
    return None


def search_with_filter(
    vectorstore: FAISS,
    question: str,
    filter_metadata: dict | None = None,
    k: int = 5,
) -> list[Document]:
    """
    메타데이터 필터를 적용한 유사도 검색.

    예시:
      filter_metadata={"status": "취업"}  → 취업한 선배 답변만 검색
      filter_metadata={"grade": "졸업생"} → 졸업생 답변만 검색
    """
    if filter_metadata:
        docs = vectorstore.similarity_search(
            question,
            k=k,
            filter=filter_metadata,
        )
    else:
        docs = vectorstore.similarity_search(question, k=k)
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# 5. 메인 인터페이스
# ══════════════════════════════════════════════════════════════════════════════

class GSMChatbot:
    """
    GSM 길잡이 챗봇 클래스.
    벡터스토어와 RAG 체인을 래핑해 간편하게 사용할 수 있습니다.
    """

    def __init__(self, force_rebuild: bool = False):
        print("🚀 GSM 길잡이 챗봇 초기화 중...")
        self.vectorstore = get_vectorstore(force_rebuild=force_rebuild)
        self.chain, self.retriever = build_rag_chain(self.vectorstore)
        print("✅ 챗봇 준비 완료!\n")

    def ask(self, question: str, verbose: bool = False) -> str:
        """
        질문에 대한 답변을 반환합니다.

        Args:
            question: 후배의 질문
            verbose:  True면 검색된 참고 문서도 함께 출력

        Returns:
            Gemini 1.5 Pro가 생성한 답변 문자열
        """
        if verbose:
            docs = self.retriever.invoke(question)
            print("\n📚 검색된 참고 문서:")
            for i, doc in enumerate(docs, 1):
                meta = doc.metadata
                print(f"  [{i}] {meta.get('respondent')} — {meta.get('category')}")
                print(f"      {meta.get('raw_content', '')[:80]}...")
            print()

        answer = self.chain.invoke(question)
        return answer

    def ask_filtered(
        self,
        question: str,
        status: str | None = None,
        grade: str | None = None,
    ) -> str:
        """
        특정 상태(재학/취업)나 기수로 필터링해서 답변합니다.

        Args:
            question: 후배의 질문
            status:   "재학" 또는 "취업" (None이면 전체)
            grade:    "졸업생", "8기" 등 (None이면 전체)
        """
        filter_meta = {}
        if status:
            filter_meta["status"] = status
        if grade:
            filter_meta["grade"] = grade

        docs = search_with_filter(
            self.vectorstore, question,
            filter_metadata=filter_meta if filter_meta else None,
            k=5,
        )

        if not docs:
            return "해당 조건에 맞는 선배 답변을 찾지 못했어요."

        context = format_context(docs)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human",  HUMAN_TEMPLATE),
        ])
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    def stream(self, question: str):
        """스트리밍 방식으로 답변을 출력합니다."""
        for chunk in self.chain.stream(question):
            print(chunk, end="", flush=True)
        print()


# ══════════════════════════════════════════════════════════════════════════════
# 6. 실행 예시
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bot = GSMChatbot(force_rebuild=False)

    questions = [
        "학교에 처음 입학하면 뭐부터 해야 해?",
        "프로젝트 처음 할 때 어떻게 시작하면 돼?",
        "공부가 너무 어려운데 어떻게 극복해?",
        "취업 준비는 언제부터 어떻게 해야 해?",
        "공기업 가려면 어떻게 해야 해?",
        "학교생활 꿀팁 있어?",
    ]

    print("=" * 60)
    print("GSM 길잡이 AI 챗봇 (Gemini 버전)")
    print("=" * 60)

    for q in questions:
        print(f"\n❓ 질문: {q}")
        print("-" * 40)
        answer = bot.ask(q, verbose=False)
        print(f"💬 답변:\n{answer}")
        print("=" * 60)

    print("\n📌 취업한 선배들의 답변만 보기:")
    print(bot.ask_filtered("학교생활에서 가장 중요한 게 뭐야?", status="취업"))

    print("\n📌 스트리밍 답변:")
    bot.stream("슬럼프가 왔을 때 어떻게 해야 해?")