import json
import logging
import os
import tempfile
import uuid
from io import BytesIO
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).with_name(".env"))

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)
MAX_PDF_BYTES = 20 * 1024 * 1024
GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
)
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "eng")

app = FastAPI(title="RAG Backend API", version="1.0.0")
allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3005").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# In-memory sessions are suitable for this single-process demo. They are cleared on restart.
vector_stores: dict[str, FAISS] = {}


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=4000)


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[Trang {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
        for doc in docs
    )


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    # E5 models expect different prefixes for indexed passages and user queries.
    is_e5 = "multilingual-e5" in EMBEDDING_MODEL.lower()
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True,
            **({"prompt": "passage: "} if is_e5 else {}),
        },
        query_encode_kwargs={
            "normalize_embeddings": True,
            **({"prompt": "query: "} if is_e5 else {}),
        },
    )


def retrieval_queries(question: str) -> list[str]:
    """Add common CV field names so short, conversational questions still retrieve facts."""
    normalized = question.casefold()
    field_aliases = (
        (("điện thoại", "số điện thoại", "sđt", "phone", "mobile"),
         "số điện thoại liên hệ phone number mobile"),
        (("địa chỉ", "address", "ở đâu", "nơi ở"),
         "địa chỉ nơi ở thường trú address location"),
        (("email", "e-mail", "thư điện tử"),
         "email thư điện tử contact"),
        (("trường", "đại học", "học vấn", "học ở", "study", "school", "university", "college", "education"),
         "trường đại học học vấn đào tạo study university college education"),
    )
    queries = [question]
    for triggers, aliases in field_aliases:
        if any(trigger in normalized for trigger in triggers):
            queries.append(f"{question} {aliases}")
    return queries


def retrieve_relevant_docs(vector_store: FAISS, question: str):
    # Search with both the original wording and a CV field expansion; dedupe overlap.
    documents = []
    seen = set()
    for index, query in enumerate(retrieval_queries(question)):
        for doc in vector_store.similarity_search(query, k=8 if index == 0 else 4):
            key = (doc.metadata.get("page"), doc.page_content)
            if key not in seen:
                seen.add(key)
                documents.append(doc)
    return documents


@lru_cache(maxsize=1)
def get_chat_model(api_key: str, thinking_budget: int) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        temperature=0,
        thinking_budget=thinking_budget,
    )


def build_answer_chain():
    prompt = ChatPromptTemplate.from_template(
        "Bạn là trợ lý đọc và tra cứu hồ sơ. Hãy trả lời trực tiếp bằng cùng ngôn ngữ "
        "với câu hỏi của người dùng. "
        "Chỉ dùng dữ kiện có trong context, không suy đoán. Với câu hỏi về trường học, "
        "số điện thoại, email hoặc địa chỉ, hãy dò đúng dòng thông tin tương ứng; giữ "
        "nguyên tên riêng, chữ viết tắt và chữ số như trong tài liệu. Nếu không thấy dữ "
        "kiện trong context, nói rõ là không tìm thấy. Nêu số trang khi có thể.\n\n"
        "Context:\n{context}\n\nCâu hỏi: {input}"
    )
    return prompt | get_chat_model(
        os.environ["GEMINI_API_KEY"], GEMINI_THINKING_BUDGET
    ) | StrOutputParser()


def source_items(docs):
    return [
        {"text": doc.page_content, "page": doc.metadata.get("page", 0) + 1}
        for doc in docs
    ]


def stream_answer(chain, inputs, sources):
    sent_token = False
    try:
        for token in chain.stream(inputs):
            if token:
                sent_token = True
                yield json.dumps(
                    {"type": "token", "content": token}, ensure_ascii=False
                ) + "\n"
        yield json.dumps(
            {"type": "sources", "sources": sources}, ensure_ascii=False
        ) + "\n"
    except Exception as stream_error:
        logger.exception("Streaming chat request failed")
        # Some provider/network failures happen before the first streamed token.
        # Retry once with a normal generation so the chat can still complete.
        if not sent_token:
            try:
                answer = chain.invoke(inputs)
                yield json.dumps(
                    {"type": "token", "content": answer}, ensure_ascii=False
                ) + "\n"
                yield json.dumps(
                    {"type": "sources", "sources": sources}, ensure_ascii=False
                ) + "\n"
                return
            except Exception as retry_error:
                logger.exception("Non-streaming chat retry failed")
                error_message = chat_error_message(retry_error)
        else:
            error_message = chat_error_message(stream_error)
        yield json.dumps(
            {"type": "error", "message": error_message},
            ensure_ascii=False,
        ) + "\n"


def chat_error_message(error: Exception) -> str:
    details = f"{type(error).__name__} {error}".casefold()
    if any(term in details for term in ("429", "quota", "resourceexhausted", "rate limit")):
        return "Gemini đang giới hạn số yêu cầu. Hãy đợi một chút rồi thử lại."
    if any(term in details for term in ("401", "403", "permissiondenied", "api key", "unauthorized")):
        return "Gemini từ chối yêu cầu. Hãy kiểm tra GEMINI_API_KEY và quyền truy cập model."
    if any(term in details for term in ("timeout", "deadline exceeded")):
        return "Gemini phản hồi quá thời gian chờ. Hãy thử lại sau."
    return "Không thể tạo câu trả lời lúc này. Hãy kiểm tra kết nối Gemini rồi thử lại."


def stream_static_answer(answer: str, sources):
    yield json.dumps(
        {"type": "token", "content": answer}, ensure_ascii=False
    ) + "\n"
    yield json.dumps(
        {"type": "sources", "sources": sources}, ensure_ascii=False
    ) + "\n"


def extract_readable_pages(path: str, docs):
    unreadable_pages = []
    try:
        import pymupdf
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.warning("OCR packages are missing; keeping PDF text extraction only")
        return docs, [
            index + 1
            for index, doc in enumerate(docs)
            if sum(char.isalnum() for char in doc.page_content) < 12
        ]

    pdf = pymupdf.open(path)
    readable_docs = []
    for index, doc in enumerate(docs):
        text = doc.page_content.strip()
        if sum(char.isalnum() for char in text) < 24 and index < len(pdf):
            try:
                pixmap = pdf[index].get_pixmap(
                    matrix=pymupdf.Matrix(2, 2), alpha=False
                )
                image = Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB")
                ocr_text = pytesseract.image_to_string(
                    image, lang=OCR_LANGUAGES, timeout=15
                ).strip()
                if sum(char.isalnum() for char in ocr_text) > sum(
                    char.isalnum() for char in text
                ):
                    doc.page_content = ocr_text
                    text = ocr_text
            except Exception:
                logger.exception("OCR failed for PDF page %s", index + 1)

        if sum(char.isalnum() for char in text) < 12:
            unreadable_pages.append(index + 1)
        else:
            readable_docs.append(doc)

    pdf.close()
    return readable_docs, unreadable_pages


def index_pdf(path: str) -> tuple[FAISS, int, list[int]]:
    docs = PyPDFLoader(path).load()
    if not docs:
        raise ValueError(
            "Không trích xuất được chữ từ PDF. Tệp có thể là bản scan/ảnh; hãy OCR "
            "tài liệu trước khi tải lên."
        )

    docs, unreadable_pages = extract_readable_pages(path, docs)
    if not docs:
        raise ValueError(
            "Không đọc được chữ trong PDF, kể cả sau khi OCR. Hãy kiểm tra tệp hoặc "
            "cài Tesseract với gói ngôn ngữ phù hợp."
        )

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    if not chunks:
        raise ValueError("Không thể trích xuất nội dung từ PDF này.")

    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings), len(chunks), unreadable_pages


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    filename = file.filename or ""
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read(MAX_PDF_BYTES + 1)
    await file.close()
    if not contents:
        raise HTTPException(status_code=400, detail="PDF không được để trống.")
    if len(contents) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF vượt quá giới hạn 20 MB.")
    if not contents.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Tệp tải lên không phải PDF hợp lệ.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_path = temp_file.name
            temp_file.write(contents)

        vector_store, chunk_count, unreadable_pages = await run_in_threadpool(
            index_pdf, temp_path
        )
        session_id = str(uuid.uuid4())
        vector_stores[session_id] = vector_store
        return {
            "session_id": session_id,
            "message": "File processed successfully.",
            "chunks": chunk_count,
            "unreadable_pages": unreadable_pages,
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("PDF processing failed")
        raise HTTPException(
            status_code=500, detail="Không thể xử lý PDF. Hãy kiểm tra tệp và thử lại."
        ) from exc
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Could not remove temporary upload: %s", temp_path)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    vector_store = vector_stores.get(request.session_id)
    if vector_store is None:
        raise HTTPException(
            status_code=404, detail="Session not found. Please upload a file first."
        )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503, detail="Backend chưa được cấu hình GEMINI_API_KEY."
        )

    try:
        retrieved_docs = await run_in_threadpool(
            retrieve_relevant_docs, vector_store, request.message
        )
        if not retrieved_docs:
            return {
                "answer": "Mình chưa tìm thấy đoạn phù hợp trong tài liệu. Bạn thử hỏi bằng từ khóa khác nhé.",
                "sources": [],
            }
        chain = build_answer_chain()
        answer = await run_in_threadpool(
            chain.invoke,
            {"context": format_docs(retrieved_docs), "input": request.message},
        )
        sources = source_items(retrieved_docs)
        return {"answer": answer, "sources": sources}
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=502, detail="Không thể tạo câu trả lời lúc này. Hãy thử lại."
        ) from exc


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    vector_store = vector_stores.get(request.session_id)
    if vector_store is None:
        raise HTTPException(
            status_code=404, detail="Session not found. Please upload a file first."
        )
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=503, detail="Backend chưa được cấu hình GEMINI_API_KEY."
        )

    try:
        docs = await run_in_threadpool(
            retrieve_relevant_docs, vector_store, request.message
        )
        if not docs:
            no_match = "Mình chưa tìm thấy đoạn phù hợp trong tài liệu. Bạn thử hỏi bằng từ khóa khác nhé."
            return StreamingResponse(
                stream_static_answer(no_match, []),
                media_type="application/x-ndjson",
            )
        chain = build_answer_chain()
        inputs = {"context": format_docs(docs), "input": request.message}
        return StreamingResponse(
            stream_answer(chain, inputs, source_items(docs)),
            media_type="application/x-ndjson",
        )
    except Exception as exc:
        logger.exception("Could not prepare streaming chat request")
        raise HTTPException(
            status_code=502, detail="Không thể bắt đầu câu trả lời lúc này. Hãy thử lại."
        ) from exc
