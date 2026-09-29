# Neural Sync RAG

Ứng dụng hỏi đáp trên tài liệu PDF, gồm giao diện Next.js và API FastAPI. Backend tạo embedding đa ngôn ngữ bằng `intfloat/multilingual-e5-small`, lưu chỉ mục FAISS trong bộ nhớ, rồi dùng Gemini để tạo câu trả lời có trích nguồn theo trang. Câu trả lời được gửi về giao diện theo luồng để nội dung bắt đầu xuất hiện trước khi Gemini hoàn tất.

## Yêu cầu

- Python 3.10+
- Node.js 20+
- Tesseract OCR (cần gói ngôn ngữ tương ứng nếu PDF là bản scan)
- Gemini API key

## Chạy backend

```bash
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
```

Điền `GEMINI_API_KEY` trong `backend/.env`, sau đó chạy:

```bash
uvicorn main:app --app-dir backend --reload --port 8000
```

API kiểm tra trạng thái tại `http://localhost:8000/api/health`.

## Chạy frontend

```bash
cd frontend
npm ci
cp .env.example .env.local
npm run dev
```

Mở `http://localhost:3005`. Nếu backend chạy ở địa chỉ khác, cập nhật `NEXT_PUBLIC_API_URL` trong `frontend/.env.local` và thêm origin frontend vào `CORS_ORIGINS` trong `backend/.env`.

## Kiểm tra

```bash
cd frontend
npm run lint
npm run build
```

PDF được giới hạn ở 20 MB. Lần lập chỉ mục đầu tiên sẽ tải model embedding về máy. Backend thử OCR các trang PDF không trích được chữ bằng Tesseract; `OCR_LANGUAGES=eng` là mặc định. Để OCR tiếng Việt, cài thêm Tesseract Vietnamese trained data, kiểm tra `tesseract --list-langs`, rồi đổi cấu hình thành `OCR_LANGUAGES=vie+eng`. Backend mặc định tắt thinking budget của Gemini cho dạng tra cứu ngắn để giảm thời gian chờ; có thể tăng `GEMINI_THINKING_BUDGET` trong `backend/.env` nếu tài liệu cần suy luận nhiều bước. Session và chỉ mục được lưu trong RAM của tiến trình backend, nên sẽ mất khi backend khởi động lại; sau khi đổi model embedding, hãy khởi động lại backend và tải lại tài liệu để tạo chỉ mục mới.
