# Neural Sync RAG — Tổng quan kỹ thuật

## Mục tiêu

Neural Sync RAG cho phép tải PDF lên và đặt câu hỏi dựa trên nội dung trích xuất từ tài liệu. Hệ thống đưa các đoạn tìm được cho Gemini để tạo câu trả lời, đồng thời hiển thị đoạn nguồn và số trang để người dùng kiểm chứng.

## Luồng xử lý

### 1. Nạp tài liệu

1. FastAPI nhận PDF và giới hạn kích thước tệp ở 20 MB.
2. PyPDF trích xuất văn bản theo trang. Với trang gần như không có chữ, backend thử OCR cục bộ qua PyMuPDF và Tesseract.
3. Recursive Character Text Splitter chia nội dung thành đoạn tối đa 1.000 ký tự, chồng lấn 200 ký tự.
4. `intfloat/multilingual-e5-small` tạo embedding. Tài liệu được gắn tiền tố `passage:`; câu hỏi được gắn `query:`.
5. FAISS lưu vector theo session trong RAM của backend.

### 2. Truy vấn

1. Backend tìm tối đa 8 đoạn theo câu hỏi gốc; với câu hỏi về trường học, điện thoại, email hoặc địa chỉ, hệ thống tìm thêm bằng một truy vấn có các cách gọi tương đương.
2. Gemini 2.5 Flash nhận câu hỏi cùng các đoạn nguồn và trả lời bằng ngôn ngữ của câu hỏi. Thinking budget mặc định bằng 0 cho tác vụ tra cứu ngắn; có thể cấu hình trong `backend/.env`.
3. API gửi câu trả lời về giao diện theo NDJSON streaming. Nguồn được gửi kèm số trang.

## Công nghệ

| Thành phần | Công nghệ | Vai trò |
| --- | --- | --- |
| Giao diện | Next.js, React, TypeScript | Tải PDF, hiển thị hội thoại và nguồn |
| API | FastAPI, LangChain | Trích xuất, tìm kiếm và điều phối câu trả lời |
| Embedding | Multilingual E5 Small | Biểu diễn nội dung và câu hỏi đa ngôn ngữ |
| Vector search | FAISS | Tìm các đoạn gần với câu hỏi |
| OCR | PyMuPDF, Tesseract | Đọc trang PDF scan hoặc trang thiếu lớp văn bản |
| LLM | Gemini 2.5 Flash | Tạo câu trả lời từ các đoạn được truy xuất |

## Giới hạn cần biết

- Chất lượng phụ thuộc vào văn bản PDF trích xuất được và các đoạn tìm thấy; RAG không đảm bảo câu trả lời luôn đúng. Người dùng nên đối chiếu nguồn hiển thị.
- OCR cần cài Tesseract cùng dữ liệu ngôn ngữ phù hợp. Mặc định là `eng`; CV tiếng Việt nên cấu hình `OCR_LANGUAGES=vie+eng` sau khi cài dữ liệu `vie`.
- Chỉ mục FAISS và session lưu trong RAM của một tiến trình; khởi động lại backend sẽ xóa chúng. Tài liệu cần tải lên lại.
- Lần đầu dùng model embedding cần tải model về máy. Embedding chạy cục bộ; sinh câu trả lời cần kết nối Gemini API và API key hợp lệ.

## Chạy ứng dụng

Xem [README.md](README.md) để cài đặt, cấu hình biến môi trường và khởi chạy frontend/backend.
