'use client';

import { useState, useRef, useEffect } from 'react';

type Message = {
  role: 'user' | 'assistant';
  content: string;
  sources?: { text: string; page: number }[];
};

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000').replace(/\/$/, '');

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Đã xảy ra lỗi không xác định.';
}

function isSource(value: unknown): value is { text: string; page: number } {
  return typeof value === 'object' && value !== null &&
    'text' in value && typeof value.text === 'string' &&
    'page' in value && typeof value.page === 'number';
}

export default function Home() {
  // apiKey is now handled securely in the backend
  const [file, setFile] = useState<File | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');

  const [isUploading, setIsUploading] = useState(false);
  const [isChatting, setIsChatting] = useState(false);
  const [status, setStatus] = useState({ type: '', message: '' });

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setSessionId(null);
    setMessages([]);

    setIsUploading(true);
    setStatus({ type: 'info', message: 'Đang tải lên và xử lý...' });

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || 'Không thể tải tài liệu lên.');
      }

      const data = await response.json();
      setSessionId(data.session_id);
      const unreadablePages = Array.isArray(data.unreadable_pages)
        ? data.unreadable_pages.filter((page: unknown): page is number => typeof page === 'number')
        : [];
      setStatus(unreadablePages.length
        ? {
            type: 'warning',
            message: `Đã xử lý ${data.chunks} đoạn, nhưng chưa đọc được trang ${unreadablePages.join(', ')}. Hãy kiểm tra OCR và gói ngôn ngữ Tesseract.`,
          }
        : { type: 'success', message: `Hoàn tất! Đã xử lý ${data.chunks} đoạn văn bản.` });
    } catch (err: unknown) {
      setStatus({ type: 'error', message: getErrorMessage(err) });
      setFile(null);
    } finally {
      setIsUploading(false);
    }
  };

  const handleSend = async (e?: React.FormEvent) => {
    e?.preventDefault();

    // API key check removed, now handled by backend

    if (!input.trim() || !sessionId) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [
      ...prev,
      { role: 'user', content: userMessage },
      { role: 'assistant', content: '' },
    ]);
    setIsChatting(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          message: userMessage
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          typeof errorData.detail === 'string'
            ? errorData.detail
            : 'Lỗi kết nối tới máy chủ',
        );
      }

      if (!response.body) throw new Error('Trình duyệt không hỗ trợ nhận câu trả lời dạng luồng.');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = '';
      const processLine = (line: string) => {
        if (!line.trim()) return;
        const event = JSON.parse(line) as {
          type: 'token' | 'sources' | 'error';
          content?: string;
          sources?: unknown[];
          message?: string;
        };
        if (event.type === 'token' && event.content) {
          setMessages(prev => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === 'assistant') {
              next[next.length - 1] = { ...last, content: last.content + event.content };
            }
            return next;
          });
        } else if (event.type === 'sources') {
          const sources = Array.isArray(event.sources) ? event.sources.filter(isSource) : [];
          setMessages(prev => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === 'assistant') next[next.length - 1] = { ...last, sources };
            return next;
          });
        } else if (event.type === 'error') {
          throw new Error(event.message || 'Không thể tạo câu trả lời lúc này.');
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        pending += decoder.decode(value, { stream: !done });
        const lines = pending.split('\n');
        pending = lines.pop() ?? '';
        lines.forEach(processLine);
        if (done) break;
      }
      if (pending.trim()) processLine(pending);
    } catch (err: unknown) {
      setMessages(prev => {
        const next = [...prev];
        const last = next[next.length - 1];
        const error = `Lỗi: ${getErrorMessage(err)}`;
        if (last?.role === 'assistant') {
          next[next.length - 1] = {
            ...last,
            content: last.content ? `${last.content}\n\n${error}` : error,
          };
        }
        return next;
      });
    } finally {
      setIsChatting(false);
    }
  };

  return (
    <div className="layout-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon" aria-hidden="true">
            <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="brand-gradient" x1="5" y1="5" x2="43" y2="44" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#6D7CFF" />
                  <stop offset="1" stopColor="#8B5CF6" />
                </linearGradient>
              </defs>
              <rect x="2" y="2" width="44" height="44" rx="14" fill="url(#brand-gradient)" />
              <path d="M15 11.5h11l7 7v18H15v-25Z" stroke="white" strokeWidth="1.8" strokeLinejoin="round" />
              <path d="M26 12v7h7M19 25h10M19 29h6" stroke="white" strokeWidth="1.8" strokeLinecap="round" />
              <circle cx="30" cy="30" r="5" fill="#DDF7FF" stroke="white" strokeWidth="1.5" />
              <path d="m33.7 33.7 2.8 2.8" stroke="white" strokeWidth="2" strokeLinecap="round" />
              <circle cx="30" cy="30" r="1.4" fill="#6366F1" />
            </svg>
          </div>
          <h1>Neural Sync RAG</h1>
        </div>



        <div className="input-group">
          <label>Tài liệu (PDF)</label>
          <div className="file-upload-wrapper">
            <div className="upload-box" style={{ borderColor: file ? 'var(--success)' : '' }}>
              {isUploading ? (
                <>
                  <div className="spinner"></div>
                  <span style={{ fontSize: '0.9rem' }}>Đang xử lý...</span>
                </>
              ) : file ? (
                <>
                  <div className="upload-icon">📄</div>
                  <span style={{ fontSize: '0.9rem', wordBreak: 'break-all' }}>{file.name}</span>
                </>
              ) : (
                <>
                  <div className="upload-icon">📁</div>
                  <span style={{ fontSize: '0.9rem' }}>Kéo thả hoặc nhấp để chọn PDF</span>
                </>
              )}
            </div>
            <input
              type="file"
              accept=".pdf"
              onChange={handleUpload}
              disabled={isUploading}
            />
          </div>
        </div>

        {status.message && (
          <div className={`status-alert ${status.type}`}>
            {status.type === 'error' ? '⚠️' : status.type === 'success' ? '✅' : 'ℹ️'}
            <div>{status.message}</div>
          </div>
        )}
      </aside>

      {/* Main Chat Area */}
      <main className="chat-area">
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>Hỏi bất cứ điều gì</h2>
              <p>Tải lên tệp PDF để bắt đầu trò chuyện với tài liệu.</p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`message-wrapper ${msg.role}`}>
                <div className={`avatar ${msg.role}`}>
                  {msg.role === 'user' ? 'U' : 'AI'}
                </div>
                <div className="message-content">
                  {msg.content ? (
                    <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                  ) : isChatting && msg.role === 'assistant' && idx === messages.length - 1 ? (
                    <div className="assistant-loading" role="status" aria-label="AI đang trả lời">
                      <div className="spinner" />
                      <span>Đang trả lời…</span>
                    </div>
                  ) : null}

                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources-container">
                      {msg.sources.map((src, i) => (
                        <div key={i} className="source-item">
                          <strong>Trang {src.page}: </strong>
                          {src.text.substring(0, 180)}{src.text.length > 180 ? '…' : ''}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          <div ref={messagesEndRef} />
        </div>

        <div className="input-area">
          <form className="chat-input-wrapper" onSubmit={handleSend}>
            <input
              type="text"
              className="chat-input"
              placeholder={sessionId ? "Nhập câu hỏi của bạn..." : "Vui lòng tải lên tài liệu trước"}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={!sessionId || isChatting}
            />
            <button
              type="submit"
              className="send-button"
              disabled={!input.trim() || !sessionId || isChatting}
            >
              ↑
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}
