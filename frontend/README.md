# Frontend

Next.js chat UI for the Neural Sync RAG API. The app runs on port `3005` by default.

```bash
npm ci
cp .env.example .env.local
npm run dev
```

Set `NEXT_PUBLIC_API_URL` in `.env.local` if the FastAPI backend is not at `http://localhost:8000`.

From this directory, run `npm run lint` and `npm run build` to check the frontend.
