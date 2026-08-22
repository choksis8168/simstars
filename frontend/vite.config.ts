import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Dev-only convenience: `npm run dev` proxies API calls to the FastAPI
    // backend (assumed running on :8000 via `uv run simstars serve`), so
    // there's no CORS setup needed and the frontend code can just call
    // relative /api/... paths in both dev and the built, backend-served app.
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
