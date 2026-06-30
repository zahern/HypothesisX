import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// During `npm run dev`, proxy backend traffic to the FastAPI server on :8000
// so the frontend and backend can be developed independently with hot reload.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/ws':  { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
})
