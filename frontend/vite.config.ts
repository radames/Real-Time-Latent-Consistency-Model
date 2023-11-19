import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  server: {
    proxy: {
      '^/settings|/queue_size|/stream': 'http://localhost:7860',
      '/ws': {
        target: 'ws://localhost:7860',
        ws: true
      }
    },
  }
});