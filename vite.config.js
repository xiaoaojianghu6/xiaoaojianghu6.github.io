import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { resolve } from 'path'
import { existsSync } from 'fs'

const PUBLIC = resolve(__dirname, 'public');

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    // Serve directory index.html files for taikisato static copy
    {
      name: 'serve-dir-index',
      configureServer(server) {
        server.middlewares.use((req, _res, next) => {
          const url = req.url?.split('?')[0] || '/';
          // Serve builder.html (Vite-processed React) for /builder/ path
          if (url === '/builder/' || url === '/builder') {
            req.url = '/builder.html';
            return next();
          }
          // Try serving <path>/index.html from public/ for any path ending with /
          const indexPath = url.endsWith('/')
            ? `${url}index.html`
            : `${url}/index.html`;
          if (existsSync(resolve(PUBLIC, `.${indexPath}`))) {
            req.url = indexPath;
          }
          next();
        });
      },
    },
  ],
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        builder: resolve(__dirname, 'builder.html'),
      },
    },
  },
});
