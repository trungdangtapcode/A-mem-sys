import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { stdioMcpBridgePlugin } from "./mcp/stdio-bridge";

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    stdioMcpBridgePlugin({
      repoRoot: path.resolve(__dirname, ".."),
    }),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
