import path from "node:path";
import type { IncomingMessage, ServerResponse } from "node:http";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import type { Plugin } from "vite";

type BridgeOptions = {
  repoRoot: string;
};

type ToolResultContent =
  | {
      type: "text";
      text: string;
    }
  | {
      type: string;
      [key: string]: unknown;
    };

class StdioMcpBridge {
  private readonly pythonCommand: string;
  private readonly serverScript: string;
  private readonly repoRoot: string;
  private client: Client | null = null;
  private transport: StdioClientTransport | null = null;
  private connecting: Promise<Client> | null = null;

  constructor({ repoRoot }: BridgeOptions) {
    this.repoRoot = repoRoot;
    this.pythonCommand =
      process.env.MCP_PYTHON_COMMAND || path.join(repoRoot, ".venv", "bin", "python");
    this.serverScript =
      process.env.MCP_SERVER_SCRIPT || path.join(repoRoot, "mcp_server.py");
  }

  async callJsonTool<T>(name: string, args: Record<string, unknown> = {}) {
    const client = await this.ensureClient();
    const result = await client.callTool({
      name,
      arguments: args,
    });

    const textEntry = (result.content as ToolResultContent[]).find(
      (entry) => entry.type === "text" && typeof entry.text === "string",
    );

    if (!textEntry || typeof textEntry.text !== "string") {
      throw new Error(`Tool ${name} did not return a text payload`);
    }

    return JSON.parse(textEntry.text) as T;
  }

  status() {
    return {
      transport: "stdio",
      repoRoot: this.repoRoot,
      serverScript: this.serverScript,
    };
  }

  async close() {
    if (this.client) {
      await this.client.close();
    } else if (this.transport) {
      await this.transport.close();
    }
    this.client = null;
    this.transport = null;
    this.connecting = null;
  }

  private async ensureClient() {
    if (this.client) {
      return this.client;
    }
    if (this.connecting) {
      return this.connecting;
    }

    this.connecting = (async () => {
      const transport = new StdioClientTransport({
        command: this.pythonCommand,
        args: [this.serverScript],
        cwd: this.repoRoot,
        env: {
          ...process.env,
          MEMORY_VECTOR_BACKEND:
            process.env.MCP_UI_MEMORY_VECTOR_BACKEND ?? process.env.MEMORY_VECTOR_BACKEND ?? "chroma",
        },
        stderr: "inherit",
      });

      const client = new Client({
        name: "amem-vite-ui",
        version: "0.1.0",
      });

      await client.connect(transport);
      this.transport = transport;
      this.client = client;
      return client;
    })();

    try {
      return await this.connecting;
    } catch (error) {
      this.client = null;
      this.transport = null;
      throw error;
    } finally {
      this.connecting = null;
    }
  }
}

function sendJson(res: ServerResponse, status: number, payload: unknown) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(payload));
}

function sendError(res: ServerResponse, error: unknown) {
  const message = error instanceof Error ? error.message : "Unknown MCP bridge error";
  sendJson(res, 500, { error: message });
}

function createMiddleware(bridge: StdioMcpBridge) {
  return async (req: IncomingMessage, res: ServerResponse, next: () => void) => {
    const url = new URL(req.url ?? "/", "http://127.0.0.1");
    if (!url.pathname.startsWith("/api/mcp")) {
      next();
      return;
    }

    if (req.method !== "GET") {
      sendJson(res, 405, { error: "Method not allowed" });
      return;
    }

    try {
      if (url.pathname === "/api/mcp/status") {
        sendJson(res, 200, bridge.status());
        return;
      }

      if (url.pathname === "/api/mcp/graph") {
        if (url.searchParams.get("sync") === "1") {
          await bridge.callJsonTool("sync_from_disk");
        }
        const payload = await bridge.callJsonTool("graph_snapshot");
        sendJson(res, 200, payload);
        return;
      }

      if (url.pathname.startsWith("/api/mcp/memory/")) {
        const memoryId = decodeURIComponent(url.pathname.replace("/api/mcp/memory/", ""));
        const payload = await bridge.callJsonTool("read_memory", {
          memory_id: memoryId,
        });
        sendJson(res, 200, payload);
        return;
      }

      if (url.pathname === "/api/mcp/search") {
        const query = url.searchParams.get("q") ?? "";
        const k = Number(url.searchParams.get("k") ?? "8");
        const payload = await bridge.callJsonTool("search_memory", {
          query,
          k,
        });
        sendJson(res, 200, payload);
        return;
      }

      next();
    } catch (error) {
      await bridge.close().catch(() => undefined);
      sendError(res, error);
    }
  };
}

export function stdioMcpBridgePlugin(options: BridgeOptions): Plugin {
  const bridge = new StdioMcpBridge(options);

  return {
    name: "stdio-mcp-bridge",
    configureServer(server) {
      server.middlewares.use(createMiddleware(bridge));
      server.httpServer?.once("close", () => {
        void bridge.close();
      });
    },
    configurePreviewServer(server) {
      server.middlewares.use(createMiddleware(bridge));
      server.httpServer?.once("close", () => {
        void bridge.close();
      });
    },
  };
}
