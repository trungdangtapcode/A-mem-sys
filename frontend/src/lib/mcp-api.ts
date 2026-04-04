import type { McpGraphSnapshot } from "@/lib/mcp-types";

async function readJson<T>(input: RequestInfo | URL) {
  const response = await fetch(input);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with ${response.status}`);
  }
  return (await response.json()) as T;
}

export function fetchGraphSnapshot({ sync = false }: { sync?: boolean } = {}) {
  const url = sync ? "/api/mcp/graph?sync=1" : "/api/mcp/graph";
  return readJson<McpGraphSnapshot>(url);
}
