export type McpGraphGroup = {
  id: string;
  label: string;
  query: string;
  color: string;
  pathPrefix: string;
  tag: string;
  description: string;
};

export type McpGraphNode = {
  id: string;
  title: string;
  path: string;
  pathLabel?: string | null;
  excerpt: string;
  content: string;
  summary?: string | null;
  keywords?: string[];
  tags: string[];
  groupId: string;
  color: string;
  weight: number;
  connections: number;
  timestamp?: string;
  retrievalCount?: number;
};

export type McpGraphLink = {
  source: string;
  target: string;
  strength: number;
};

export type McpGraphSnapshot = {
  groups: McpGraphGroup[];
  nodes: McpGraphNode[];
  links: McpGraphLink[];
  stats: {
    total_memories: number;
    total_links: number;
    persist_dir: string;
    transport: string;
  };
};
