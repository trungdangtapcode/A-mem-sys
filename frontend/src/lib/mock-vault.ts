export type VaultGroup = {
  id: string;
  label: string;
  query: string;
  color: string;
  pathPrefix: string;
  tag: string;
  description: string;
  angle?: number;
  radius?: number;
  titles?: string[];
};

export type VaultNode = {
  id: string;
  title: string;
  path: string;
  excerpt: string;
  content?: string;
  summary?: string | null;
  keywords?: string[];
  tags: string[];
  groupId: string;
  color: string;
  weight: number;
  baseX: number;
  baseY: number;
  orbitRadius: number;
  orbitSpeed: number;
  phase: number;
  connections: number;
};

export type VaultLink = {
  source: string;
  target: string;
  strength: number;
};

export type VaultDataset = {
  groups: VaultGroup[];
  nodes: VaultNode[];
  links: VaultLink[];
};

type DraftNode = Omit<VaultNode, "connections">;

const copyPatterns = [
  "Explains how {title} keeps the vault legible when note relationships start to branch quickly.",
  "Captures the reasoning behind {title} so future sessions inherit the context instead of re-deriving it.",
  "Summarizes the operating rules for {title}, with enough detail to reconnect the note back into the wider graph.",
  "Documents the design trade-offs around {title} and how it influences nearby clusters in the workspace.",
  "Tracks the practical cues for {title}, including what to surface in the graph and what to leave in the note body.",
];

const groupBlueprints: VaultGroup[] = [
  {
    id: "vault-core",
    label: "Vault Core",
    query: "tag:#vault-core",
    color: "#8b5cf6",
    pathPrefix: "system/core",
    tag: "#vault-core",
    description: "structural rules for naming, linking, and clustering notes",
    angle: -1.1,
    radius: 88,
    titles: [
      "Context-aware analysis",
      "Backlink integrity",
      "Summary embeddings",
      "Semantic filenames",
      "Manual note ingestion",
      "Timeline recall",
      "Evolution loop",
      "Directory consistency",
      "Long-form capture",
      "Path recovery",
    ],
  },
  {
    id: "retrieval",
    label: "Retrieval",
    query: 'path:"retrieval/semantic/"',
    color: "#facc15",
    pathPrefix: "retrieval/semantic",
    tag: "#retrieval-layer",
    description: "ranking, search tuning, and recall quality",
    angle: -2.6,
    radius: 330,
    titles: [
      "Hybrid vector ranking",
      "BM25 fallback",
      "Similarity blending",
      "Semantic path boost",
      "Contextual rerank",
      "Sparse keyword bridge",
      "Result confidence band",
      "Search intent cache",
      "Metadata weighting",
      "Recall drift check",
    ],
  },
  {
    id: "agent-workflow",
    label: "Agent Workflow",
    query: "tag:#agent-workflow",
    color: "#2563eb",
    pathPrefix: "agents/workflow",
    tag: "#agent-workflow",
    description: "how sessions save state, plan, and hand context forward",
    angle: -0.05,
    radius: 336,
    titles: [
      "Session handoff",
      "Reflection prompts",
      "Tool trace memory",
      "Decision journal",
      "Failure recovery",
      "Prompt inheritance",
      "Context window trim",
      "Task framing",
      "Review cadence",
      "Follow-up hooks",
    ],
  },
  {
    id: "research-cluster",
    label: "Research",
    query: "tag:#research-cluster",
    color: "#d4d4d8",
    pathPrefix: "research/knowledge",
    tag: "#research-cluster",
    description: "source notes, paper synthesis, and idea weaving",
    angle: 1.55,
    radius: 324,
    titles: [
      "Transformer refresher",
      "Zettelkasten linking",
      "Paper digest flow",
      "Evidence ladder",
      "Terminology map",
      "Retrieval benchmarks",
      "Source triangulation",
      "Experiment matrix",
      "Open questions",
      "Reading backlog",
    ],
  },
  {
    id: "ops-runtime",
    label: "Ops Runtime",
    query: 'path:"ops/runtime/"',
    color: "#16a34a",
    pathPrefix: "ops/runtime",
    tag: "#ops-runtime",
    description: "storage backends, persistence, and runtime health",
    angle: 2.75,
    radius: 322,
    titles: [
      "Chroma persistence",
      "zvec storage",
      "Lock file hygiene",
      "Memory bootstrap",
      "Index compaction",
      "Warm start cache",
      "API latency window",
      "Deployment health",
      "Snapshot strategy",
      "Fault isolation",
    ],
  },
  {
    id: "graph-studio",
    label: "Graph Studio",
    query: "file:Graph-studio",
    color: "#ff5d5d",
    pathPrefix: "workspace/graph",
    tag: "#graph-studio",
    description: "interface behavior, graph controls, and visual tuning",
    angle: 0.6,
    radius: 246,
    titles: [
      "Graph tuning",
      "Cluster palette",
      "Node emphasis",
      "Label fade threshold",
      "Edge arrows",
      "Canvas drag states",
      "Zoom handling",
      "Inspector panel",
      "Focus mode",
      "Mobile fallback",
    ],
  },
];

const coreNotes = [
  {
    id: "core-root",
    title: "Vault graph",
    path: "system/core/vault-graph.md",
    excerpt:
      "The graph is the navigational layer for the memory system: dense enough to reveal patterns, quiet enough to let one note become the center of gravity.",
    tags: ["#vault-core", "#graph"],
    groupId: "vault-core",
    color: "#b18cff",
    weight: 3.2,
    baseX: 0,
    baseY: 0,
    orbitRadius: 3,
    orbitSpeed: 0.12,
    phase: 0.1,
  },
  {
    id: "core-memory",
    title: "Memory topology",
    path: "system/core/memory-topology.md",
    excerpt:
      "Defines how paths, backlinks, and semantic tags work together so the vault behaves like a connected system rather than a loose pile of notes.",
    tags: ["#vault-core", "#structure"],
    groupId: "vault-core",
    color: "#8b5cf6",
    weight: 2.7,
    baseX: -64,
    baseY: -26,
    orbitRadius: 5,
    orbitSpeed: 0.18,
    phase: 0.4,
  },
  {
    id: "core-index",
    title: "Link index",
    path: "system/core/link-index.md",
    excerpt:
      "Keeps track of which notes pull the graph together and which ones only matter in a local neighborhood.",
    tags: ["#vault-core", "#index"],
    groupId: "vault-core",
    color: "#9e75ff",
    weight: 2.5,
    baseX: 58,
    baseY: 34,
    orbitRadius: 5,
    orbitSpeed: 0.2,
    phase: 1.1,
  },
  {
    id: "core-summary",
    title: "Summary bridge",
    path: "system/core/summary-bridge.md",
    excerpt:
      "Long notes get distilled into summary vectors so retrieval remains fast without losing the nuance of the source note.",
    tags: ["#vault-core", "#summary"],
    groupId: "vault-core",
    color: "#c5b0ff",
    weight: 2.35,
    baseX: 8,
    baseY: 80,
    orbitRadius: 4,
    orbitSpeed: 0.16,
    phase: 1.7,
  },
];

function slugify(input: string) {
  return input
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/[\s_]+/g, "-")
    .replace(/-+/g, "-")
    .trim();
}

function nodeCopy(title: string, group: VaultGroup, index: number) {
  return copyPatterns[index % copyPatterns.length].replace("{title}", title.toLowerCase());
}

function buildVault() {
  const nodes: DraftNode[] = [...coreNotes];
  const links: VaultLink[] = [
    { source: "core-root", target: "core-memory", strength: 0.95 },
    { source: "core-root", target: "core-index", strength: 0.95 },
    { source: "core-root", target: "core-summary", strength: 0.92 },
    { source: "core-memory", target: "core-index", strength: 0.8 },
    { source: "core-index", target: "core-summary", strength: 0.74 },
  ];

  for (const group of groupBlueprints) {
    const angle = group.angle ?? 0;
    const radius = group.radius ?? 0;
    const titles = group.titles ?? [];
    const centerX = Math.cos(angle) * radius;
    const centerY = Math.sin(angle) * radius * 0.74;
    const hubId = `${group.id}-hub`;

    nodes.push({
      id: hubId,
      title: group.label,
      path: `${group.pathPrefix}/index.md`,
      excerpt: `Acts as the anchor for ${group.description}, giving the cluster a clear center when the graph is zoomed out.`,
      tags: [group.tag, "#index"],
      groupId: group.id,
      color: group.color,
      weight: group.id === "vault-core" ? 2.5 : 2.15,
      baseX: centerX,
      baseY: centerY,
      orbitRadius: 7,
      orbitSpeed: 0.13,
      phase: angle,
    });

    links.push({ source: hubId, target: "core-root", strength: 0.76 });
    links.push({
      source: hubId,
      target: group.id === "vault-core" ? "core-memory" : "core-index",
      strength: 0.68,
    });

    titles.forEach((title, index) => {
      const ring = 90 + (index % 3) * 20 + (group.id === "vault-core" ? 0 : 12);
      const theta =
        angle * 0.48 +
        (index / titles.length) * Math.PI * 2 +
        (index % 2 === 0 ? -0.08 : 0.11);
      const nodeId = `${group.id}-${slugify(title)}`;

      nodes.push({
        id: nodeId,
        title,
        path: `${group.pathPrefix}/${slugify(title)}.md`,
        excerpt: nodeCopy(title, group, index),
        tags: [group.tag, `#${slugify(group.label)}`, index % 2 === 0 ? "#linked-note" : "#working-note"],
        groupId: group.id,
        color: group.color,
        weight: 1.05 + ((index + 1) % 4) * 0.16,
        baseX: centerX + Math.cos(theta) * ring,
        baseY: centerY + Math.sin(theta) * ring * 0.82,
        orbitRadius: 6 + (index % 4) * 2,
        orbitSpeed: 0.18 + (index % 5) * 0.03,
        phase: index * 0.42 + angle,
      });

      links.push({ source: hubId, target: nodeId, strength: 0.74 });
      if (index > 0) {
        const previous = `${group.id}-${slugify(titles[index - 1])}`;
        links.push({ source: previous, target: nodeId, strength: 0.36 });
      }
      if (index % 3 === 0) {
        links.push({
          source: nodeId,
          target: index % 2 === 0 ? "core-memory" : "core-summary",
          strength: 0.43,
        });
      }
    });
  }

  const bridges: Array<[string, string, number]> = [
    ["retrieval-hybrid-vector-ranking", "agent-workflow-session-handoff", 0.48],
    ["retrieval-bm25-fallback", "research-cluster-retrieval-benchmarks", 0.44],
    ["agent-workflow-tool-trace-memory", "ops-runtime-api-latency-window", 0.46],
    ["research-cluster-zettelkasten-linking", "graph-studio-focus-mode", 0.41],
    ["ops-runtime-chroma-persistence", "retrieval-metadata-weighting", 0.52],
    ["graph-studio-label-fade-threshold", "vault-core-summary-embeddings", 0.54],
    ["graph-studio-canvas-drag-states", "agent-workflow-task-framing", 0.39],
    ["research-cluster-open-questions", "agent-workflow-reflection-prompts", 0.43],
    ["ops-runtime-warm-start-cache", "core-summary", 0.42],
    ["graph-studio-mobile-fallback", "ops-runtime-deployment-health", 0.37],
  ];

  for (const [source, target, strength] of bridges) {
    links.push({ source, target, strength });
  }

  const uniqueLinks = new Map<string, VaultLink>();
  for (const link of links) {
    const key = [link.source, link.target].sort().join("::");
    const existing = uniqueLinks.get(key);
    if (!existing || link.strength > existing.strength) {
      uniqueLinks.set(key, link);
    }
  }

  const connectionCounts = new Map<string, number>();
  for (const link of uniqueLinks.values()) {
    connectionCounts.set(link.source, (connectionCounts.get(link.source) ?? 0) + 1);
    connectionCounts.set(link.target, (connectionCounts.get(link.target) ?? 0) + 1);
  }

  const finalizedNodes: VaultNode[] = nodes.map((node) => ({
    ...node,
    connections: connectionCounts.get(node.id) ?? 0,
  }));

  return {
    groups: groupBlueprints,
    nodes: finalizedNodes,
    links: Array.from(uniqueLinks.values()),
  };
}

export const vaultData: VaultDataset = buildVault();

export function matchesNode(node: VaultNode, query: string) {
  if (!query) {
    return true;
  }

  const haystack = [
    node.title,
    node.path,
    node.excerpt,
    node.content ?? "",
    node.summary ?? "",
    (node.keywords ?? []).join(" "),
    node.tags.join(" "),
  ]
    .join(" ")
    .toLowerCase();

  return haystack.includes(query.toLowerCase());
}
