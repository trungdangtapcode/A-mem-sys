import { startTransition, useDeferredValue, useEffect, useState } from "react";
import {
  ArrowUpRight,
  BrainCircuit,
  FileStack,
  FolderTree,
  Network,
  RefreshCw,
  Search,
  Sparkles,
  X,
} from "lucide-react";
import { GraphCanvas, type GraphSettings } from "@/components/graph/graph-canvas";
import { fetchGraphSnapshot } from "@/lib/mcp-api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { matchesNode, type VaultDataset, vaultData } from "@/lib/mock-vault";
import { cn } from "@/lib/utils";
import { buildMcpStatusLabel, layoutSnapshot } from "@/lib/vault-layout";

const defaultSettings: GraphSettings = {
  arrows: false,
  textFadeThreshold: 0.56,
  nodeSize: 0.34,
  linkThickness: 0.28,
  animate: true,
  centerForce: 0.58,
  repelForce: 0.54,
  linkForce: 0.44,
  linkDistance: 0.5,
};

const emptyVault: VaultDataset = {
  groups: [],
  nodes: [],
  links: [],
};

const allowDemoFallback =
  import.meta.env.VITE_ENABLE_DEMO_FALLBACK === "true";

function App() {
  const [graphData, setGraphData] = useState<VaultDataset>(emptyVault);
  const [selectedNodeId, setSelectedNodeId] = useState("core-root");
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [visibleGroupIds, setVisibleGroupIds] = useState(vaultData.groups.map((group) => group.id));
  const [settings, setSettings] = useState(defaultSettings);
  const [connectionMode, setConnectionMode] = useState<"loading" | "mcp" | "demo" | "error">("loading");
  const [connectionMessage, setConnectionMessage] = useState("Connecting to the local stdio MCP server...");
  const [isRefreshing, setIsRefreshing] = useState(false);
  const deferredSearchQuery = useDeferredValue(searchQuery.trim().toLowerCase());
  const activeNodeId = hoveredNodeId ?? selectedNodeId;
  const activeNode =
    graphData.nodes.find((node) => node.id === activeNodeId) ?? graphData.nodes[0] ?? null;
  const visibleGroupSet = new Set(visibleGroupIds);
  const visibleNodes = graphData.nodes.filter((node) => visibleGroupSet.has(node.groupId));
  const matchingNodes = visibleNodes.filter((node) => matchesNode(node, deferredSearchQuery));
  const noteList = (deferredSearchQuery ? matchingNodes : visibleNodes)
    .slice()
    .sort((left, right) => right.connections - left.connections || left.title.localeCompare(right.title))
    .slice(0, 14);
  const pinnedNotes = visibleNodes
    .slice()
    .sort((left, right) => right.weight - left.weight || right.connections - left.connections)
    .slice(0, 6);

  const relatedIds = new Set<string>();
  if (activeNode) {
    for (const link of graphData.links) {
      if (link.source === activeNode.id) {
        relatedIds.add(link.target);
      }
      if (link.target === activeNode.id) {
        relatedIds.add(link.source);
      }
    }
  }

  const relatedNotes = visibleNodes
    .filter((node) => relatedIds.has(node.id))
    .slice()
    .sort((left, right) => right.connections - left.connections)
    .slice(0, 6);

  useEffect(() => {
    void refreshGraph();
  }, []);

  function updateSetting<K extends keyof GraphSettings>(key: K, value: GraphSettings[K]) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  function selectNode(nodeId: string) {
    startTransition(() => {
      setSelectedNodeId(nodeId);
    });
  }

  function toggleGroup(groupId: string) {
    setVisibleGroupIds((current) =>
      current.includes(groupId)
        ? current.filter((id) => id !== groupId)
        : [...current, groupId],
    );
  }

  function restoreGroups() {
    setVisibleGroupIds(graphData.groups.map((group) => group.id));
    setSearchQuery("");
  }

  async function refreshGraph() {
    setIsRefreshing(true);
    setConnectionMode((current) => (current === "demo" ? current : "loading"));
    setConnectionMessage("Connecting to the local stdio MCP server...");

    try {
      const snapshot = await fetchGraphSnapshot({ sync: true });
      const nextGraph = layoutSnapshot(snapshot);

      setGraphData(nextGraph);
      setVisibleGroupIds(nextGraph.groups.map((group) => group.id));
      setSelectedNodeId((current) =>
        nextGraph.nodes.some((node) => node.id === current)
          ? current
          : nextGraph.nodes[0]?.id ?? current,
      );
      setConnectionMode("mcp");
      setConnectionMessage(buildMcpStatusLabel(snapshot));
    } catch (error) {
      if (allowDemoFallback) {
        setGraphData(vaultData);
        setVisibleGroupIds(vaultData.groups.map((group) => group.id));
        setSelectedNodeId((current) =>
          vaultData.nodes.some((node) => node.id === current)
            ? current
            : vaultData.nodes[0]?.id ?? current,
        );
        setConnectionMode("demo");
        setConnectionMessage(
          error instanceof Error ? `${error.message} Using demo vault data.` : "Using demo vault data.",
        );
      } else {
        setGraphData(emptyVault);
        setVisibleGroupIds([]);
        setSelectedNodeId("");
        setConnectionMode("error");
        setConnectionMessage(
          error instanceof Error
            ? `stdio MCP connection failed: ${error.message}`
            : "stdio MCP connection failed",
        );
      }
    } finally {
      setIsRefreshing(false);
    }
  }

  return (
    <div className="min-h-screen px-3 py-3 sm:px-5 sm:py-5">
      <div className="mx-auto flex min-h-[calc(100vh-1.5rem)] max-w-[1680px] flex-col gap-4 rounded-[30px] border border-white/8 bg-black/18 p-3 shadow-[0_36px_120px_-60px_rgba(0,0,0,1)] backdrop-blur-xl sm:p-4">
        <header className="flex flex-col gap-3 rounded-[24px] border border-border/70 bg-white/4 px-4 py-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-start gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-primary/18 text-primary shadow-[inset_0_1px_0_rgba(255,255,255,0.12)]">
              <BrainCircuit className="h-5 w-5" />
            </div>
            <div className="space-y-1">
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="text-lg font-semibold tracking-wide text-foreground sm:text-xl">
                  A-MEM Vault
                </h1>
                <Badge variant="secondary">Vite + shadcn/ui + TSX</Badge>
              </div>
              <p className="max-w-3xl text-sm text-muted-foreground">
                An Obsidian-inspired graph workspace for exploring agentic memory, linked notes,
                and cluster-level relationships.
              </p>
              <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground/80">
                {connectionMessage}
              </p>
            </div>
          </div>

          <div className="flex flex-col gap-3 lg:min-w-[420px] lg:max-w-[520px] lg:flex-1">
            <div className="relative">
              <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                className="pl-11"
                placeholder="Search files, tags, paths, and note summaries..."
                value={searchQuery}
                onChange={(event) => {
                  const value = event.target.value;
                  startTransition(() => setSearchQuery(value));
                }}
              />
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge variant="secondary">
                {visibleNodes.length} visible notes
              </Badge>
              <Badge variant="secondary">
                {graphData.links.length} connections
              </Badge>
              <Badge variant="secondary">
                {deferredSearchQuery ? `${matchingNodes.length} matches` : "Live graph"}
              </Badge>
              <Badge variant="secondary">
                {connectionMode === "mcp"
                  ? "stdio MCP live"
                  : connectionMode === "loading"
                    ? "Connecting..."
                    : connectionMode === "demo"
                      ? "Demo fallback"
                      : "MCP error"}
              </Badge>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => void refreshGraph()}
                disabled={isRefreshing}
              >
                <RefreshCw className={cn("h-3.5 w-3.5", isRefreshing && "animate-spin")} />
                {isRefreshing ? "Refreshing" : "Refresh MCP"}
              </Button>
            </div>
          </div>
        </header>

        <div className="grid flex-1 gap-4 xl:grid-cols-[280px_minmax(0,1fr)_320px]">
          <aside className="order-2 flex flex-col gap-4 xl:order-1">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2">
                  <FileStack className="h-4 w-4 text-primary" />
                  Pinned Notes
                </CardTitle>
                <CardDescription>High-signal nodes that anchor the current vault.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {pinnedNotes.map((note) => (
                  <button
                    key={note.id}
                    className={cn(
                      "flex w-full items-start justify-between rounded-2xl border border-transparent bg-white/3 px-3 py-3 text-left transition hover:border-border/70 hover:bg-white/6",
                      activeNode?.id === note.id && "border-primary/50 bg-primary/10",
                    )}
                    onClick={() => selectNode(note.id)}
                  >
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium text-foreground">{note.title}</p>
                      <p className="mt-1 truncate text-xs text-muted-foreground">{note.path}</p>
                    </div>
                    <span className="ml-3 text-xs text-muted-foreground">{note.connections}</span>
                  </button>
                ))}
                {pinnedNotes.length === 0 ? (
                  <div className="rounded-2xl border border-dashed border-border/80 px-4 py-6 text-sm text-muted-foreground">
                    No live MCP notes loaded yet.
                  </div>
                ) : null}
              </CardContent>
            </Card>

            <Card className="flex-1">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2">
                  <FolderTree className="h-4 w-4 text-primary" />
                  Vault Notes
                </CardTitle>
                <CardDescription>
                  Search narrows the explorer without collapsing the overall graph.
                </CardDescription>
              </CardHeader>
              <CardContent className="pb-0">
                <ScrollArea className="obsidian-scrollbar h-[360px] pr-3 xl:h-[calc(100vh-26rem)]">
                  <div className="space-y-2 pb-5">
                    {noteList.map((note) => (
                      <button
                        key={note.id}
                        className={cn(
                          "w-full rounded-2xl border border-transparent px-3 py-3 text-left transition hover:border-border/70 hover:bg-white/6",
                          selectedNodeId === note.id && "border-primary/50 bg-primary/10",
                        )}
                        onClick={() => selectNode(note.id)}
                      >
                        <div className="flex items-center justify-between gap-3">
                          <span className="truncate text-sm font-medium text-foreground">
                            {note.title}
                          </span>
                          <ArrowUpRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                        </div>
                        <p className="mt-1 truncate text-xs text-muted-foreground">{note.path}</p>
                      </button>
                    ))}
                    {noteList.length === 0 ? (
                      <div className="rounded-2xl border border-dashed border-border/80 px-4 py-6 text-sm text-muted-foreground">
                        No notes match that search yet.
                      </div>
                    ) : null}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </aside>

          <main className="order-1 flex min-w-0 flex-col gap-4 xl:order-2">
            <Card className="flex-1">
              <CardHeader className="pb-4">
                <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2 text-base">
                      <Network className="h-4 w-4 text-primary" />
                      Vault Graph
                    </CardTitle>
                    <CardDescription>
                      Inspired by Obsidian&apos;s graph view: color-coded clusters, live controls,
                      and note-first inspection.
                    </CardDescription>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="secondary">Search-aware graph</Badge>
                    <Badge variant="secondary">Cluster controls</Badge>
                    <Badge variant="secondary">Mobile ready</Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <GraphCanvas
                  className="min-h-[560px] xl:min-h-[calc(100vh-17rem)]"
                  hoveredNodeId={hoveredNodeId}
                  links={graphData.links}
                  nodes={graphData.nodes}
                  searchQuery={deferredSearchQuery}
                  selectedNodeId={selectedNodeId}
                  settings={settings}
                  visibleGroupIds={visibleGroupIds}
                  onHoverNode={setHoveredNodeId}
                  onSelectNode={selectNode}
                />

                {connectionMode === "error" ? (
                  <div className="rounded-2xl border border-red-500/30 bg-red-500/8 px-4 py-4 text-sm text-red-100">
                    The frontend is not connected to the stdio MCP server.
                    Check the terminal running `npm run dev`, then click `Refresh MCP`.
                  </div>
                ) : null}

                <div className="grid gap-4 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,0.95fr)]">
                  <Card className="bg-white/3">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm">Focused Note</CardTitle>
                      <CardDescription>{activeNode?.path ?? "No memory selected"}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <h2 className="text-xl font-semibold tracking-tight text-foreground">
                          {activeNode?.title ?? "No memories yet"}
                        </h2>
                        <p className="mt-2 text-sm leading-6 text-muted-foreground">
                          {activeNode?.excerpt ??
                            "The MCP server is connected, but there are no notes to render yet."}
                        </p>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {(activeNode?.tags ?? []).map((tag) => (
                          <Badge key={tag} variant="secondary">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-white/3">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm">Related Notes</CardTitle>
                      <CardDescription>
                        Direct neighbors of the selected node inside the visible graph.
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      {relatedNotes.map((note) => (
                        <button
                          key={note.id}
                          className="flex w-full items-center justify-between rounded-2xl px-3 py-2.5 text-left transition hover:bg-white/6"
                          onClick={() => selectNode(note.id)}
                        >
                          <div className="flex min-w-0 items-center gap-3">
                            <span
                              className="h-2.5 w-2.5 shrink-0 rounded-full"
                              style={{ backgroundColor: note.color }}
                            />
                            <span className="truncate text-sm text-foreground">{note.title}</span>
                          </div>
                          <span className="text-xs text-muted-foreground">{note.connections}</span>
                        </button>
                      ))}
                      {relatedNotes.length === 0 ? (
                        <div className="rounded-2xl border border-dashed border-border/80 px-4 py-5 text-sm text-muted-foreground">
                          This note currently has no visible neighbors in the active filter set.
                        </div>
                      ) : null}
                    </CardContent>
                  </Card>
                </div>
              </CardContent>
            </Card>
          </main>

          <aside className="order-3 flex flex-col gap-4">
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-primary" />
                      Groups
                    </CardTitle>
                    <CardDescription>Toggle color clusters the way Obsidian handles graph groups.</CardDescription>
                  </div>
                  <Button size="sm" onClick={restoreGroups}>
                    Restore groups
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-2">
                {graphData.groups.map((group) => {
                  const isActive = visibleGroupIds.includes(group.id);
                  return (
                    <div
                      key={group.id}
                      className={cn(
                        "flex items-center gap-3 rounded-2xl border px-3 py-3 transition",
                        isActive
                          ? "border-border/80 bg-white/4"
                          : "border-transparent bg-transparent opacity-55",
                      )}
                    >
                      <span
                        className="h-5 w-5 shrink-0 rounded-full border border-white/12"
                        style={{ backgroundColor: group.color }}
                      />
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm text-foreground">{group.query}</p>
                        <p className="mt-1 truncate text-xs text-muted-foreground">{group.label}</p>
                      </div>
                      <Button
                        size="icon"
                        variant="ghost"
                        onClick={() => toggleGroup(group.id)}
                        aria-label={isActive ? `Hide ${group.label}` : `Show ${group.label}`}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  );
                })}
                {graphData.groups.length === 0 ? (
                  <div className="rounded-2xl border border-dashed border-border/80 px-4 py-6 text-sm text-muted-foreground">
                    No graph groups are available yet.
                  </div>
                ) : null}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Graph Controls</CardTitle>
                <CardDescription>Display and force tuning modeled after the official graph settings.</CardDescription>
              </CardHeader>
              <CardContent>
                <Accordion type="multiple" defaultValue={["display", "forces"]}>
                  <AccordionItem value="display">
                    <AccordionTrigger>Display</AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-5">
                        <div className="flex items-center justify-between gap-3">
                          <div>
                            <p className="text-sm text-foreground">Arrows</p>
                            <p className="text-xs text-muted-foreground">Show link direction</p>
                          </div>
                          <Switch
                            checked={settings.arrows}
                            onCheckedChange={(checked) => updateSetting("arrows", checked)}
                          />
                        </div>

                        <ControlSlider
                          label="Text fade threshold"
                          value={settings.textFadeThreshold}
                          onChange={(value) => updateSetting("textFadeThreshold", value)}
                        />
                        <ControlSlider
                          label="Node size"
                          value={settings.nodeSize}
                          onChange={(value) => updateSetting("nodeSize", value)}
                        />
                        <ControlSlider
                          label="Link thickness"
                          value={settings.linkThickness}
                          onChange={(value) => updateSetting("linkThickness", value)}
                        />

                        <Button
                          className="w-full"
                          variant={settings.animate ? "default" : "secondary"}
                          onClick={() => updateSetting("animate", !settings.animate)}
                        >
                          {settings.animate ? "Pause animation" : "Animate"}
                        </Button>
                      </div>
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="forces">
                    <AccordionTrigger>Forces</AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-5">
                        <ControlSlider
                          label="Center force"
                          value={settings.centerForce}
                          onChange={(value) => updateSetting("centerForce", value)}
                        />
                        <ControlSlider
                          label="Repel force"
                          value={settings.repelForce}
                          onChange={(value) => updateSetting("repelForce", value)}
                        />
                        <ControlSlider
                          label="Link force"
                          value={settings.linkForce}
                          onChange={(value) => updateSetting("linkForce", value)}
                        />
                        <ControlSlider
                          label="Link distance"
                          value={settings.linkDistance}
                          onChange={(value) => updateSetting("linkDistance", value)}
                        />
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
              </CardContent>
            </Card>

            <Card className="flex-1">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Inspector</CardTitle>
                <CardDescription>Quick metadata for the node in focus.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="rounded-2xl border border-border/70 bg-white/3 p-4">
                  <p className="text-xs uppercase tracking-[0.22em] text-muted-foreground">
                    {activeNode?.groupId.replace(/-/g, " ") ?? "vault"}
                  </p>
                  <h3 className="mt-2 text-lg font-semibold text-foreground">
                    {activeNode?.title ?? "No memories yet"}
                  </h3>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {activeNode?.path ?? "The graph is waiting for notes."}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <StatCard label="Connections" value={(activeNode?.connections ?? 0).toString()} />
                  <StatCard
                    label="Cluster"
                    value={
                      activeNode
                        ? visibleNodes.filter((node) => node.groupId === activeNode.groupId).length.toString()
                        : "0"
                    }
                  />
                  <StatCard label="Visible groups" value={visibleGroupIds.length.toString()} />
                  <StatCard label="Search hits" value={matchingNodes.length.toString()} />
                </div>

                <Separator />

                <div className="space-y-2">
                  <p className="text-xs uppercase tracking-[0.22em] text-muted-foreground">
                    Tags
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {(activeNode?.tags ?? []).map((tag) => (
                      <Badge key={tag} variant="secondary">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </aside>
        </div>
      </div>
    </div>
  );
}

type ControlSliderProps = {
  label: string;
  value: number;
  onChange: (value: number) => void;
};

function ControlSlider({ label, value, onChange }: ControlSliderProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm text-foreground">{label}</p>
        <span className="text-xs text-muted-foreground">{Math.round(value * 100)}</span>
      </div>
      <Slider
        max={1}
        min={0}
        step={0.01}
        value={[value]}
        onValueChange={(nextValue) => onChange(nextValue[0] ?? value)}
      />
    </div>
  );
}

type StatCardProps = {
  label: string;
  value: string;
};

function StatCard({ label, value }: StatCardProps) {
  return (
    <div className="rounded-2xl border border-border/70 bg-white/3 p-3">
      <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">{label}</p>
      <p className="mt-2 text-xl font-semibold text-foreground">{value}</p>
    </div>
  );
}

export default App;
