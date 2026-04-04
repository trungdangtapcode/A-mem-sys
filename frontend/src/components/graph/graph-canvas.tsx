import {
  useEffect,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { cn } from "@/lib/utils";
import { matchesNode, type VaultLink, type VaultNode } from "@/lib/mock-vault";

export type GraphSettings = {
  arrows: boolean;
  textFadeThreshold: number;
  nodeSize: number;
  linkThickness: number;
  animate: boolean;
  centerForce: number;
  repelForce: number;
  linkForce: number;
  linkDistance: number;
};

type GraphCanvasProps = {
  className?: string;
  links: VaultLink[];
  nodes: VaultNode[];
  hoveredNodeId: string | null;
  searchQuery: string;
  selectedNodeId: string;
  settings: GraphSettings;
  visibleGroupIds: string[];
  onHoverNode: (id: string | null) => void;
  onSelectNode: (id: string) => void;
};

type Position = {
  x: number;
  y: number;
};

const VIEWBOX = "-720 -460 1440 920";
const MIN_ZOOM = 0.52;
const MAX_ZOOM = 1.95;
const DRAG_THRESHOLD = 4;

type DragState = {
  pointerId: number | null;
  originClientX: number;
  originClientY: number;
  /** The graph-space point that was under the cursor at drag start */
  anchorX: number;
  anchorY: number;
  moved: boolean;
};

export function GraphCanvas({
  className,
  links,
  nodes,
  hoveredNodeId,
  searchQuery,
  selectedNodeId,
  settings,
  visibleGroupIds,
  onHoverNode,
  onSelectNode,
}: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(0.92);
  const [tick, setTick] = useState(0);
  const [isPanning, setIsPanning] = useState(false);
  const panRef = useRef(pan);
  const zoomRef = useRef(zoom);
  const dragStateRef = useRef<DragState>({
    pointerId: null,
    originClientX: 0,
    originClientY: 0,
    anchorX: 0,
    anchorY: 0,
    moved: false,
  });
  const suppressClickRef = useRef(false);
  const visibleGroupSet = new Set(visibleGroupIds);
  const activeNodeId = hoveredNodeId ?? selectedNodeId;
  const query = searchQuery.trim().toLowerCase();
  const visibleNodes = nodes.filter((node) => visibleGroupSet.has(node.groupId));
  const visibleNodeIds = new Set(visibleNodes.map((node) => node.id));
  const visibleLinks = links.filter(
    (link) => visibleNodeIds.has(link.source) && visibleNodeIds.has(link.target),
  );
  const searchMatches = new Set(
    visibleNodes.filter((node) => matchesNode(node, query)).map((node) => node.id),
  );

  const adjacency = new Map<string, Set<string>>();
  for (const link of visibleLinks) {
    adjacency.set(link.source, new Set([...(adjacency.get(link.source) ?? []), link.target]));
    adjacency.set(link.target, new Set([...(adjacency.get(link.target) ?? []), link.source]));
  }

  const neighborhood = new Set<string>();
  if (activeNodeId && visibleNodeIds.has(activeNodeId)) {
    neighborhood.add(activeNodeId);
    for (const id of adjacency.get(activeNodeId) ?? []) {
      neighborhood.add(id);
    }
  }

  const positions: Record<string, Position> = {};
  const compactness = 1.16 - settings.centerForce * 0.42;
  const distanceFactor = 0.88 + settings.linkDistance * 0.68;
  const wobble = settings.animate ? 0.22 + settings.repelForce * 0.56 : 0;

  for (const node of visibleNodes) {
    const x =
      node.baseX * compactness * distanceFactor +
      Math.cos(tick * node.orbitSpeed + node.phase) * node.orbitRadius * wobble;
    const y =
      node.baseY * compactness * distanceFactor +
      Math.sin(tick * (node.orbitSpeed + 0.08) + node.phase) *
        node.orbitRadius *
        wobble *
        0.84;

    positions[node.id] = { x, y };
  }

  useEffect(() => {
    panRef.current = pan;
  }, [pan]);

  useEffect(() => {
    zoomRef.current = zoom;
  }, [zoom]);

  useEffect(() => {
    if (!settings.animate) {
      return;
    }

    let frameId = 0;
    const animateFrame = (time: number) => {
      setTick(time / 1000);
      frameId = window.requestAnimationFrame(animateFrame);
    };

    frameId = window.requestAnimationFrame(animateFrame);
    return () => window.cancelAnimationFrame(frameId);
  }, [settings.animate]);

  /**
   * Convert client (screen) coordinates to SVG viewBox coordinates using
   * the SVG's native getScreenCTM(). This correctly handles
   * preserveAspectRatio, letterboxing, and any CSS transforms.
   */
  function clientToSvg(clientX: number, clientY: number) {
    const svg = svgRef.current;
    if (!svg) {
      return { x: 0, y: 0 };
    }
    const ctm = svg.getScreenCTM();
    if (!ctm) {
      return { x: 0, y: 0 };
    }
    const inv = ctm.inverse();
    return {
      x: inv.a * clientX + inv.c * clientY + inv.e,
      y: inv.b * clientX + inv.d * clientY + inv.f,
    };
  }

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      event.stopPropagation();

      const oldZoom = zoomRef.current;
      const nextZoom = Math.max(
        MIN_ZOOM,
        Math.min(MAX_ZOOM, oldZoom - event.deltaY * 0.0009),
      );

      // Zoom toward cursor: keep the SVG point under the cursor fixed.
      const cursor = clientToSvg(event.clientX, event.clientY);
      // The SVG point under cursor (in the inner g's coordinate frame) is:
      //   graphPt = (cursor - pan) / oldZoom
      // After zoom we want the same graphPt under the cursor:
      //   graphPt = (cursor - newPan) / nextZoom
      // Solving:  newPan = cursor - graphPt * nextZoom
      const currentPan = panRef.current;
      const graphX = (cursor.x - currentPan.x) / oldZoom;
      const graphY = (cursor.y - currentPan.y) / oldZoom;
      const newPanX = cursor.x - graphX * nextZoom;
      const newPanY = cursor.y - graphY * nextZoom;

      setZoom(nextZoom);
      setPan({ x: newPanX, y: newPanY });
    };

    container.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      container.removeEventListener("wheel", handleWheel);
    };
  }, []);

  function handlePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (!event.isPrimary || event.button !== 0) {
      return;
    }

    suppressClickRef.current = false;
    setIsPanning(true);

    // Record the graph-space point under the cursor as the anchor.
    // Transform is translate(pan) scale(zoom), so:
    //   svgPt = graphPt * zoom + pan  →  graphPt = (svgPt - pan) / zoom
    const svgPt = clientToSvg(event.clientX, event.clientY);
    const currentPan = panRef.current;
    const currentZoom = zoomRef.current;

    dragStateRef.current = {
      pointerId: event.pointerId,
      originClientX: event.clientX,
      originClientY: event.clientY,
      anchorX: (svgPt.x - currentPan.x) / currentZoom,
      anchorY: (svgPt.y - currentPan.y) / currentZoom,
      moved: false,
    };
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handlePointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const dragState = dragStateRef.current;
    if (!isPanning || dragState.pointerId !== event.pointerId) {
      return;
    }

    const rawDeltaX = event.clientX - dragState.originClientX;
    const rawDeltaY = event.clientY - dragState.originClientY;
    if (!dragState.moved && Math.hypot(rawDeltaX, rawDeltaY) >= DRAG_THRESHOLD) {
      dragState.moved = true;
      suppressClickRef.current = true;
      onHoverNode(null);
    }

    // Anchor-point panning: set pan so that the anchor graph point
    // stays exactly under the current cursor position.
    // svgPt = anchorGraphPt * zoom + newPan  →  newPan = svgPt - anchor * zoom
    const svgPt = clientToSvg(event.clientX, event.clientY);
    const currentZoom = zoomRef.current;
    setPan({
      x: svgPt.x - dragState.anchorX * currentZoom,
      y: svgPt.y - dragState.anchorY * currentZoom,
    });
  }

  function handlePointerUp(event: ReactPointerEvent<HTMLDivElement>) {
    const dragState = dragStateRef.current;
    if (dragState.pointerId !== event.pointerId) {
      return;
    }

    suppressClickRef.current = dragState.moved;
    dragStateRef.current = {
      pointerId: null,
      originClientX: 0,
      originClientY: 0,
      anchorX: 0,
      anchorY: 0,
      moved: false,
    };
    setIsPanning(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function handlePointerCancel(event: ReactPointerEvent<HTMLDivElement>) {
    const dragState = dragStateRef.current;
    if (dragState.pointerId !== event.pointerId) {
      return;
    }

    dragStateRef.current = {
      pointerId: null,
      originClientX: 0,
      originClientY: 0,
      anchorX: 0,
      anchorY: 0,
      moved: false,
    };
    setIsPanning(false);
  }

  function handleNodeClick(
    event: ReactMouseEvent<SVGGElement, MouseEvent>,
    nodeId: string,
  ) {
    event.stopPropagation();
    if (suppressClickRef.current) {
      suppressClickRef.current = false;
      return;
    }
    onSelectNode(nodeId);
  }

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative overflow-hidden rounded-[26px] border border-border/60 bg-[#121317] touch-none select-none overscroll-contain",
        isPanning ? "cursor-grabbing" : "cursor-grab",
        className,
      )}
      onPointerDown={handlePointerDown}
      onPointerLeave={() => {
        if (!dragStateRef.current.pointerId) {
          onHoverNode(null);
        }
      }}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerCancel}
    >
      <div className="pointer-events-none absolute inset-x-4 top-4 z-10 flex items-center justify-between gap-3">
        <div className="rounded-full border border-white/8 bg-black/35 px-3 py-1 text-[11px] uppercase tracking-[0.22em] text-white/65 backdrop-blur">
          Drag to pan, scroll to zoom
        </div>
        <div className="rounded-full border border-white/8 bg-black/35 px-3 py-1 text-xs text-white/70 backdrop-blur">
          {visibleNodes.length} nodes · {visibleLinks.length} links · {Math.round(zoom * 100)}%
        </div>
      </div>

      {visibleNodes.length === 0 ? (
        <div className="flex h-full min-h-[520px] items-center justify-center px-6 text-center text-sm text-muted-foreground">
          No clusters are visible right now. Re-enable a group to bring the vault graph back.
        </div>
      ) : (
        <svg ref={svgRef} className="h-full min-h-[520px] w-full" viewBox={VIEWBOX}>
          <defs>
            <pattern id="graph-grid" width="42" height="42" patternUnits="userSpaceOnUse">
              <path d="M 42 0 L 0 0 0 42" fill="none" stroke="rgba(255,255,255,0.03)" />
            </pattern>
            <marker
              id="graph-arrow"
              viewBox="0 0 10 10"
              refX="8.5"
              refY="5"
              markerWidth="5.5"
              markerHeight="5.5"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(202, 208, 221, 0.8)" />
            </marker>
          </defs>

          <rect x="-720" y="-460" width="1440" height="920" fill="url(#graph-grid)" />

          <g opacity="0.28">
            {[140, 240, 340, 440].map((radius) => (
              <circle
                key={radius}
                cx="0"
                cy="0"
                r={radius}
                fill="none"
                stroke="rgba(255,255,255,0.05)"
              />
            ))}
          </g>

          <g transform={`translate(${pan.x} ${pan.y}) scale(${zoom})`}>
            {visibleLinks.map((link) => {
              const source = positions[link.source];
              const target = positions[link.target];

              if (!source || !target) {
                return null;
              }

              const searchVisible =
                !query || (searchMatches.has(link.source) && searchMatches.has(link.target));
              const inFocus =
                neighborhood.size === 0 ||
                neighborhood.has(link.source) ||
                neighborhood.has(link.target);
              const dx = target.x - source.x;
              const dy = target.y - source.y;
              const distance = Math.max(1, Math.hypot(dx, dy));
              const normalX = -dy / distance;
              const normalY = dx / distance;
              const curve = Math.min(distance * 0.16, 34 + settings.linkForce * 26);
              const controlX = (source.x + target.x) / 2 + normalX * curve;
              const controlY = (source.y + target.y) / 2 + normalY * curve;
              const opacity = searchVisible ? (inFocus ? 0.48 : 0.12) : 0.05;
              const strokeWidth = 0.45 + settings.linkThickness * 2.1 + link.strength * 0.45;

              return (
                <path
                  key={`${link.source}-${link.target}`}
                  d={`M ${source.x} ${source.y} Q ${controlX} ${controlY} ${target.x} ${target.y}`}
                  fill="none"
                  markerEnd={settings.arrows ? "url(#graph-arrow)" : undefined}
                  stroke="rgba(202, 208, 221, 0.9)"
                  strokeOpacity={opacity}
                  strokeWidth={strokeWidth}
                />
              );
            })}

            {visibleNodes.map((node) => {
              const position = positions[node.id];
              if (!position) {
                return null;
              }

              const isActive = node.id === activeNodeId;
              const searchVisible = !query || searchMatches.has(node.id);
              const inFocus = neighborhood.size === 0 || neighborhood.has(node.id);
              const baseRadius = 2.6 + node.weight * 2 + settings.nodeSize * 5.2;
              const opacity = searchVisible ? (inFocus ? 1 : 0.3) : 0.14;
              const labelStrength =
                node.weight * 0.22 +
                (isActive ? 0.46 : 0) +
                (searchVisible ? 0.18 : 0) +
                zoom * 0.14;
              const labelOpacity = Math.max(
                0.06,
                Math.min(0.95, (labelStrength - settings.textFadeThreshold + 0.38) * 1.6),
              );

              return (
                <g
                  key={node.id}
                  className="cursor-pointer"
                  onClick={(event) => handleNodeClick(event, node.id)}
                  onPointerEnter={(event) => {
                    event.stopPropagation();
                    if (dragStateRef.current.pointerId) {
                      return;
                    }
                    onHoverNode(node.id);
                  }}
                  onPointerLeave={(event) => {
                    event.stopPropagation();
                    if (dragStateRef.current.pointerId) {
                      return;
                    }
                    onHoverNode(null);
                  }}
                >
                  {isActive ? (
                    <circle
                      cx={position.x}
                      cy={position.y}
                      r={baseRadius + 7}
                      fill="none"
                      stroke={node.color}
                      strokeOpacity="0.36"
                      strokeWidth="2"
                    />
                  ) : null}

                  <circle
                    cx={position.x}
                    cy={position.y}
                    r={baseRadius}
                    fill={node.color}
                    fillOpacity={opacity}
                    stroke="rgba(255,255,255,0.18)"
                    strokeWidth={isActive ? 1.4 : 0.5}
                  />

                  <text
                    x={position.x + baseRadius + 5}
                    y={position.y - baseRadius - 3}
                    fill="rgba(240,243,249,0.95)"
                    fillOpacity={labelOpacity * opacity}
                    fontSize="12"
                    letterSpacing="0.02em"
                    pointerEvents="none"
                  >
                    {node.title}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>
      )}
    </div>
  );
}
