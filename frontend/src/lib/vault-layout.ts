import type { McpGraphSnapshot } from "@/lib/mcp-types";
import type { VaultDataset, VaultGroup, VaultNode } from "@/lib/mock-vault";

export function layoutSnapshot(snapshot: McpGraphSnapshot): VaultDataset {
  const groups = snapshot.groups.map((group) => ({ ...group }));
  const groupCount = groups.length || 1;
  const groupCenters = new Map<string, { angle: number; radius: number; x: number; y: number }>();

  groups.forEach((group, index) => {
    const angle = -Math.PI / 2 + (index / groupCount) * Math.PI * 2;
    const radius = groupCount === 1 ? 0 : 248 + (index % 2) * 34;
    groupCenters.set(group.id, {
      angle,
      radius,
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius * 0.74,
    });
  });

  const nodesByGroup = new Map<string, VaultNode[]>();
  const seededNodes: VaultNode[] = snapshot.nodes.map((node) => ({
    ...node,
    orbitRadius: 0,
    orbitSpeed: 0,
    phase: 0,
    baseX: 0,
    baseY: 0,
  }));

  for (const node of seededNodes) {
    const groupNodes = nodesByGroup.get(node.groupId) ?? [];
    groupNodes.push(node);
    nodesByGroup.set(node.groupId, groupNodes);
  }

  const positionedNodes: VaultNode[] = [];
  for (const group of groups) {
    const members = (nodesByGroup.get(group.id) ?? []).slice().sort((left, right) => {
      return right.connections - left.connections || right.weight - left.weight || left.title.localeCompare(right.title);
    });
    const center = groupCenters.get(group.id) ?? { angle: 0, radius: 0, x: 0, y: 0 };

    members.forEach((member, index) => {
      if (index === 0) {
        positionedNodes.push({
          ...member,
          baseX: center.x,
          baseY: center.y,
          orbitRadius: 5,
          orbitSpeed: 0.14,
          phase: center.angle,
          weight: Math.max(member.weight, 2),
        });
        return;
      }

      const ring = 98 + ((index - 1) % 4) * 22 + Math.floor((index - 1) / 12) * 20;
      const theta =
        center.angle * 0.4 +
        (((index - 1) % Math.max(members.length - 1, 1)) / Math.max(members.length - 1, 1)) *
          Math.PI *
          2 +
        (index % 2 === 0 ? -0.08 : 0.11);

      positionedNodes.push({
        ...member,
        baseX: center.x + Math.cos(theta) * ring,
        baseY: center.y + Math.sin(theta) * ring * 0.84,
        orbitRadius: 5 + (index % 4) * 2,
        orbitSpeed: 0.16 + (index % 5) * 0.025,
        phase: index * 0.41 + center.angle,
      });
    });
  }

  return {
    groups,
    nodes: positionedNodes,
    links: snapshot.links.map((link) => ({ ...link })),
  };
}

export function buildMcpStatusLabel(snapshot: McpGraphSnapshot) {
  return `${snapshot.stats.total_memories} memories via ${snapshot.stats.transport} MCP`;
}
