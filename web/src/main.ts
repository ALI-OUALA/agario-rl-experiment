import "./styles.css";

type Vec2 = { x: number; y: number };

type Cell = {
  id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  mass: number;
  radius: number;
};

type Agent = {
  id: string;
  name: string;
  color: string;
  alive: boolean;
  totalMass: number;
  center: Vec2;
  cells: Cell[];
  split: { attempts: number; successful: number; unsafe: number; useful: number };
  threat?: { id: string; delta: Vec2; distance: number; massRatio: number } | null;
  target?: { id: string; delta: Vec2; distance: number; massRatio: number } | null;
};

type BrowserFrame = {
  type: "frame";
  mode: string;
  tick: number;
  fps: number;
  scenario: string;
  mapSize: number;
  playerId: string | null;
  agents: Agent[];
  leaderboard: Array<{ id: string; name: string; mass: number; color: string }>;
  pellets: Array<{ id: string | number; x: number; y: number; mass: number }>;
  viruses: Array<{ id: string | number; x: number; y: number; mass: number; radius: number; fed: number }>;
  ejected: Array<{ id: string | number; ownerId: string; x: number; y: number; mass: number }>;
  training: {
    policySource: string;
    checkpoint: string;
    updateCount: number;
    metrics: Record<string, number>;
    humanReadiness: {
      splitSafety: number;
      unsafeSplits: number;
      usefulSplits: number;
      finalMassLeader: number;
    };
  };
};

const canvasElement = document.querySelector<HTMLCanvasElement>("#game");
const minimapElement = document.querySelector<HTMLCanvasElement>("#minimap");
if (!canvasElement || !minimapElement) {
  throw new Error("Canvas elements are missing.");
}
const canvas: HTMLCanvasElement = canvasElement;
const minimap: HTMLCanvasElement = minimapElement;

const canvasContext = canvas.getContext("2d", { alpha: false });
const minimapContext = minimap.getContext("2d", { alpha: true });
if (!canvasContext || !minimapContext) {
  throw new Error("Canvas context is unavailable.");
}
const ctx: CanvasRenderingContext2D = canvasContext;
const miniCtx: CanvasRenderingContext2D = minimapContext;

const elements = {
  modeSelect: document.querySelector<HTMLSelectElement>("#modeSelect")!,
  resetButton: document.querySelector<HTMLButtonElement>("#resetButton")!,
  leaderboard: document.querySelector<HTMLOListElement>("#leaderboard")!,
  scenario: document.querySelector<HTMLElement>("#scenario")!,
  updateCount: document.querySelector<HTMLElement>("#updateCount")!,
  policySource: document.querySelector<HTMLElement>("#policySource")!,
  checkpoint: document.querySelector<HTMLElement>("#checkpoint")!,
  usefulSplits: document.querySelector<HTMLElement>("#usefulSplits")!,
  unsafeSplits: document.querySelector<HTMLElement>("#unsafeSplits")!,
  splitSafety: document.querySelector<HTMLElement>("#splitSafety")!,
  leaderMass: document.querySelector<HTMLElement>("#leaderMass")!,
  fps: document.querySelector<HTMLElement>("#fps")!,
  agentCount: document.querySelector<HTMLElement>("#agentCount")!,
  pelletCount: document.querySelector<HTMLElement>("#pelletCount")!,
  status: document.querySelector<HTMLElement>("#status")!,
};

const apiPort = import.meta.env.VITE_AGARIO_API_PORT ?? "8765";
const defaultMode = import.meta.env.VITE_AGARIO_DEFAULT_MODE ?? "showcase";
const defaultCheckpoint = import.meta.env.VITE_AGARIO_DEFAULT_CHECKPOINT ?? "checkpoints/human_ready_v1/latest.pt";

let socket: WebSocket | null = null;
let frame: BrowserFrame | null = null;
let previousFrame: BrowserFrame | null = null;
let lastFrameAt = performance.now();
let reconnectTimer = 0;
let pointer = { x: 0, y: 0, active: false };
let keyboard = { x: 0, y: 0 };
let smoothedSteer: Vec2 = { x: 0, y: 0 };
let lastInputAt = 0;
let pendingSplit = false;
let camera = { x: 500, y: 500, zoom: 1 };
let currentMode = new URLSearchParams(location.search).get("mode") ?? defaultMode;

const inputIntervalMs = 1000 / 30;
const pointerDeadZone = 14;
const pointerFullSpeedDistance = 220;

elements.modeSelect.value = currentMode;

function connect() {
  clearTimeout(reconnectTimer);
  socket?.close();
  const checkpoint = encodeURIComponent(defaultCheckpoint);
  const url = `ws://127.0.0.1:${apiPort}/ws?mode=${encodeURIComponent(currentMode)}&checkpoint=${checkpoint}`;
  socket = new WebSocket(url);
  elements.status.textContent = "Connecting";

  socket.addEventListener("open", () => {
    elements.status.textContent = "Live";
  });

  socket.addEventListener("message", (event) => {
    const parsed = JSON.parse(event.data) as BrowserFrame;
    if (parsed.type !== "frame") {
      return;
    }
    previousFrame = frame;
    frame = parsed;
    lastFrameAt = performance.now();
    updateHud(parsed);
  });

  socket.addEventListener("close", () => {
    elements.status.textContent = "Reconnecting";
    reconnectTimer = window.setTimeout(connect, 900);
  });

  socket.addEventListener("error", () => {
    elements.status.textContent = "Socket error";
  });
}

function sendInput(split = false, force = false) {
  if (!socket || socket.readyState !== WebSocket.OPEN || !frame || currentMode !== "play") {
    return;
  }
  const now = performance.now();
  if (!force && now - lastInputAt < inputIntervalMs) {
    return;
  }
  lastInputAt = now;
  const steer = currentSteer();
  socket.send(JSON.stringify({ type: "input", steer, split }));
}

function currentSteer(): Vec2 {
  const keyboardLength = Math.hypot(keyboard.x, keyboard.y);
  const target =
    keyboardLength > 0
      ? { x: keyboard.x / keyboardLength, y: keyboard.y / keyboardLength }
      : pointer.active
        ? screenToSteer(pointer.x, pointer.y)
        : { x: 0, y: 0 };
  smoothedSteer = {
    x: lerp(smoothedSteer.x, target.x, 0.38),
    y: lerp(smoothedSteer.y, target.y, 0.38),
  };
  if (Math.hypot(smoothedSteer.x, smoothedSteer.y) < 0.015) {
    smoothedSteer = { x: 0, y: 0 };
  }
  return smoothedSteer;
}

function playerScreenOrigin(): Vec2 {
  const player =
    frame?.playerId ? frame.agents.find((agent) => agent.id === frame?.playerId && agent.alive) : undefined;
  return player ? worldToScreen(player.center) : { x: canvas.clientWidth * 0.5, y: canvas.clientHeight * 0.5 };
}

function screenToSteer(x: number, y: number): Vec2 {
  const origin = playerScreenOrigin();
  const dx = x - origin.x;
  const dy = y - origin.y;
  const length = Math.hypot(dx, dy);
  if (length < pointerDeadZone) {
    return { x: 0, y: 0 };
  }
  const strength = Math.min(1, (length - pointerDeadZone) / pointerFullSpeedDistance);
  return { x: (dx / length) * strength, y: (dy / length) * strength };
}

function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  const width = window.innerWidth;
  const height = window.innerHeight;
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function pointerFromEvent(event: PointerEvent): Vec2 {
  const rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

function updateHud(next: BrowserFrame) {
  elements.leaderboard.innerHTML = next.leaderboard
    .slice(0, 7)
    .map(
      (item) =>
        `<li><span style="--agent-color:${item.color}">${escapeText(item.name)}</span><strong>${item.mass.toFixed(0)}</strong></li>`,
    )
    .join("");
  elements.scenario.textContent = next.scenario;
  elements.updateCount.textContent = String(next.training.updateCount || 0);
  elements.policySource.textContent = next.training.policySource;
  elements.checkpoint.textContent = shortCheckpoint(next.training.checkpoint);
  elements.usefulSplits.textContent = String(next.training.humanReadiness.usefulSplits);
  elements.unsafeSplits.textContent = String(next.training.humanReadiness.unsafeSplits);
  elements.splitSafety.textContent = String(next.training.humanReadiness.splitSafety);
  elements.leaderMass.textContent = next.training.humanReadiness.finalMassLeader.toFixed(0);
  elements.fps.textContent = next.fps.toFixed(0);
  elements.agentCount.textContent = String(next.agents.filter((agent) => agent.alive).length);
  elements.pelletCount.textContent = String(next.pellets.length);
}

function shortCheckpoint(path: string) {
  const parts = path.replaceAll("\\", "/").split("/");
  return parts.slice(-2).join("/");
}

function escapeText(value: string) {
  const div = document.createElement("div");
  div.textContent = value;
  return div.innerHTML;
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function agentById(snapshot: BrowserFrame | null, id: string): Agent | undefined {
  return snapshot?.agents.find((agent) => agent.id === id);
}

function cellPosition(cell: Cell, agentId: string, alpha: number): Vec2 {
  const oldAgent = agentById(previousFrame, agentId);
  const oldCell = oldAgent?.cells.find((candidate) => candidate.id === cell.id);
  if (!oldCell) {
    return { x: cell.x, y: cell.y };
  }
  return { x: lerp(oldCell.x, cell.x, alpha), y: lerp(oldCell.y, cell.y, alpha) };
}

function updateCamera(snapshot: BrowserFrame, alpha: number) {
  const playerFocus = snapshot.playerId ? snapshot.agents.find((agent) => agent.id === snapshot.playerId) : undefined;
  const leaderFocus =
    snapshot.leaderboard.length > 0
      ? snapshot.agents.find((agent) => agent.id === snapshot.leaderboard[0]?.id)
      : undefined;
  const focus = playerFocus ?? leaderFocus ?? snapshot.agents[0];
  if (!focus) {
    return;
  }
  const center = interpolatedAgentCenter(focus, alpha);
  const targetZoom = Math.max(0.34, Math.min(1.18, 46 / Math.sqrt(Math.max(focus.totalMass, 24))));
  camera.x = lerp(camera.x, center.x, 0.11);
  camera.y = lerp(camera.y, center.y, 0.11);
  camera.zoom = lerp(camera.zoom, targetZoom, 0.06);
}

function interpolatedAgentCenter(agent: Agent, alpha: number): Vec2 {
  const oldAgent = agentById(previousFrame, agent.id);
  if (!oldAgent) {
    return agent.center;
  }
  return {
    x: lerp(oldAgent.center.x, agent.center.x, alpha),
    y: lerp(oldAgent.center.y, agent.center.y, alpha),
  };
}

function worldToScreen(pos: Vec2): Vec2 {
  return {
    x: (pos.x - camera.x) * camera.zoom + canvas.clientWidth * 0.5,
    y: (pos.y - camera.y) * camera.zoom + canvas.clientHeight * 0.5,
  };
}

function draw() {
  requestAnimationFrame(draw);
  if (!frame) {
    drawLoading();
    return;
  }
  const alpha = Math.min((performance.now() - lastFrameAt) / 34, 1);
  updateCamera(frame, alpha);
  drawArena(frame);
  drawMinimap(frame);
  sendInput(pendingSplit);
  pendingSplit = false;
}

function drawLoading() {
  ctx.fillStyle = "#091017";
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  ctx.fillStyle = "#d7f7ff";
  ctx.font = "600 18px Inter, system-ui, sans-serif";
  ctx.fillText("Connecting to Python simulation...", 32, 48);
}

function drawArena(snapshot: BrowserFrame) {
  ctx.fillStyle = "#f7fbf8";
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  drawGrid(snapshot.mapSize);
  drawWorldBounds(snapshot.mapSize);

  for (const pellet of snapshot.pellets) {
    const p = worldToScreen(pellet);
    const radius = Math.max(2.2, Math.sqrt(pellet.mass) * 1.15 * camera.zoom);
    ctx.beginPath();
    ctx.fillStyle = pelletColor(pellet.id);
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  for (const ejected of snapshot.ejected) {
    const p = worldToScreen(ejected);
    ctx.beginPath();
    ctx.fillStyle = "#7fd6ff";
    ctx.arc(p.x, p.y, Math.max(3.2, 5 * camera.zoom), 0, Math.PI * 2);
    ctx.fill();
  }

  for (const virus of snapshot.viruses) {
    const p = worldToScreen(virus);
    drawVirus(p.x, p.y, Math.max(10, virus.radius * camera.zoom));
  }

  const alpha = Math.min((performance.now() - lastFrameAt) / 34, 1);
  for (const agent of snapshot.agents) {
    for (const cell of agent.cells) {
      const p = worldToScreen(cellPosition(cell, agent.id, alpha));
      drawCell(p.x, p.y, Math.max(5, cell.radius * camera.zoom), agent);
    }
  }

  for (const agent of snapshot.agents) {
    if (agent.threat) {
      drawEdgeIndicator(agent, agent.threat, "#ff3b5f");
    }
    if (agent.target) {
      drawEdgeIndicator(agent, agent.target, "#2ebd59");
    }
  }

  drawCursor();
}

function drawGrid(mapSize: number) {
  const grid = 80 * camera.zoom;
  const offsetX = ((-camera.x * camera.zoom + canvas.clientWidth * 0.5) % grid) - grid;
  const offsetY = ((-camera.y * camera.zoom + canvas.clientHeight * 0.5) % grid) - grid;
  ctx.strokeStyle = "rgba(80, 98, 112, 0.15)";
  ctx.lineWidth = 1;
  for (let x = offsetX; x < canvas.clientWidth + grid; x += grid) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.clientHeight);
    ctx.stroke();
  }
  for (let y = offsetY; y < canvas.clientHeight + grid; y += grid) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.clientWidth, y);
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(24, 45, 60, 0.08)";
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.fillText(`${mapSize.toFixed(0)} x ${mapSize.toFixed(0)} arena`, 18, canvas.clientHeight - 24);
}

function drawWorldBounds(mapSize: number) {
  const a = worldToScreen({ x: 0, y: 0 });
  const b = worldToScreen({ x: mapSize, y: mapSize });
  ctx.strokeStyle = "rgba(22, 38, 52, 0.55)";
  ctx.lineWidth = 2;
  ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
}

function drawCell(x: number, y: number, radius: number, agent: Agent) {
  const gradient = ctx.createRadialGradient(x - radius * 0.28, y - radius * 0.35, radius * 0.1, x, y, radius);
  gradient.addColorStop(0, "#ffffff");
  gradient.addColorStop(0.18, agent.color);
  gradient.addColorStop(1, shade(agent.color, -32));
  ctx.beginPath();
  ctx.fillStyle = gradient;
  ctx.strokeStyle = "rgba(255, 255, 255, 0.96)";
  ctx.lineWidth = Math.max(2, radius * 0.07);
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  if (radius > 16) {
    ctx.fillStyle = "rgba(11, 20, 30, 0.74)";
    ctx.font = "700 13px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(agent.name, x, y);
    ctx.textAlign = "start";
  }
}

function drawVirus(x: number, y: number, radius: number) {
  ctx.save();
  ctx.translate(x, y);
  ctx.beginPath();
  const spikes = 20;
  for (let i = 0; i < spikes; i += 1) {
    const angle = (i / spikes) * Math.PI * 2;
    const r = i % 2 === 0 ? radius * 1.16 : radius * 0.86;
    const px = Math.cos(angle) * r;
    const py = Math.sin(angle) * r;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fillStyle = "#20c86d";
  ctx.strokeStyle = "#107b44";
  ctx.lineWidth = 2;
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawEdgeIndicator(agent: Agent, relation: NonNullable<Agent["threat"]>, color: string) {
  if (!frame || agent.cells.length === 0) {
    return;
  }
  const target = { x: agent.center.x + relation.delta.x, y: agent.center.y + relation.delta.y };
  const screen = worldToScreen(target);
  const margin = 26;
  if (screen.x >= margin && screen.x <= canvas.clientWidth - margin && screen.y >= margin && screen.y <= canvas.clientHeight - margin) {
    return;
  }
  const center = { x: canvas.clientWidth * 0.5, y: canvas.clientHeight * 0.5 };
  const angle = Math.atan2(screen.y - center.y, screen.x - center.x);
  const x = Math.min(canvas.clientWidth - margin, Math.max(margin, center.x + Math.cos(angle) * (canvas.clientWidth * 0.47)));
  const y = Math.min(canvas.clientHeight - margin, Math.max(margin, center.y + Math.sin(angle) * (canvas.clientHeight * 0.43)));
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.beginPath();
  ctx.moveTo(13, 0);
  ctx.lineTo(-8, -9);
  ctx.lineTo(-5, 0);
  ctx.lineTo(-8, 9);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.restore();
}

function drawCursor() {
  if (!pointer.active || currentMode !== "play") {
    return;
  }
  const origin = playerScreenOrigin();
  ctx.strokeStyle = "rgba(23, 45, 59, 0.44)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(origin.x, origin.y);
  ctx.lineTo(pointer.x, pointer.y);
  ctx.stroke();

  ctx.beginPath();
  ctx.fillStyle = "rgba(23, 45, 59, 0.18)";
  ctx.arc(origin.x, origin.y, pointerDeadZone, 0, Math.PI * 2);
  ctx.fill();
}

function drawMinimap(snapshot: BrowserFrame) {
  const size = minimap.width;
  miniCtx.clearRect(0, 0, size, size);
  miniCtx.fillStyle = "rgba(250, 253, 250, 0.82)";
  miniCtx.fillRect(0, 0, size, size);
  miniCtx.strokeStyle = "rgba(22, 35, 47, 0.35)";
  miniCtx.strokeRect(0.5, 0.5, size - 1, size - 1);
  const scale = size / snapshot.mapSize;
  for (const pellet of snapshot.pellets.slice(0, 120)) {
    miniCtx.fillStyle = "rgba(83, 107, 122, 0.22)";
    miniCtx.fillRect(pellet.x * scale, pellet.y * scale, 1.5, 1.5);
  }
  for (const virus of snapshot.viruses) {
    miniCtx.fillStyle = "#1fb760";
    miniCtx.beginPath();
    miniCtx.arc(virus.x * scale, virus.y * scale, 3, 0, Math.PI * 2);
    miniCtx.fill();
  }
  for (const agent of snapshot.agents) {
    miniCtx.fillStyle = agent.color;
    miniCtx.beginPath();
    miniCtx.arc(agent.center.x * scale, agent.center.y * scale, Math.max(3, Math.sqrt(agent.totalMass) * 0.25), 0, Math.PI * 2);
    miniCtx.fill();
  }
  const viewW = canvas.clientWidth / camera.zoom;
  const viewH = canvas.clientHeight / camera.zoom;
  miniCtx.strokeStyle = "rgba(10, 20, 30, 0.66)";
  miniCtx.strokeRect((camera.x - viewW * 0.5) * scale, (camera.y - viewH * 0.5) * scale, viewW * scale, viewH * scale);
}

function pelletColor(id: string | number) {
  const palette = ["#ff6b7a", "#41c46a", "#3fa7ff", "#ffcc4d", "#ad7cff", "#ff8a3d"];
  const key = String(id);
  const index = [...key].reduce((acc, char) => acc + char.charCodeAt(0), 0) % palette.length;
  return palette[index];
}

function shade(hex: string, delta: number) {
  const raw = hex.replace("#", "");
  const value = Number.parseInt(raw, 16);
  const r = Math.max(0, Math.min(255, (value >> 16) + delta));
  const g = Math.max(0, Math.min(255, ((value >> 8) & 255) + delta));
  const b = Math.max(0, Math.min(255, (value & 255) + delta));
  return `rgb(${r}, ${g}, ${b})`;
}

window.addEventListener("resize", resizeCanvas);
canvas.addEventListener("pointermove", (event) => {
  const nextPointer = pointerFromEvent(event);
  pointer = { x: nextPointer.x, y: nextPointer.y, active: true };
});
canvas.addEventListener("pointerdown", (event) => {
  canvas.setPointerCapture(event.pointerId);
  const nextPointer = pointerFromEvent(event);
  pointer = { x: nextPointer.x, y: nextPointer.y, active: true };
});
canvas.addEventListener("pointerleave", () => {
  pointer.active = false;
  sendInput(false, true);
});
canvas.addEventListener("pointerup", () => {
  pointer.active = false;
  sendInput(false, true);
});
window.addEventListener("keydown", (event) => {
  if (event.code === "KeyW" || event.code === "ArrowUp") keyboard.y = -1;
  if (event.code === "KeyS" || event.code === "ArrowDown") keyboard.y = 1;
  if (event.code === "KeyA" || event.code === "ArrowLeft") keyboard.x = -1;
  if (event.code === "KeyD" || event.code === "ArrowRight") keyboard.x = 1;
  if (event.code === "Space") {
    event.preventDefault();
    if (!event.repeat) {
      pendingSplit = true;
      sendInput(true, true);
      pendingSplit = false;
    }
  }
  if (event.code === "KeyR") {
    socket?.send(JSON.stringify({ type: "reset" }));
  }
});
window.addEventListener("keyup", (event) => {
  if ((event.code === "KeyW" || event.code === "ArrowUp") && keyboard.y < 0) keyboard.y = 0;
  if ((event.code === "KeyS" || event.code === "ArrowDown") && keyboard.y > 0) keyboard.y = 0;
  if ((event.code === "KeyA" || event.code === "ArrowLeft") && keyboard.x < 0) keyboard.x = 0;
  if ((event.code === "KeyD" || event.code === "ArrowRight") && keyboard.x > 0) keyboard.x = 0;
});
elements.resetButton.addEventListener("click", () => {
  socket?.send(JSON.stringify({ type: "reset" }));
});
elements.modeSelect.addEventListener("change", () => {
  currentMode = elements.modeSelect.value;
  pointer.active = false;
  keyboard = { x: 0, y: 0 };
  smoothedSteer = { x: 0, y: 0 };
  const nextUrl = new URL(location.href);
  nextUrl.searchParams.set("mode", currentMode);
  history.replaceState(null, "", nextUrl);
  connect();
});

resizeCanvas();
connect();
draw();
