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
  pellets: { x: number[]; y: number[]; mass: number[] };
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
let frameIntervalEstimate = 34;
let lastRenderAt = performance.now();
let lastHudUpdateAt = 0;
let leaderboardHtml = "";
let reconnectTimer = 0;
let reconnectAttempt = 0;
let pointer = { x: 0, y: 0, active: false };
let keyboard = { x: 0, y: 0 };
let smoothedSteer: Vec2 = { x: 0, y: 0 };
let lastInputAt = 0;
let camera = { x: 500, y: 500, zoom: 1 };
let currentMode = new URLSearchParams(location.search).get("mode") ?? defaultMode;

const inputIntervalMs = 1000 / 30;
const pointerDeadZone = 14;
const pointerFullSpeedDistance = 220;

elements.modeSelect.value = currentMode;

function connect() {
  clearTimeout(reconnectTimer);
  const previousSocket = socket;
  socket = null;
  previousSocket?.close();
  const checkpoint = encodeURIComponent(defaultCheckpoint);
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const url = `${protocol}://${location.hostname}:${apiPort}/ws?mode=${encodeURIComponent(currentMode)}&checkpoint=${checkpoint}`;
  const nextSocket = new WebSocket(url);
  socket = nextSocket;
  elements.status.textContent = "Connecting";

  nextSocket.addEventListener("open", () => {
    if (socket !== nextSocket) return;
    reconnectAttempt = 0;
    elements.status.textContent = "Live";
  });

  nextSocket.addEventListener("message", (event) => {
    if (socket !== nextSocket) return;
    const parsed = JSON.parse(event.data) as BrowserFrame;
    if (parsed.type !== "frame") {
      return;
    }
    previousFrame = frame;
    frame = parsed;
    const now = performance.now();
    const observedInterval = now - lastFrameAt;
    if (observedInterval > 5 && observedInterval < 500) {
      // Track actual server frame spacing so interpolation keeps gliding
      // for the full gap between frames instead of assuming a fixed 30Hz
      // and sitting frozen once that fixed window elapses.
      frameIntervalEstimate = lerp(frameIntervalEstimate, observedInterval, 0.25);
    }
    lastFrameAt = now;
    if (now - lastHudUpdateAt >= 200) {
      updateHud(parsed);
      lastHudUpdateAt = now;
    }
  });

  nextSocket.addEventListener("close", () => {
    if (socket !== nextSocket) return;
    socket = null;
    elements.status.textContent = "Reconnecting";
    const delay = Math.min(500 * 2 ** reconnectAttempt, 4000);
    reconnectAttempt += 1;
    reconnectTimer = window.setTimeout(connect, delay);
  });

  nextSocket.addEventListener("error", () => {
    if (socket !== nextSocket) return;
    elements.status.textContent = "Socket error";
  });
}

function sendInput(split = false, force = false, eject = false) {
  if (!socket || socket.readyState !== WebSocket.OPEN || !frame || currentMode !== "play") {
    return;
  }
  const now = performance.now();
  if (!force && now - lastInputAt < inputIntervalMs) {
    return;
  }
  lastInputAt = now;
  const steer = currentSteer();
  socket.send(JSON.stringify({ type: "input", steer, split, eject }));
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
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
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
  const nextLeaderboardHtml = next.leaderboard
    .slice(0, 7)
    .map(
      (item) =>
        `<li><span style="--agent-color:${item.color}">${escapeText(item.name)}</span><strong>${item.mass.toFixed(0)}</strong></li>`,
    )
    .join("");
  if (nextLeaderboardHtml !== leaderboardHtml) {
    elements.leaderboard.innerHTML = nextLeaderboardHtml;
    leaderboardHtml = nextLeaderboardHtml;
  }
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
  elements.pelletCount.textContent = String(next.pellets.x.length);
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

let cameraFocusId: string | null = null;
let cameraFocusHoldFrames = 0;
const CAMERA_SWITCH_HOLD_FRAMES = 90;
const CAMERA_SWITCH_LEAD_RATIO = 1.15;

function resetCameraFocus() {
  cameraFocusId = null;
  cameraFocusHoldFrames = 0;
}

function pickCameraFocus(snapshot: BrowserFrame): Agent | undefined {
  if (snapshot.playerId) {
    const player = snapshot.agents.find((agent) => agent.id === snapshot.playerId);
    if (player && player.alive) {
      cameraFocusId = player.id;
      return player;
    }
  }

  const aliveAgents = snapshot.agents.filter((agent) => agent.alive);
  if (aliveAgents.length === 0) {
    return snapshot.agents[0];
  }
  const topAgent = aliveAgents.reduce((best, agent) => (agent.totalMass > best.totalMass ? agent : best));
  const current = cameraFocusId ? aliveAgents.find((agent) => agent.id === cameraFocusId) : undefined;

  if (!current) {
    cameraFocusId = topAgent.id;
    cameraFocusHoldFrames = CAMERA_SWITCH_HOLD_FRAMES;
    return topAgent;
  }
  if (current.id === topAgent.id) {
    return current;
  }
  // Debounce: require the new leader to hold a clear mass lead for a while
  // before stealing camera focus, otherwise near-tied masses cause the
  // camera to snap back and forth between agents every frame.
  if (cameraFocusHoldFrames > 0) {
    cameraFocusHoldFrames -= 1;
    return current;
  }
  if (topAgent.totalMass >= current.totalMass * CAMERA_SWITCH_LEAD_RATIO) {
    cameraFocusId = topAgent.id;
    cameraFocusHoldFrames = CAMERA_SWITCH_HOLD_FRAMES;
    return topAgent;
  }
  return current;
}

function updateCamera(snapshot: BrowserFrame, alpha: number, elapsedMs: number) {
  const focus = pickCameraFocus(snapshot);
  if (!focus) {
    return;
  }
  const center = interpolatedAgentCenter(focus, alpha);
  const targetZoom = Math.max(0.34, Math.min(1.18, 46 / Math.sqrt(Math.max(focus.totalMass, 24))));
  const elapsedSeconds = Math.min(elapsedMs, 50) / 1000;
  const positionBlend = 1 - Math.exp(-7 * elapsedSeconds);
  const zoomBlend = 1 - Math.exp(-3.7 * elapsedSeconds);
  camera.x = lerp(camera.x, center.x, positionBlend);
  camera.y = lerp(camera.y, center.y, positionBlend);
  camera.zoom = lerp(camera.zoom, targetZoom, zoomBlend);
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

function draw(timestamp: number) {
  requestAnimationFrame(draw);
  ensureCanvasSize();
  if (!frame) {
    drawLoading();
    return;
  }
  const elapsedMs = timestamp - lastRenderAt;
  lastRenderAt = timestamp;
  const alpha = Math.min((timestamp - lastFrameAt) / frameIntervalEstimate, 1);
  updateCamera(frame, alpha, elapsedMs);
  drawArena(frame, alpha);
  drawMinimap(frame);
  sendInput();
}

function ensureCanvasSize() {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = window.innerWidth;
  const height = window.innerHeight;
  if (width <= 0 || height <= 0) {
    return;
  }
  const expectedWidth = Math.floor(width * ratio);
  const expectedHeight = Math.floor(height * ratio);
  if (canvas.width !== expectedWidth || canvas.height !== expectedHeight) {
    resizeCanvas();
  }
}

function drawLoading() {
  const width = canvas.clientWidth || window.innerWidth;
  const height = canvas.clientHeight || window.innerHeight;
  ctx.fillStyle = BACKGROUND_COLOR;
  ctx.fillRect(0, 0, width, height);
  drawGridPattern(width, height);
  ctx.fillStyle = "#4a5a68";
  ctx.font = "600 20px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("Connecting to Python simulation...", width * 0.5, height * 0.5);
  ctx.textAlign = "start";
  ctx.textBaseline = "alphabetic";
}

function drawArena(snapshot: BrowserFrame, alpha: number) {
  ctx.fillStyle = BACKGROUND_COLOR;
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  drawGridPattern(canvas.clientWidth, canvas.clientHeight);
  drawWorldBounds(snapshot.mapSize);

  const agentColorById = new Map(snapshot.agents.map((agent) => [agent.id, agent.color] as const));

  const pelletXs = snapshot.pellets.x;
  const pelletYs = snapshot.pellets.y;
  const pelletMasses = snapshot.pellets.mass;
  const pelletPaths = PELLET_PALETTE.map(() => new Path2D());
  const populatedPaths = new Array(PELLET_PALETTE.length).fill(false) as boolean[];
  for (let i = 0; i < pelletXs.length; i += 1) {
    const p = worldToScreen({ x: pelletXs[i], y: pelletYs[i] });
    const radius = Math.max(2.4, Math.sqrt(pelletMasses[i]) * 1.5 * camera.zoom);
    if (p.x < -radius || p.x > canvas.clientWidth + radius || p.y < -radius || p.y > canvas.clientHeight + radius) {
      continue;
    }
    const colorIndex = pelletColorIndex(pelletXs[i], pelletYs[i]);
    pelletPaths[colorIndex].moveTo(p.x + radius, p.y);
    pelletPaths[colorIndex].arc(p.x, p.y, radius, 0, Math.PI * 2);
    populatedPaths[colorIndex] = true;
  }
  for (let i = 0; i < pelletPaths.length; i += 1) {
    if (populatedPaths[i]) {
      ctx.fillStyle = PELLET_PALETTE[i];
      ctx.fill(pelletPaths[i]);
    }
  }

  for (const ejected of snapshot.ejected) {
    const p = worldToScreen(ejected);
    const radius = Math.max(3.5, 5.5 * camera.zoom);
    const color = agentColorById.get(ejected.ownerId) ?? "#cfd6dc";
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.strokeStyle = shade(color, -30);
    ctx.lineWidth = Math.max(1, radius * 0.12);
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  for (const virus of snapshot.viruses) {
    const p = worldToScreen(virus);
    drawVirus(p.x, p.y, Math.max(12, virus.radius * camera.zoom));
  }

  for (const agent of snapshot.agents) {
    for (const cell of agent.cells) {
      const p = worldToScreen(cellPosition(cell, agent.id, alpha));
      drawCell(p.x, p.y, Math.max(6, cell.radius * camera.zoom), agent);
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

const BACKGROUND_COLOR = "#f2f4f7";
const GRID_LINE_COLOR = "rgba(50, 70, 90, 0.08)";
const GRID_WORLD_SPACING = 50;

function drawGridPattern(width: number, height: number) {
  const grid = GRID_WORLD_SPACING * camera.zoom;
  if (grid < 4) {
    return;
  }
  const offsetX = ((-camera.x * camera.zoom + width * 0.5) % grid) - grid;
  const offsetY = ((-camera.y * camera.zoom + height * 0.5) % grid) - grid;
  ctx.strokeStyle = GRID_LINE_COLOR;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = offsetX; x < width + grid; x += grid) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
  }
  for (let y = offsetY; y < height + grid; y += grid) {
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
  }
  ctx.stroke();
}

function drawWorldBounds(mapSize: number) {
  const a = worldToScreen({ x: 0, y: 0 });
  const b = worldToScreen({ x: mapSize, y: mapSize });
  ctx.strokeStyle = "rgba(60, 70, 82, 0.5)";
  ctx.lineWidth = 2;
  ctx.setLineDash([12, 9]);
  ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
  ctx.setLineDash([]);
}

function drawCell(x: number, y: number, radius: number, agent: Agent) {
  ctx.beginPath();
  ctx.fillStyle = agent.color;
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = shade(agent.color, -38);
  ctx.lineWidth = Math.max(2, radius * 0.09);
  ctx.stroke();

  if (radius > 13) {
    const nameSize = Math.max(11, Math.min(22, radius * 0.34));
    drawOutlinedText(agent.name, x, y - (radius > 26 ? nameSize * 0.42 : 0), nameSize);
    if (radius > 26) {
      const massSize = Math.max(9, nameSize * 0.7);
      drawOutlinedText(Math.round(agent.totalMass).toString(), x, y + massSize * 0.7, massSize);
    }
  }
}

function drawOutlinedText(text: string, x: number, y: number, size: number) {
  ctx.font = `700 ${size}px Inter, system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineWidth = Math.max(2, size * 0.22);
  ctx.strokeStyle = "rgba(0, 0, 0, 0.55)";
  ctx.strokeText(text, x, y);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(text, x, y);
  ctx.textAlign = "start";
  ctx.textBaseline = "alphabetic";
}

function drawVirus(x: number, y: number, radius: number) {
  ctx.save();
  ctx.translate(x, y);
  ctx.beginPath();
  const spikes = 24;
  for (let i = 0; i < spikes; i += 1) {
    const angle = (i / spikes) * Math.PI * 2;
    const r = i % 2 === 0 ? radius * 1.14 : radius * 0.84;
    const px = Math.cos(angle) * r;
    const py = Math.sin(angle) * r;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fillStyle = "#33ff33";
  ctx.strokeStyle = "#1fae1f";
  ctx.lineWidth = 3;
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
  miniCtx.fillStyle = "rgba(242, 244, 247, 0.9)";
  miniCtx.fillRect(0, 0, size, size);
  miniCtx.strokeStyle = "rgba(22, 35, 47, 0.35)";
  miniCtx.strokeRect(0.5, 0.5, size - 1, size - 1);
  const scale = size / snapshot.mapSize;
  miniCtx.fillStyle = "rgba(83, 107, 122, 0.22)";
  const miniPelletLimit = Math.min(120, snapshot.pellets.x.length);
  for (let i = 0; i < miniPelletLimit; i += 1) {
    miniCtx.fillRect(snapshot.pellets.x[i] * scale, snapshot.pellets.y[i] * scale, 1.5, 1.5);
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

const PELLET_PALETTE = ["#ff5252", "#33cc66", "#3399ff", "#ffcc33", "#cc66ff", "#ff9933", "#33cccc"];

function pelletColorIndex(x: number, y: number) {
  const key = Math.round(x) * 131071 + Math.round(y);
  return Math.abs(key) % PELLET_PALETTE.length;
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
  if (event.code === "ArrowUp") keyboard.y = -1;
  if (event.code === "ArrowDown") keyboard.y = 1;
  if (event.code === "ArrowLeft") keyboard.x = -1;
  if (event.code === "ArrowRight") keyboard.x = 1;
  if (event.code === "Space") {
    event.preventDefault();
    if (!event.repeat) {
      sendInput(true, true);
    }
  }
  if (event.code === "KeyW") {
    sendInput(false, true, true);
  }
  if (event.code === "KeyR") {
    socket?.send(JSON.stringify({ type: "reset" }));
    resetCameraFocus();
  }
});
window.addEventListener("keyup", (event) => {
  if (event.code === "ArrowUp" && keyboard.y < 0) keyboard.y = 0;
  if (event.code === "ArrowDown" && keyboard.y > 0) keyboard.y = 0;
  if (event.code === "ArrowLeft" && keyboard.x < 0) keyboard.x = 0;
  if (event.code === "ArrowRight" && keyboard.x > 0) keyboard.x = 0;
});
elements.resetButton.addEventListener("click", () => {
  socket?.send(JSON.stringify({ type: "reset" }));
  resetCameraFocus();
});
elements.modeSelect.addEventListener("change", () => {
  currentMode = elements.modeSelect.value;
  pointer.active = false;
  keyboard = { x: 0, y: 0 };
  smoothedSteer = { x: 0, y: 0 };
  resetCameraFocus();
  const nextUrl = new URL(location.href);
  nextUrl.searchParams.set("mode", currentMode);
  history.replaceState(null, "", nextUrl);
  connect();
});

resizeCanvas();
connect();
requestAnimationFrame(draw);
