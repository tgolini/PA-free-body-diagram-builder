"""
Free-Body Diagram Builder
A Streamlit app for practicing Newton's laws.

Students:
  1. Read a situation.
  2. Drag forces onto a free-body diagram, aiming each arrow the way the force acts.
  3. Once the diagram is correct, write Newton's second law for the x- and y-axes.

Run locally:   streamlit run app.py
Files needed: app.py and requirements.txt (nothing else)
"""

import re
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_application,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

st.set_page_config(page_title="Free-Body Diagram Builder", page_icon="🧭", layout="wide")

# HTML + JavaScript for the drag-and-drop diagram
COMPONENT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  :root { --fg: #262730; --border: rgba(128,128,128,.35); --panel: rgba(128,128,128,.06); --grid: rgba(128,128,128,.13); }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: transparent; color: var(--fg);
    font-family: "Source Sans Pro", "Source Sans 3", system-ui, -apple-system, sans-serif; }
  #root { padding: 2px 2px 10px; }
  .panels { display: flex; gap: 14px; flex-wrap: wrap; }
  .panel { flex: 1 1 300px; min-width: 270px; border: 1px solid var(--border); border-radius: 10px;
    background: var(--panel); padding: 8px 10px 6px; }
  .panel h4 { margin: 0 0 4px; font-size: 15px; font-weight: 600; }
  svg { width: 100%; height: auto; display: block; }
  #fbd { touch-action: none; }
  .note { font-size: 13px; opacity: .78; margin: 2px 0 0; min-height: 18px; }
  .palette-title { margin: 14px 0 6px; font-size: 14px; font-weight: 600; }
  #palette { display: flex; flex-wrap: wrap; gap: 8px; }
  .chip { display: inline-flex; align-items: center; gap: 7px; padding: 6px 11px; border-radius: 999px;
    border: 1.5px solid var(--border); background: var(--panel); cursor: grab; user-select: none;
    -webkit-user-select: none; touch-action: none; font-size: 14px; color: var(--fg); }
  .chip:hover { border-color: currentColor; }
  .chip.used { opacity: .5; }
  .chip.locked { cursor: default; opacity: .45; }
  .sw { width: 20px; height: 4px; border-radius: 2px; flex: none; }
  .sym { font-family: "Times New Roman", Times, serif; font-style: italic; font-size: 17px; font-weight: 600; }
  .sym sub { font-size: 11px; }
  #placed { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; font-size: 13px; }
  .tag { display: inline-flex; align-items: center; gap: 2px; padding: 1px 2px 1px 9px; border-radius: 6px;
    border: 1px solid var(--border); }
  .tag button, .clear { border: none; background: transparent; color: inherit; cursor: pointer;
    font-size: 15px; line-height: 1; padding: 3px 7px; border-radius: 4px; }
  .tag button:hover, .clear:hover { background: rgba(128,128,128,.2); }
  .clear { margin-left: auto; font-size: 13px; border: 1px solid var(--border); }
  .ghost { position: fixed; pointer-events: none; z-index: 1000; transform: translate(-50%, -50%); }
</style>
</head>
<body>
<div id="root">
  <div class="panels">
    <div class="panel">
      <h4>Situation</h4>
      <svg id="scene" viewBox="0 0 320 240" role="img" aria-label="Sketch of the situation"></svg>
      <p class="note" id="caption"></p>
    </div>
    <div class="panel">
      <h4>Your free-body diagram</h4>
      <svg id="fbd" viewBox="-170 -150 340 300" role="img" aria-label="Free-body diagram workspace"></svg>
      <p class="note" id="status"></p>
    </div>
  </div>
  <div class="palette-title" id="paletteTitle">Forces you can use (not all of them are real forces)</div>
  <div id="palette"></div>
  <div id="placed"></div>
</div>

<script>
const NS = "http://www.w3.org/2000/svg";
const fbd = document.getElementById("fbd");
const sceneSvg = document.getElementById("scene");
const statusEl = document.getElementById("status");

let scenario = null, forces = [], placed = [], locked = false, initialized = false, preview = null;

// ---------- Streamlit component protocol (no build tools needed) ----------
function post(type, extra) {
  window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type: type }, extra || {}), "*");
}
function sendValue() {
  post("streamlit:setComponentValue", {
    value: placed.map(p => ({ id: p.id, angle: p.angle, len: Math.round(p.len) })),
    dataType: "json",
  });
}
let lastHeight = 0;
function sendHeight() {
  const h = Math.ceil(document.getElementById("root").getBoundingClientRect().height) + 6;
  if (h !== lastHeight) { lastHeight = h; post("streamlit:setFrameHeight", { height: h }); }
}

window.addEventListener("message", (event) => {
  const d = event.data;
  if (!d || d.type !== "streamlit:render") return;
  const args = d.args || {};
  if (d.theme && d.theme.textColor) document.documentElement.style.setProperty("--fg", d.theme.textColor);
  if (!initialized || (scenario && args.scenario && args.scenario.id !== scenario.id)) {
    scenario = args.scenario;
    forces = args.forces || [];
    placed = (args.initial || []).map(p => ({ id: p.id, angle: p.angle, len: p.len || 115 }));
    initialized = true;
    drawScene();
    buildPalette();
  }
  locked = !!args.locked;
  render();
  sendHeight();
});
window.addEventListener("resize", sendHeight);
post("streamlit:componentReady", { apiVersion: 1 });

// ---------- Helpers ----------
function el(tag, attrs, parent) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}
const forceById = id => forces.find(f => f.id === id);
const rad = d => d * Math.PI / 180;
const norm = d => ((d % 360) + 360) % 360;
function symHTML(f) { return f.base + (f.sub ? "<sub>" + f.sub + "</sub>" : ""); }
function svgPoint(e) {
  const pt = fbd.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
  return pt.matrixTransform(fbd.getScreenCTM().inverse());
}
function overFbd(e) {
  const r = fbd.getBoundingClientRect();
  return e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
}
// Direction from the object to the pointer, snapped to 15° steps
function aimFrom(e) {
  const p = svgPoint(e);
  const dist = Math.hypot(p.x, p.y);
  const angle = norm(Math.round(Math.atan2(-p.y, p.x) * 180 / Math.PI / 15) * 15);
  return { dist: dist, angle: angle, len: Math.max(70, Math.min(130, dist)) };
}

// ---------- Drawing the free-body diagram ----------
function drawArrow(parent, angle, len, color, dashed) {
  const ux = Math.cos(rad(angle)), uy = -Math.sin(rad(angle));
  const tx = ux * len, ty = uy * len, hl = 14, hw = 7;
  const bx = tx - ux * hl, by = ty - uy * hl, px = -uy, py = ux;
  const g = el("g", {}, parent);
  el("line", { x1: 0, y1: 0, x2: tx, y2: ty, stroke: "transparent", "stroke-width": 20 }, g); // easy to grab
  el("line", { x1: 0, y1: 0, x2: bx, y2: by, stroke: color, "stroke-width": 3.5,
               "stroke-linecap": "round", "stroke-dasharray": dashed ? "6 5" : "none" }, g);
  el("polygon", { points: `${tx},${ty} ${bx + px * hw},${by + py * hw} ${bx - px * hw},${by - py * hw}`,
                  fill: color }, g);
  return g;
}
function drawLabel(parent, f, angle, len) {
  // sit beside the tip (counterclockwise side) so labels don't land on the axis labels
  const ux = Math.cos(rad(angle)), uy = -Math.sin(rad(angle)), px = uy, py = -ux;
  const t = el("text", { x: ux * (len - 4) + px * 20, y: uy * (len - 4) + py * 20 + 6, "text-anchor": "middle", fill: f.color,
    "font-size": 18, "font-weight": 700, "font-style": "italic",
    "font-family": "'Times New Roman', Times, serif" }, parent);
  const base = el("tspan", {}, t); base.textContent = f.base;
  if (f.sub) { const s = el("tspan", { dy: 5, "font-size": 12 }, t); s.textContent = f.sub; }
}

function render() {
  fbd.innerHTML = "";
  // graph-paper background
  const defs = el("defs", {}, fbd);
  const pat = el("pattern", { id: "grid", width: 20, height: 20, patternUnits: "userSpaceOnUse", x: -170, y: -150 }, defs);
  el("path", { d: "M20,0 L0,0 0,20", fill: "none", style: "stroke: var(--grid)", "stroke-width": 1 }, pat);
  el("rect", { x: -170, y: -150, width: 340, height: 300, fill: "url(#grid)" }, fbd);

  // axes (tilted for inclines)
  const A = scenario.axis_angle || 0, L = 140;
  [[A, "x"], [A + 90, "y"]].forEach(([ang, lab]) => {
    const ux = Math.cos(rad(ang)), uy = -Math.sin(rad(ang)), px = -uy, py = ux;
    el("line", { x1: -ux * L, y1: -uy * L, x2: ux * L, y2: uy * L, stroke: "currentColor",
                 "stroke-opacity": .5, "stroke-width": 1.5 }, fbd);
    const tx = ux * L, ty = uy * L;
    el("polygon", { points: `${tx},${ty} ${tx - ux * 10 + px * 5},${ty - uy * 10 + py * 5} ${tx - ux * 10 - px * 5},${ty - uy * 10 - py * 5}`,
                    fill: "currentColor", "fill-opacity": .5 }, fbd);
    const t = el("text", { x: ux * (L - 14) + px * 14, y: uy * (L - 14) + py * 14 + 5, "text-anchor": "middle",
      fill: "currentColor", "fill-opacity": .75, "font-size": 15, "font-family": "'Times New Roman', Times, serif" }, fbd);
    t.textContent = "+";
    const ts = el("tspan", { "font-style": "italic" }, t); ts.textContent = lab;
  });

  placed.forEach(p => {
    const f = forceById(p.id); if (!f) return;
    const g = drawArrow(fbd, p.angle, p.len, f.color, false);
    drawLabel(g, f, p.angle, p.len);
    if (p.off) g.setAttribute("opacity", .25);
    if (!locked) {
      g.style.cursor = "grab";
      g.addEventListener("pointerdown", ev => startArrowDrag(ev, p));
    }
  });
  if (preview) {
    const f = forceById(preview.id);
    const g = drawArrow(fbd, preview.angle, preview.len, f.color, true);
    drawLabel(g, f, preview.angle, preview.len);
    g.setAttribute("opacity", .7);
  }
  // the object, drawn as a dot
  el("circle", { cx: 0, cy: 0, r: 7, fill: "currentColor" }, fbd);

  if (locked) statusEl.textContent = "Diagram checked and locked.";
  updatePalette();
  updatePlacedList();
}

// ---------- Dragging ----------
function startPaletteDrag(ev, f, chip) {
  if (locked) return;
  ev.preventDefault();
  chip.setPointerCapture(ev.pointerId);
  const ghost = document.createElement("div");
  ghost.className = "chip ghost";
  ghost.innerHTML = chip.innerHTML;
  document.body.appendChild(ghost);

  const move = e => {
    ghost.style.left = e.clientX + "px"; ghost.style.top = e.clientY + "px";
    if (overFbd(e)) {
      const a = aimFrom(e);
      preview = a.dist >= 20 ? { id: f.id, angle: a.angle, len: a.len } : null;
      ghost.style.opacity = .3;
    } else { preview = null; ghost.style.opacity = 1; }
    render();
  };
  const end = e => {
    chip.removeEventListener("pointermove", move);
    chip.removeEventListener("pointerup", end);
    chip.removeEventListener("pointercancel", end);
    ghost.remove(); preview = null;
    if (e.type === "pointerup" && overFbd(e)) {
      const a = aimFrom(e);
      if (a.dist < 20) {
        statusEl.textContent = "Drop the force a little farther from the dot so its direction is clear.";
      } else {
        placed = placed.filter(q => q.id !== f.id);
        placed.push({ id: f.id, angle: a.angle, len: a.len });
        statusEl.textContent = "";
        sendValue();
      }
    }
    render(); sendHeight();
  };
  move(ev);
  chip.addEventListener("pointermove", move);
  chip.addEventListener("pointerup", end);
  chip.addEventListener("pointercancel", end);
}

function startArrowDrag(ev, p) {
  if (locked) return;
  ev.preventDefault(); ev.stopPropagation();
  fbd.setPointerCapture(ev.pointerId);
  const move = e => {
    if (overFbd(e)) {
      const a = aimFrom(e);
      if (a.dist >= 20) { p.angle = a.angle; p.len = a.len; }
      p.off = false; statusEl.textContent = "";
    } else {
      p.off = true; statusEl.textContent = "Let go to remove this force.";
    }
    render();
  };
  const end = e => {
    fbd.removeEventListener("pointermove", move);
    fbd.removeEventListener("pointerup", end);
    fbd.removeEventListener("pointercancel", end);
    if (!overFbd(e)) placed = placed.filter(q => q !== p);
    delete p.off; statusEl.textContent = "";
    render(); sendValue(); sendHeight();
  };
  fbd.addEventListener("pointermove", move);
  fbd.addEventListener("pointerup", end);
  fbd.addEventListener("pointercancel", end);
}

// ---------- Palette and list of placed forces ----------
function buildPalette() {
  const pal = document.getElementById("palette");
  pal.innerHTML = "";
  forces.forEach(f => {
    const chip = document.createElement("div");
    chip.className = "chip"; chip.dataset.id = f.id;
    chip.innerHTML = `<span class="sw" style="background:${f.color}"></span><span class="sym" style="color:${f.color}">${symHTML(f)}</span><span>${f.name}</span>`;
    chip.addEventListener("pointerdown", ev => startPaletteDrag(ev, f, chip));
    pal.appendChild(chip);
  });
}
function updatePalette() {
  document.querySelectorAll("#palette .chip").forEach(chip => {
    chip.classList.toggle("used", placed.some(p => p.id === chip.dataset.id));
    chip.classList.toggle("locked", locked);
  });
  document.getElementById("paletteTitle").textContent = locked
    ? "Your diagram is correct. On to the equations."
    : "Forces you can use (not all of them are real forces)";
}
function updatePlacedList() {
  const box = document.getElementById("placed");
  box.innerHTML = "";
  if (!placed.length) {
    const s = document.createElement("span"); s.style.opacity = .65;
    s.textContent = "No forces on your diagram yet.";
    box.appendChild(s); return;
  }
  const lab = document.createElement("span"); lab.style.opacity = .75; lab.textContent = "On your diagram:";
  box.appendChild(lab);
  placed.forEach(p => {
    const f = forceById(p.id); if (!f) return;
    const tag = document.createElement("span"); tag.className = "tag";
    tag.innerHTML = `<span class="sym" style="color:${f.color}">${symHTML(f)}</span>`;
    if (!locked) {
      const b = document.createElement("button");
      b.title = "Remove " + f.name; b.setAttribute("aria-label", "Remove " + f.name); b.textContent = "×";
      b.onclick = () => { placed = placed.filter(q => q !== p); render(); sendValue(); sendHeight(); };
      tag.appendChild(b);
    }
    box.appendChild(tag);
  });
  if (!locked) {
    const c = document.createElement("button"); c.className = "clear"; c.textContent = "Clear diagram";
    c.onclick = () => { placed = []; render(); sendValue(); sendHeight(); };
    box.appendChild(c);
  }
}

// ---------- Situation sketches ----------
const ST = 'stroke="currentColor" stroke-width="2"';
function hatch(x1, x2, y, up) {
  let s = `<line x1="${x1}" y1="${y}" x2="${x2}" y2="${y}" ${ST}/>`;
  for (let x = x1 + 8; x <= x2; x += 12) {
    const y2 = up ? y - 8 : y + 8;
    s += `<line x1="${x}" y1="${y}" x2="${x - 8}" y2="${y2}" stroke="currentColor" stroke-opacity=".5" stroke-width="1.2"/>`;
  }
  return s;
}
function arrow(x1, y1, x2, y2, color) {
  const ang = Math.atan2(y2 - y1, x2 - x1), hl = 9, hw = 5;
  const bx = x2 - Math.cos(ang) * hl, by = y2 - Math.sin(ang) * hl, px = -Math.sin(ang), py = Math.cos(ang);
  return `<line x1="${x1}" y1="${y1}" x2="${bx}" y2="${by}" stroke="${color}" stroke-width="2.2"/>` +
         `<polygon points="${x2},${y2} ${bx + px * hw},${by + py * hw} ${bx - px * hw},${by - py * hw}" fill="${color}"/>`;
}
function txt(x, y, s, o) {
  o = o || {};
  return `<text x="${x}" y="${y}" text-anchor="${o.anchor || "middle"}" font-size="${o.size || 13}" fill="${o.color || "currentColor"}"` +
         `${o.italic ? ' font-style="italic" font-family="Times New Roman, serif"' : ""}>${s}</text>`;
}
function person(cx, headY) {
  return `<circle cx="${cx}" cy="${headY}" r="11" fill="none" ${ST}/>` +
    `<line x1="${cx}" y1="${headY + 11}" x2="${cx}" y2="${headY + 60}" ${ST}/>` +
    `<line x1="${cx}" y1="${headY + 27}" x2="${cx - 18}" y2="${headY + 47}" ${ST}/>` +
    `<line x1="${cx}" y1="${headY + 27}" x2="${cx + 18}" y2="${headY + 47}" ${ST}/>` +
    `<line x1="${cx}" y1="${headY + 60}" x2="${cx - 14}" y2="${headY + 100}" ${ST}/>` +
    `<line x1="${cx}" y1="${headY + 60}" x2="${cx + 14}" y2="${headY + 100}" ${ST}/>`;
}

const SCENES = {
  table: () =>
    `<rect x="50" y="158" width="220" height="12" rx="2" fill="#b07d4f" fill-opacity=".5" ${ST}/>` +
    `<line x1="70" y1="170" x2="70" y2="222" ${ST}/><line x1="250" y1="170" x2="250" y2="222" ${ST}/>` +
    hatch(30, 290, 222, false) +
    `<rect x="122" y="128" width="76" height="30" rx="3" fill="#2f6fd6" fill-opacity=".35" ${ST}/>` +
    txt(160, 148, "book"),

  hanging: () =>
    hatch(90, 230, 30, true) +
    `<line x1="160" y1="30" x2="160" y2="108" ${ST}/>` +
    `<path d="M128,140 L146,108 L174,108 L192,140 Z" fill="#e8c547" fill-opacity=".55" ${ST}/>` +
    `<circle cx="160" cy="147" r="7" fill="#fff4b0" ${ST}/>` +
    txt(178, 72, "cord", { anchor: "start" }),

  skydiver: () =>
    `<circle cx="160" cy="78" r="11" fill="none" ${ST}/>` +
    `<line x1="160" y1="89" x2="160" y2="140" ${ST}/>` +
    `<line x1="160" y1="100" x2="122" y2="80" ${ST}/><line x1="160" y1="100" x2="198" y2="80" ${ST}/>` +
    `<line x1="160" y1="140" x2="132" y2="172" ${ST}/><line x1="160" y1="140" x2="188" y2="172" ${ST}/>` +
    [110, 140, 180, 210].map(x => `<line x1="${x}" y1="225" x2="${x}" y2="195" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5"/>`).join("") +
    arrow(255, 70, 255, 160, "currentColor") + txt(268, 120, "v", { italic: true, size: 16 }),

  elevator: () =>
    `<rect x="100" y="40" width="120" height="180" rx="4" fill="currentColor" fill-opacity=".05" ${ST}/>` +
    `<line x1="160" y1="0" x2="160" y2="40" ${ST}/>` +
    person(160, 108) +
    arrow(255, 190, 255, 80, "currentColor") + txt(268, 140, "a", { italic: true, size: 16 }) +
    arrow(285, 190, 285, 110, "currentColor") + txt(298, 155, "v", { italic: true, size: 16 }),

  push: () =>
    hatch(30, 290, 190, false) +
    `<rect x="135" y="132" width="64" height="58" rx="3" fill="#c8a26b" fill-opacity=".55" ${ST}/>` +
    txt(167, 166, "box") +
    arrow(62, 161, 131, 161, "#d6336c") + txt(95, 150, "push", { color: "#d6336c" }) +
    arrow(140, 110, 200, 110, "currentColor") + txt(170, 100, "speeding up") +
    txt(160, 220, "rough floor", { italic: true }),

  sled: () =>
    hatch(20, 300, 192, false) +
    `<rect x="105" y="158" width="76" height="16" rx="3" fill="#c0392b" fill-opacity=".4" ${ST}/>` +
    `<path d="M98,186 L186,186 Q198,186 196,174" fill="none" ${ST}/>` +
    `<line x1="118" y1="174" x2="118" y2="186" ${ST}/><line x1="168" y1="174" x2="168" y2="186" ${ST}/>` +
    `<line x1="181" y1="166" x2="262" y2="119.2" ${ST}/>` +
    `<line x1="181" y1="166" x2="250" y2="166" stroke="currentColor" stroke-opacity=".5" stroke-dasharray="4 4"/>` +
    `<path d="M216,166 A35,35 0 0 0 211.3,148.5" fill="none" ${ST}/>` +
    txt(232, 160, "θ", { italic: true, size: 16 }) + txt(240, 112, "rope", { anchor: "end" }) +
    arrow(110, 128, 170, 128, "currentColor") + txt(140, 118, "speeding up"),

  incline: (variant) => {
    let s = `<polygon points="40,200 280,200 40,61.4" fill="currentColor" fill-opacity=".08" ${ST}/>` +
      `<g transform="translate(160,130.7) rotate(30)"><rect x="-24" y="-30" width="48" height="30" rx="3" fill="#c8a26b" fill-opacity=".6" ${ST}/></g>` +
      `<path d="M240,200 A40,40 0 0 1 245.36,180" fill="none" ${ST}/>` +
      txt(229, 195, "θ", { italic: true, size: 16 });
    if (variant !== "frictionless") {
      for (let i = 0; i < 14; i++) {
        const x = 52 + i * 16.5, y = 61.4 + (x - 40) * 0.5774;
        s += `<line x1="${x}" y1="${y}" x2="${x - 4}" y2="${y + 7}" stroke="currentColor" stroke-opacity=".45" stroke-width="1.2"/>`;
      }
    }
    if (variant === "static") s += txt(205, 100, "at rest", { anchor: "start" });
    else s += arrow(165.2, 81.7, 199.8, 101.7, "currentColor") + txt(205, 100, "speeding up", { anchor: "start" });
    return s;
  },
};

function drawScene() {
  const fn = SCENES[scenario.scene];
  sceneSvg.innerHTML = fn ? fn(scenario.variant) : "";
  document.getElementById("caption").textContent = scenario.caption || "";
}
</script>
</body>
</html>
"""

# The drag-and-drop diagram is a small custom component (plain HTML + JavaScript).
# Its HTML is stored at the bottom of this file and written to a temporary folder
# at startup, so the app deploys as a single file with no extra folders to upload.
def _component_dir() -> str:
    folder = Path(tempfile.gettempdir()) / "fbd_builder_component"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "index.html").write_text(COMPONENT_HTML, encoding="utf-8")
    return str(folder)

ANGLE_TOLERANCE = 20  # degrees a student's arrow may be off and still count as correct

# ---------------------------------------------------------------------------
# Forces in the palette (real forces plus common misconceptions)
# ---------------------------------------------------------------------------
FORCES = {
    "weight": dict(
        name="Gravitational force (weight)", base="F", sub="g", color="#2f6fd6", real=True,
        hint="Gravity always pulls straight down, toward the center of Earth.",
    ),
    "normal": dict(
        name="Normal force", base="F", sub="N", color="#23925a", real=True,
        hint="A normal force is perpendicular to the surface, pushing away from it.",
    ),
    "applied": dict(
        name="Applied force (push)", base="F", sub="A", color="#d6336c", real=True,
        hint="An applied force points in the direction of the push.",
    ),
    "motion": dict(
        name="Force of motion", base="F", sub="motion", color="#a0522d", real=False,
        hint="There is no \"force of motion.\" A moving object keeps moving because of inertia "
             "(Newton's first law); nothing has to push it along.",
    ),
    "tension": dict(
        name="Tension", base="T", sub="", color="#7b4fd6", real=True,
        hint="Tension pulls along the rope or cord, away from the object.",
    ),
    "kfriction": dict(
        name="Kinetic friction", base="f", sub="k", color="#e0702b", real=True,
        hint="Kinetic friction acts along the surface, opposite the direction the object slides.",
    ),
    "net": dict(
        name="Net force", base="F", sub="net", color="#0b7a75", real=False,
        hint="The net force isn't a separate force on the object. It's the vector sum of the forces "
             "already in your diagram, so drawing it counts those forces twice.",
    ),
    "sfriction": dict(
        name="Static friction", base="f", sub="s", color="#b8860b", real=True,
        hint="Static friction acts along the surface, opposite the way the object would slide "
             "if the surface were frictionless.",
    ),
    "air": dict(
        name="Air resistance", base="F", sub="air", color="#1597b0", real=True,
        hint="Air resistance points opposite the object's velocity.",
    ),
    "inertia": dict(
        name="Force of inertia", base="F", sub="inertia", color="#5b6b2f", real=False,
        hint="Inertia is a property of the object (its resistance to changes in motion), "
             "not a force that something exerts on it.",
    ),
}

FORCE_PAYLOAD = [
    dict(id=fid, name=f["name"], base=f["base"], sub=f["sub"], color=f["color"])
    for fid, f in FORCES.items()
]

# ---------------------------------------------------------------------------
# Situations
# Angles are in degrees, measured counterclockwise from the +x direction (pointing right).
# Equations are written as (left side − right side) of ΣF = ma, using the symbols below.
# The inclines and the sled rope are drawn at θ = 30°.
# ---------------------------------------------------------------------------
STANDARD_AXES = "+x points to the right and +y points straight up."
INCLINE_AXES = ("Tilt your axes: +x points **down the incline**, and +y points perpendicular "
                "to the incline, away from the surface.")

SCENARIOS = [
    dict(id="book", title="Book on a table", scene="table", variant="",
         description="A textbook sits at rest on a level table.",
         caption="At rest on a level table", axis_angle=0, axes=STANDARD_AXES,
         answer={"weight": 270, "normal": 90},
         eq_x="0", eq_y="F_N - F_g", ax=False, ay=False,
         tex_x=r"\Sigma F_x = 0 \quad (\text{no forces along } x)",
         tex_y=r"\Sigma F_y = F_N - F_g = 0"),
    dict(id="lamp", title="Hanging lamp", scene="hanging", variant="",
         description="A lamp hangs at rest from a single vertical cord attached to the ceiling.",
         caption="Hanging at rest from one cord", axis_angle=0, axes=STANDARD_AXES,
         answer={"tension": 90, "weight": 270},
         eq_x="0", eq_y="T - F_g", ax=False, ay=False,
         tex_x=r"\Sigma F_x = 0 \quad (\text{no forces along } x)",
         tex_y=r"\Sigma F_y = T - F_g = 0"),
    dict(id="skydiver", title="Skydiver at terminal velocity", scene="skydiver", variant="",
         description="A skydiver falls straight down at a constant (terminal) velocity.",
         caption="Falling at constant velocity", axis_angle=0, axes=STANDARD_AXES,
         answer={"weight": 270, "air": 90},
         eq_x="0", eq_y="F_air - F_g", ax=False, ay=False,
         tex_x=r"\Sigma F_x = 0 \quad (\text{no forces along } x)",
         tex_y=r"\Sigma F_y = F_{air} - F_g = 0"),
    dict(id="elevator", title="Person in an elevator", scene="elevator", variant="",
         description="A person stands on the floor of an elevator that is moving upward "
                     "and speeding up.",
         caption="Moving upward and speeding up", axis_angle=0, axes=STANDARD_AXES,
         answer={"normal": 90, "weight": 270},
         eq_x="0", eq_y="F_N - F_g - m*a", ax=False, ay=True,
         tex_x=r"\Sigma F_x = 0 \quad (\text{no forces along } x)",
         tex_y=r"\Sigma F_y = F_N - F_g = ma"),
    dict(id="push", title="Pushing a box", scene="push", variant="",
         description="A student pushes horizontally on a box. The box slides to the right across "
                     "a rough floor and speeds up.",
         caption="Sliding right on a rough floor, speeding up", axis_angle=0, axes=STANDARD_AXES,
         answer={"applied": 0, "kfriction": 180, "weight": 270, "normal": 90},
         eq_x="F_A - f_k - m*a", eq_y="F_N - F_g", ax=True, ay=False,
         tex_x=r"\Sigma F_x = F_A - f_k = ma",
         tex_y=r"\Sigma F_y = F_N - F_g = 0"),
    dict(id="sled", title="Pulling a sled", scene="sled", variant="",
         description="A child pulls a sled across level snow with a rope held at angle θ above "
                     "the horizontal. The snow exerts friction on the sliding sled, and the sled "
                     "speeds up.",
         caption="Pulled at angle θ, speeding up", axis_angle=0, axes=STANDARD_AXES,
         answer={"tension": 30, "kfriction": 180, "weight": 270, "normal": 90},
         eq_x="T*cos(theta) - f_k - m*a", eq_y="F_N + T*sin(theta) - F_g", ax=True, ay=False,
         tex_x=r"\Sigma F_x = T\cos\theta - f_k = ma",
         tex_y=r"\Sigma F_y = F_N + T\sin\theta - F_g = 0"),
    dict(id="ramp_smooth", title="Block on a frictionless ramp", scene="incline",
         variant="frictionless",
         description="A block slides down a frictionless ramp inclined at angle θ.",
         caption="Frictionless ramp, sliding down", axis_angle=-30, axes=INCLINE_AXES,
         answer={"weight": 270, "normal": 60},
         eq_x="F_g*sin(theta) - m*a", eq_y="F_N - F_g*cos(theta)", ax=True, ay=False,
         tex_x=r"\Sigma F_x = F_g\sin\theta = ma",
         tex_y=r"\Sigma F_y = F_N - F_g\cos\theta = 0"),
    dict(id="ramp_static", title="Box resting on a rough ramp", scene="incline", variant="static",
         description="A box sits at rest on a rough ramp inclined at angle θ. It does not slide.",
         caption="Rough ramp, box at rest", axis_angle=-30, axes=INCLINE_AXES,
         answer={"weight": 270, "normal": 60, "sfriction": 150},
         eq_x="F_g*sin(theta) - f_s", eq_y="F_N - F_g*cos(theta)", ax=False, ay=False,
         tex_x=r"\Sigma F_x = F_g\sin\theta - f_s = 0",
         tex_y=r"\Sigma F_y = F_N - F_g\cos\theta = 0"),
    dict(id="ramp_kinetic", title="Block sliding down a rough ramp", scene="incline",
         variant="kinetic",
         description="A block slides down a rough ramp inclined at angle θ, speeding up as it goes.",
         caption="Rough ramp, sliding down and speeding up", axis_angle=-30, axes=INCLINE_AXES,
         answer={"weight": 270, "normal": 60, "kfriction": 150},
         eq_x="F_g*sin(theta) - f_k - m*a", eq_y="F_N - F_g*cos(theta)", ax=True, ay=False,
         tex_x=r"\Sigma F_x = F_g\sin\theta - f_k = ma",
         tex_y=r"\Sigma F_y = F_N - F_g\cos\theta = 0"),
]

# ---------------------------------------------------------------------------
# Symbols students can type, with common alternate spellings
# ---------------------------------------------------------------------------
m, g, a, theta, mu_k = sp.symbols("m g a theta mu_k", positive=True)
a_x, a_y = sp.symbols("a_x a_y")
F_g, F_N, T, f_k, f_s, F_A, F_air = sp.symbols("F_g F_N T f_k f_s F_A F_air", positive=True)
FORCE_SYMBOLS = {F_g, F_N, T, f_k, f_s, F_A, F_air}

NAMES = {
    "m": m, "g": g, "a": a, "theta": theta,
    "a_x": a_x, "ax": a_x, "a_y": a_y, "ay": a_y,
    "mu_k": mu_k, "muk": mu_k, "mu": mu_k,
    "F_g": F_g, "Fg": F_g, "F_w": F_g, "Fw": F_g, "W": F_g, "mg": m * g, "ma": m * a,
    "F_N": F_N, "FN": F_N, "F_n": F_N, "Fn": F_N, "N": F_N,
    "T": T, "F_T": T, "FT": T, "F_t": T, "Ft": T,
    "f_k": f_k, "fk": f_k, "F_k": f_k, "Fk": f_k,
    "f_s": f_s, "fs": f_s, "F_s": f_s, "Fs": f_s,
    "F_A": F_A, "FA": F_A, "F_a": F_A, "Fa": F_A, "F_app": F_A, "Fapp": F_A,
    "F_air": F_air, "Fair": F_air, "F_D": F_air, "FD": F_air,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "sqrt": sp.sqrt,
}

# Substitutions that let equivalent answers match (F_g = mg, f_k = μ_k F_N)
BASE_SUBS = {F_g: m * g, f_k: mu_k * F_N}

TRANSFORMS = standard_transformations + (implicit_multiplication, implicit_application, convert_xor)
ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9_+\-*/().=^\s]*$")
NET_FORCE_WORD = re.compile(r"^(Sigma|sum)?_?F_?(net)?_?[xy]$", re.IGNORECASE)


def normalize(text: str) -> str:
    replacements = {
        "θ": "theta", "Θ": "theta", "μ": "mu", "·": "*", "×": "*",
        "−": "-", "–": "-", "Σ": "", "∑": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()


def parse_side(side_text: str):
    return parse_expr(side_text, local_dict=NAMES, transformations=TRANSFORMS)


def parse_equation(text: str):
    """Return ((lhs, rhs), None) or (None, error message)."""
    s = normalize(text)
    if not s:
        return None, "Type an equation."
    if not ALLOWED_CHARS.match(s):
        return None, "Use only letters, numbers, + − * / ^ ( ) and one equals sign."
    if "=" not in s:
        return None, "Your equation needs an equals sign, for example: forces = m*a"
    parts = s.split("=")
    lhs_text, rhs_text = parts[-2], parts[-1]  # allows "ΣF_x = ... = m*a"
    if not lhs_text.strip() or not rhs_text.strip():
        return None, "Put something on both sides of the equals sign."
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", lhs_text + " " + rhs_text))
    if any(NET_FORCE_WORD.match(w) for w in words):
        return None, "Leave out the ΣF part. Type just the forces, then = m*a (or = 0)."
    unknown = sorted(w for w in words if w not in NAMES)
    if unknown:
        return None, (f"I don't recognize: {', '.join(unknown)}. Check the symbol list, and use * "
                      "between symbols (for example, F_g*sin(theta)).")
    try:
        return (parse_side(lhs_text), parse_side(rhs_text)), None
    except Exception:
        return None, "I couldn't read that equation. Check your parentheses and operators."


def equation_latex(lhs, rhs) -> str:
    return f"{sp.latex(lhs)} = {sp.latex(rhs)}"


def forces_in(expr):
    """Which forces appear in an expression (treating mg as F_g and μ_k F_N as f_k)."""
    e = expr.subs(mu_k * F_N, f_k).subs(m * g, F_g)
    return e.free_symbols & FORCE_SYMBOLS


def check_equation(text: str, sc: dict, axis: str) -> dict:
    parsed, error = parse_equation(text)
    if error:
        return dict(ok=False, msg=error)
    lhs, rhs = parsed

    accelerates = sc["ax"] if axis == "x" else sc["ay"]
    own, other = (a_x, a_y) if axis == "x" else (a_y, a_x)
    student_raw = lhs - rhs
    if student_raw.has(other):
        return dict(ok=False, msg=f"You used ${sp.latex(other)}$ in the {axis}-axis equation. "
                                  f"Use $a$ (or ${sp.latex(own)}$) for acceleration along this axis.")

    accel_subs = {own: a} if accelerates else {own: 0, a: 0}
    expected_raw = parse_side(sc[f"eq_{axis}"])
    student = sp.simplify(student_raw.subs(accel_subs).subs(BASE_SUBS))
    expected = sp.simplify(expected_raw.subs(BASE_SUBS))

    if expected == 0:
        if student == 0:
            return dict(ok=True, msg="Correct. No forces act along this axis, so the equation is 0 = 0.")
        return dict(ok=False, msg=f"None of the forces in your diagram has a component along the "
                                  f"{axis}-axis, and the object doesn't accelerate that way, "
                                  f"so this equation is just 0 = 0.")
    if student == 0:
        return dict(ok=False, msg="Your equation works out to 0 = 0, but forces in your diagram "
                                  f"do have components along the {axis}-axis.")

    ratio = sp.simplify(student / expected)
    if ratio.is_number and ratio != 0:
        note = ""
        if not accelerates and student_raw.subs(own, a).has(a):
            note = (f" Since the object doesn't accelerate along the {axis}-axis, "
                    f"$a_{axis} = 0$ and the right side is 0.")
        return dict(ok=True, msg="Correct!" + note)

    # --- Hints, most specific first ---
    expected_forces, student_forces = forces_in(expected_raw), forces_in(student_raw)
    missing = expected_forces - student_forces
    extra = student_forces - expected_forces
    if missing:
        names = ", ".join(f"${sp.latex(s)}$" for s in sorted(missing, key=str))
        return dict(ok=False, msg=f"Your equation is missing {names}. Every force with a component "
                                  f"along the {axis}-axis belongs in the sum.")
    if extra:
        names = ", ".join(f"${sp.latex(s)}$" for s in sorted(extra, key=str))
        return dict(ok=False, msg=f"Your equation includes {names}, which has no component along "
                                  f"the {axis}-axis.")
    for trig in (sp.sin(theta), sp.cos(theta)):
        if expected.has(trig) != student.has(trig):
            return dict(ok=False, msg="Check your trig. For each angled force, which component "
                                      "uses sin θ and which uses cos θ?")
    for term in sp.Add.make_args(sp.expand(expected_raw)):
        flipped = expected - 2 * term.subs(BASE_SUBS)
        r = sp.simplify(student / flipped) if flipped != 0 else None
        if r is not None and r.is_number and r != 0:
            shown = -term if term.could_extract_minus_sign() else term
            return dict(ok=False, msg=f"Check the sign of the ${sp.latex(shown)}$ term. "
                                      f"Components pointing in the −{axis} direction are negative.")
    if accelerates and not student.has(a):
        return dict(ok=False, msg="The object accelerates along this axis, so the right side "
                                  "should be $ma$, not 0.")
    return dict(ok=False, msg=f"Not quite. For each force in your diagram, ask whether it has a "
                              f"component along the {axis}-axis, and whether that component is "
                              f"positive or negative.")


# ---------------------------------------------------------------------------
# Free-body diagram checking
# ---------------------------------------------------------------------------
def angle_diff(a1: float, a2: float) -> float:
    return abs((a1 - a2 + 180) % 360 - 180)


def extra_force_hint(fid: str, sc: dict) -> str:
    answer, name = sc["answer"], FORCES[fid]["name"]
    if fid == "kfriction" and "sfriction" in answer:
        return "**Kinetic friction:** the object isn't sliding, so the friction here is static, not kinetic."
    if fid == "sfriction" and "kfriction" in answer:
        return "**Static friction:** the object is sliding across the surface, so the friction is kinetic, not static."
    if fid in ("kfriction", "sfriction"):
        if sc["variant"] == "frictionless":
            return f"**{name}:** this ramp is frictionless."
        return f"**{name}:** nothing is sliding, or trying to slide, along a surface here, so there's no friction."
    if fid == "air":
        return "**Air resistance:** we're ignoring air resistance in this situation."
    if fid == "normal":
        return "**Normal force:** no surface is pressing against the object here, so there's no normal force."
    if fid == "tension":
        return "**Tension:** no rope, string, or cord is pulling on the object here."
    if fid == "applied":
        return ("**Applied force:** nothing is pushing on the object right now. A push that started "
                "its motion earlier isn't still acting.")
    return f"**{name}:** is that force really acting on the object here?"


def check_fbd(placed: list, sc: dict):
    answer = sc["answer"]
    if not placed:
        return False, [("info", "Drag forces onto the diagram first.")]
    messages = []
    ids = [p["id"] for p in placed]
    for fid in ids:
        f = FORCES[fid]
        if not f["real"]:
            messages.append(("error", f"**{f['name']}:** {f['hint']}"))
        elif fid not in answer:
            messages.append(("error", extra_force_hint(fid, sc)))
    for p in placed:
        if p["id"] in answer and angle_diff(p["angle"], answer[p["id"]]) > ANGLE_TOLERANCE:
            f = FORCES[p["id"]]
            messages.append(("warning", f"**{f['name']}** belongs in the diagram, but check its "
                                        f"direction. {f['hint']}"))
    missing = [fid for fid in answer if fid not in ids]
    if missing:
        n = len(missing)
        messages.append(("info", f"You're missing {n} force{'s' if n > 1 else ''}. Start with gravity, "
                                 "then look at everything touching the object: surfaces, ropes, "
                                 "hands, air."))
    return not messages, messages


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("scenario_idx", 0)
ss.setdefault("completed", set())


def reset_scenario(sid: str):
    ss[f"ver_{sid}"] = ss.get(f"ver_{sid}", 0) + 1
    for key in (f"stage_{sid}", f"placed_{sid}", f"fbdmsg_{sid}", f"eqres_{sid}",
                f"eqx_{sid}", f"eqy_{sid}", f"celebrated_{sid}"):
        ss.pop(key, None)
    ss["completed"].discard(sid)


def go_to(idx: int):
    ss["scenario_idx"] = idx


fbd_builder = components.declare_component("fbd_builder", path=_component_dir())

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Situations")
    completed = ss["completed"]
    done_count = len(completed)
    st.progress(done_count / len(SCENARIOS), text=f"{done_count} of {len(SCENARIOS)} complete")
    idx = st.radio(
        "Choose a situation",
        options=range(len(SCENARIOS)),
        format_func=lambda i: ("✅ " if SCENARIOS[i]["id"] in completed else "")
        + f"{i + 1}. {SCENARIOS[i]['title']}",
        key="scenario_idx",
        label_visibility="collapsed",
    )
    sc = SCENARIOS[idx]
    sid = sc["id"]
    st.divider()
    st.button("Start this situation over", on_click=reset_scenario, args=(sid,),
              use_container_width=True)

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
stage = ss.setdefault(f"stage_{sid}", "fbd")
version = ss.get(f"ver_{sid}", 0)

st.title("Free-Body Diagram Builder")
st.subheader(f"{idx + 1}. {sc['title']}")
st.markdown(sc["description"])
st.markdown(f"**Axes:** {sc['axes']}")

st.markdown("#### Step 1: Build the free-body diagram")
st.caption("Drag every force acting **on the object** onto the diagram and drop it in the direction "
           "the force points. Drag an arrow to re-aim it, or drag it off the diagram to remove it.")

value = fbd_builder(
    scenario=dict(id=sid, scene=sc["scene"], variant=sc["variant"],
                  caption=sc["caption"], axis_angle=sc["axis_angle"]),
    forces=FORCE_PAYLOAD,
    initial=ss.get(f"placed_{sid}", []),
    locked=(stage != "fbd"),
    key=f"fbd_{sid}_{version}",
    default=None,
)
if value is not None and stage == "fbd":
    ss[f"placed_{sid}"] = value
placed = ss.get(f"placed_{sid}", [])

if stage == "fbd":
    if st.button("Check my diagram", type="primary", key=f"checkfbd_{sid}_{version}"):
        ok, msgs = check_fbd(placed, sc)
        ss[f"fbdmsg_{sid}"] = msgs
        if ok:
            ss[f"stage_{sid}"] = "eq"
            st.rerun()
    for kind, msg in ss.get(f"fbdmsg_{sid}", []):
        getattr(st, kind)(msg)
else:
    st.success("Your free-body diagram is correct.")

# ---------------------------------------------------------------------------
# Step 2: Newton's second law equations
# ---------------------------------------------------------------------------
if stage in ("eq", "done"):
    st.markdown("#### Step 2: Write Newton's second law for each axis")
    st.markdown(r"Use the forces in your diagram to write $\Sigma F_x = ma_x$ and "
                r"$\Sigma F_y = ma_y$. Type the forces on the left and $m\,a$ (or 0) on the right, "
                "for example `forces = m*a`. Use the axis directions given above.")

    with st.expander("Symbols you can type"):
        st.markdown(
            "| Quantity | Type | Shows as |\n|---|---|---|\n"
            "| Weight | `F_g` or `m*g` | $F_g$ or $mg$ |\n"
            "| Normal force | `F_N` | $F_N$ |\n"
            "| Tension | `T` | $T$ |\n"
            "| Kinetic friction | `f_k` or `mu_k*F_N` | $f_k$ or $\\mu_k F_N$ |\n"
            "| Static friction | `f_s` | $f_s$ |\n"
            "| Applied force | `F_A` | $F_A$ |\n"
            "| Air resistance | `F_air` | $F_{air}$ |\n"
            "| Mass, acceleration | `m`, `a` | $m$, $a$ |\n"
            "| Angle | `theta` or `θ` | $\\theta$ |\n"
            "| Components | `sin(theta)`, `cos(theta)` | $\\sin\\theta$, $\\cos\\theta$ |\n"
        )

    results = ss.setdefault(f"eqres_{sid}", {})
    cols = st.columns(2)
    for col, axis in zip(cols, ("x", "y")):
        with col:
            st.markdown(f"**{axis}-axis:** $\\Sigma F_{axis} = ma_{axis}$")
            text = st.text_input(f"{axis}-axis equation", key=f"eq{axis}_{sid}",
                                 placeholder="forces = m*a", disabled=(stage == "done"),
                                 label_visibility="collapsed")
            if text and stage == "eq":
                parsed, err = parse_equation(text)
                if parsed:
                    st.caption("Here's how I read your equation:")
                    st.latex(equation_latex(*parsed))

    if stage == "eq":
        if st.button("Check my equations", type="primary", key=f"checkeq_{sid}"):
            for axis in ("x", "y"):
                results[axis] = check_equation(ss.get(f"eq{axis}_{sid}", ""), sc, axis)
            if all(results.get(ax_, {}).get("ok") for ax_ in ("x", "y")):
                ss[f"stage_{sid}"] = "done"
                ss["completed"].add(sid)
                st.rerun()
        if results:
            cols = st.columns(2)
            for col, axis in zip(cols, ("x", "y")):
                res = results.get(axis)
                if res:
                    with col:
                        (st.success if res["ok"] else st.error)(f"**{axis}-axis:** {res['msg']}")

    if stage == "done":
        if not ss.get(f"celebrated_{sid}"):
            st.balloons()
            ss[f"celebrated_{sid}"] = True
        st.success("Both equations are correct. Here they are written out:")
        st.latex(sc["tex_x"])
        st.latex(sc["tex_y"])
        for axis in ("x", "y"):
            note = results.get(axis, {}).get("msg", "")
            if note.startswith("Correct!") and len(note) > len("Correct!"):
                st.caption(note[len("Correct! "):])
        if idx + 1 < len(SCENARIOS):
            st.button("Next situation", type="primary", on_click=go_to, args=(idx + 1,))
        elif len(ss["completed"]) == len(SCENARIOS):
            st.info("You've finished every situation. Nice work!")
