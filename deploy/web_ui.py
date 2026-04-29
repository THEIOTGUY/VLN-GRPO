#!/usr/bin/env python3
"""
ActiveVLN Web UI — live webcam feed, real-time log, voice + text instruction input.

Run alongside go1_nav.py (which writes frames to /tmp/vln_latest_frame.jpg
and reads instructions from /tmp/vln_instruction.txt).

Usage:
    python deploy/web_ui.py --port 5000 --log /tmp/activevln.log
Then open  http://<orin-ip>:5000  in a browser.
"""

import argparse
import json
import os
import re
import time
import threading

import cv2
import numpy as np
from flask import Flask, Response, abort, render_template_string, request, jsonify

app = Flask(__name__)

_log_path      = "/tmp/activevln.log"
_shared_frame  = "/tmp/vln_latest_frame.jpg"
_instr_file    = "/tmp/vln_instruction.txt"
_pause_flag    = "/tmp/vln_paused.flag"
_restart_flag  = "/tmp/vln_restart.flag"
_estop_flag    = "/tmp/vln_estop.flag"
_hist_file     = "/tmp/vln_history.json"
_obs_dir       = "/tmp/vln_obs"
_memory_file   = "/tmp/vln_memory.json"   # written by go1_nav spatial memory
_obstacle_file = "/tmp/vln_obstacle.flag" # present when obstacle replan fired

_frame_lock   = threading.Lock()
_latest_frame: bytes | None = None

_ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
_NOISE = (
    re.compile(r'^\s*\d+\.\d+\.\d+\.\d+ - - \['),
    re.compile(r'Press CTRL\+C to quit'),
    re.compile(r'Serving Flask app'),
    re.compile(r'Debug mode'),
    re.compile(r'Running on'),
    re.compile(r'This is a development server'),
    re.compile(r'^\s*warn\(', re.IGNORECASE),
    re.compile(r'UserWarning'),
)

def _is_noise(line: str) -> bool:
    return any(p.search(line) for p in _NOISE)


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ActiveVLN — Go1</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #0a0d12; --panel: #0d1117; --panel2: #141820;
  --border: #2a2f3a; --border2: #30363d; --muted: #8b949e; --text: #e6edf3;
  --blue: #58a6ff; --green: #56d364; --orange: #f0883e;
  --yellow: #e3b341; --red: #f85149; --purple: #bc8cff;
}
html,body { height:100%; }
body { background:var(--bg); color:var(--text);
       font-family:-apple-system,'Segoe UI',system-ui,sans-serif;
       display:flex; flex-direction:column; height:100vh; overflow:hidden; font-size:14px; }

/* header */
header { background:linear-gradient(180deg,#1a1f26 0%,#141820 100%);
         padding:12px 22px; border-bottom:1px solid var(--border);
         display:flex; align-items:center; gap:14px; flex-shrink:0; }
.brand { display:flex; align-items:center; gap:10px; }
.logo  { width:28px; height:28px; border-radius:8px;
         background:linear-gradient(135deg,#58a6ff 0%,#bc8cff 100%);
         display:flex; align-items:center; justify-content:center;
         font-weight:700; font-size:.85rem; color:#0a0d12; }
header h1 { font-size:1rem; font-weight:600; }
.spacer { flex:1; }
.badge { font-size:.68rem; padding:4px 10px; border-radius:99px; font-weight:600;
         text-transform:uppercase; letter-spacing:.6px;
         display:inline-flex; align-items:center; gap:6px; transition:background .2s; }
.badge::before { content:''; width:7px; height:7px; border-radius:50%; background:currentColor; }
.badge.live      { background:rgba(35,134,54,.18);  color:#4ac26b; }
.badge.live::before { animation:pulse 1.4s ease-out infinite; }
.badge.paused    { background:rgba(227,179,65,.18); color:#e3b341; }
.badge.offline   { background:rgba(182,35,36,.18);  color:#f85149; }
.badge.connecting{ background:rgba(139,148,158,.18);color:#8b949e; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }

/* stats */
.stats { display:grid; grid-template-columns:repeat(6,1fr);
         gap:1px; background:var(--border); flex-shrink:0; }
.stat  { background:var(--panel2); padding:10px 18px; display:flex; flex-direction:column; gap:2px; }
.stat .k { font-size:.66rem; color:var(--muted); text-transform:uppercase; letter-spacing:.6px; font-weight:600; }
.stat .v { font-size:1.02rem; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.stat.instr    .v { color:var(--orange); }
.stat.step     .v { color:var(--blue); }
.stat.action   .v { color:var(--green); font-family:'SF Mono',Menlo,Consolas,monospace; }
.stat.lat      .v { color:var(--purple); }
.stat.memory   .v { color:#58d1c9; font-family:'SF Mono',Menlo,Consolas,monospace; font-size:.8rem; }
.stat.obstacle .v { color:var(--muted); }
.stat.obstacle.alert { background:rgba(248,81,73,.12); }
.stat.obstacle.alert .v { color:var(--red); animation:pulse 1s ease-in-out infinite; }
@media (max-width:1200px) {
  .stats { grid-template-columns:repeat(3,1fr); }
}

/* ctrl bar */
.ctrl { background:var(--panel2); border-top:1px solid var(--border);
        border-bottom:1px solid var(--border);
        padding:10px 20px; display:flex; gap:10px; align-items:center; flex-shrink:0; }
.ctrl .lbl { font-size:.7rem; color:var(--muted); text-transform:uppercase;
             letter-spacing:.6px; font-weight:600; white-space:nowrap; }
#instr-input { flex:1; background:var(--panel); border:1px solid var(--border2);
               color:var(--text); padding:9px 13px; border-radius:7px;
               font-size:.92rem; font-family:inherit; transition:all .15s; }
#instr-input::placeholder { color:#586069; }
#instr-input:focus { outline:none; border-color:var(--blue);
                     box-shadow:0 0 0 3px rgba(88,166,255,.18); }
button { border:none; padding:9px 16px; border-radius:7px; cursor:pointer;
         font-size:.85rem; font-weight:600; transition:all .15s; font-family:inherit;
         display:inline-flex; align-items:center; gap:6px; }
button:hover:not(:disabled)   { transform:translateY(-1px); }
button:active:not(:disabled)  { transform:translateY(0); }
button:disabled { opacity:.45; cursor:not-allowed; }
.btn-set     { background:#238636; color:#fff; }
.btn-set:hover:not(:disabled)     { background:#2ea043; }
.btn-voice   { background:#8250df; color:#fff; }
.btn-voice:hover:not(:disabled)   { background:#9a6dea; }
.btn-voice.listening { background:#f85149; animation:pulse-btn 1.5s ease-in-out infinite; }
@keyframes pulse-btn { 0%,100%{opacity:1} 50%{opacity:.7} }
.btn-pause   { background:#30363d; color:var(--text); }
.btn-pause:hover:not(:disabled)   { background:#3d444d; }
.btn-estop   { background:#f85149; color:#fff; }
.btn-estop:hover:not(:disabled)   { background:#ff6b63; }
.btn-restart { background:#b62324; color:#fff; }
.btn-restart:hover:not(:disabled) { background:#da3633; }

/* voice panel */
.voice-panel { background:var(--panel2); border-bottom:1px solid var(--border);
               padding:10px 20px; display:none; flex-direction:column; gap:8px; flex-shrink:0; }
.voice-panel.active { display:flex; }
.voice-row { display:flex; align-items:center; gap:10px; }
.voice-dot  { width:12px; height:12px; border-radius:50%; background:#30363d; transition:all .3s; }
.voice-dot.listening  { background:#f85149; animation:pulse 1.5s ease-in-out infinite; }
.voice-dot.processing { background:#e3b341; }
.voice-lbl  { font-size:.75rem; color:var(--muted); font-weight:600;
              text-transform:uppercase; letter-spacing:.5px; }
.voice-text { font-size:.9rem; color:var(--text); padding:8px 12px;
              background:var(--panel); border-radius:6px; border:1px solid var(--border2);
              min-height:40px; }
.voice-text:empty::before { content:'Your speech will appear here…';
                             color:#586069; font-style:italic; }
.pause-banner { display:none; margin-left:auto;
                background:rgba(227,179,65,.15); border:1px solid #e3b341; color:var(--yellow);
                padding:4px 10px; border-radius:6px; font-size:.72rem; font-weight:600;
                align-items:center; gap:6px; }
body.is-paused .pause-banner { display:inline-flex; }

/* layout */
.layout { display:grid; grid-template-columns:1fr 460px;
          flex:1; min-height:0; overflow:hidden; gap:1px; background:var(--border); }

/* cam */
.cam-panel { background:#000; display:flex; align-items:stretch;
             justify-content:center; position:relative; overflow:hidden; }
.cam-wrap  { flex:1; display:flex; align-items:center; justify-content:center;
             padding:16px; position:relative; overflow:hidden; }
.cam-panel img { max-width:100%; max-height:100%; border-radius:10px;
                 box-shadow:0 4px 24px rgba(0,0,0,.55); border:1px solid #1a1f26; }
.cam-overlay { position:absolute; top:24px; left:24px; right:24px;
               display:flex; justify-content:space-between; pointer-events:none;
               font-family:'SF Mono',Menlo,Consolas,monospace; font-size:.72rem; }
.cam-tag { background:rgba(10,13,18,.72); color:#c9d1d9; padding:4px 10px;
           border-radius:5px; backdrop-filter:blur(6px);
           border:1px solid rgba(255,255,255,.08); }

/* log */
.log-panel  { background:var(--panel); display:flex; flex-direction:column; overflow:hidden; }
.log-header { padding:10px 16px; background:var(--panel2); border-bottom:1px solid var(--border);
              display:flex; align-items:center; gap:10px; flex-shrink:0; }
.log-title  { font-size:.7rem; color:var(--muted); font-weight:600;
              text-transform:uppercase; letter-spacing:.6px; flex:1; }
.flt-btn { padding:4px 10px; border-radius:5px; font-size:.68rem; background:transparent;
           border:1px solid var(--border2); color:var(--muted); cursor:pointer;
           font-weight:600; text-transform:uppercase; letter-spacing:.5px; transition:all .12s; }
.flt-btn:hover { color:var(--text); border-color:#484f58; }
.flt-btn.on { background:var(--blue); border-color:var(--blue); color:#0a0d12; }
#log { flex:1; overflow-y:auto; padding:10px 16px;
       font-family:'SF Mono',Menlo,Consolas,monospace; font-size:.76rem;
       line-height:1.6; scroll-behavior:smooth; }
#log::-webkit-scrollbar { width:7px; }
#log::-webkit-scrollbar-thumb { background:#30363d; border-radius:3px; }

.line { padding:2px 0; word-break:break-word; display:flex; gap:8px; }
.line .ts  { color:#484f58; flex-shrink:0; font-size:.68rem; padding-top:1px; min-width:44px; }
.line .txt { flex:1; }
.line.step   .txt { color:var(--blue);   font-weight:600; }
.line.action .txt { color:var(--green); }
.line.instr  .txt { color:var(--orange); font-weight:600; }
.line.warn   .txt { color:var(--yellow); }
.line.err    .txt { color:var(--red);    font-weight:500; }
.line.info   .txt { color:#6e7681; }
.line.system { margin:4px 0; padding:5px 10px; border-radius:5px;
               background:rgba(188,140,255,.08); border-left:3px solid var(--purple); }
.line.system .txt { color:var(--purple); font-weight:600; }
body.filter-on .line.info { display:none; }

.scroll-btn { position:absolute; right:16px; bottom:14px; background:var(--blue);
              color:#0a0d12; border:none; padding:6px 11px; border-radius:20px;
              font-size:.7rem; font-weight:700; cursor:pointer; display:none;
              box-shadow:0 3px 10px rgba(0,0,0,.4); }
.scroll-btn.show { display:block; }

@media (max-width:980px) {
  .layout { grid-template-columns:1fr; grid-template-rows:1fr 1fr; }
  .stats  { grid-template-columns:repeat(2,1fr); }
}
@media (max-width:600px) {
  .stats { grid-template-columns:repeat(2,1fr); }
}

/* observation history strip */
.history-panel { background:var(--panel); border-top:1px solid var(--border);
                 flex-shrink:0; display:flex; flex-direction:column; height:192px; }
.history-header { padding:7px 16px; background:var(--panel2);
                  border-bottom:1px solid var(--border);
                  display:flex; align-items:center; gap:10px; flex-shrink:0; }
.hist-count { font-size:.68rem; color:var(--muted); margin-left:auto; }
.history-strip { display:flex; gap:10px; overflow-x:auto; overflow-y:hidden;
                 padding:10px 14px; align-items:flex-start; flex:1; }
.history-strip::-webkit-scrollbar { height:5px; }
.history-strip::-webkit-scrollbar-thumb { background:#30363d; border-radius:3px; }
.hist-empty { color:var(--muted); font-size:.8rem; padding:20px; margin:auto; align-self:center; }
.obs-card { flex-shrink:0; width:128px; cursor:pointer; border-radius:7px;
            overflow:hidden; border:2px solid var(--border2);
            background:var(--panel2); transition:all .15s; display:flex; flex-direction:column; }
.obs-card:hover { border-color:var(--blue); transform:translateY(-2px);
                  box-shadow:0 4px 14px rgba(0,0,0,.45); }
.obs-card.lb-active { border-color:var(--blue);
                      box-shadow:0 0 0 3px rgba(88,166,255,.3); }
.obs-thumb { width:128px; height:96px; object-fit:cover; display:block; }
.obs-meta { padding:5px 7px; display:flex; flex-direction:column; gap:1px; }
.obs-num { font-size:.6rem; color:var(--muted); font-family:'SF Mono',Menlo,monospace; }
.obs-act { font-size:.68rem; color:var(--green); font-family:'SF Mono',Menlo,monospace;
           white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

/* lightbox */
.lightbox { position:fixed; inset:0; background:rgba(0,0,0,.9);
            z-index:9999; display:none; align-items:center; justify-content:center;
            backdrop-filter:blur(6px); }
.lightbox.open { display:flex; }
.lb-inner { position:relative; display:flex; flex-direction:column;
            align-items:center; max-width:92vw; gap:14px; }
#lb-img { max-width:88vw; max-height:72vh; border-radius:10px;
          box-shadow:0 8px 40px rgba(0,0,0,.75); border:1px solid #2a2f3a;
          display:block; }
.lb-caption { text-align:center; }
.lb-obs-num { font-size:.72rem; color:var(--muted); margin-bottom:4px; }
.lb-act-txt { font-size:1rem; color:var(--green); font-family:'SF Mono',Menlo,monospace;
              font-weight:600; }
.lb-raw-txt { font-size:.7rem; color:#6e7681; margin-top:6px; max-width:720px;
              word-break:break-word; }
.lb-close { position:absolute; top:-40px; right:0; background:none; border:none;
            color:#8b949e; font-size:1.5rem; cursor:pointer; padding:4px 8px;
            line-height:1; transition:color .12s; }
.lb-close:hover { color:var(--text); }
.lb-prev, .lb-next { position:fixed; top:50%; transform:translateY(-50%);
                     background:rgba(13,17,23,.8); border:1px solid var(--border2);
                     color:var(--text); font-size:1.1rem; padding:12px 16px;
                     border-radius:8px; cursor:pointer; transition:all .15s;
                     backdrop-filter:blur(4px); z-index:10000; }
.lb-prev { left:18px; }
.lb-next { right:18px; }
.lb-prev:hover:not(:disabled), .lb-next:hover:not(:disabled) {
  background:var(--panel2); border-color:var(--blue); }
.lb-prev:disabled, .lb-next:disabled { opacity:.2; cursor:not-allowed; }
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="logo">A</div>
    <h1>ActiveVLN — Go1</h1>
  </div>
  <div class="spacer"></div>
  <span class="badge connecting" id="status">Connecting</span>
  <span class="pause-banner">&#9208; Paused</span>
</header>

<div class="stats">
  <div class="stat instr">
    <span class="k">Instruction</span>
    <span class="v" id="stat-instr">—</span>
  </div>
  <div class="stat step">
    <span class="k">Step</span>
    <span class="v" id="stat-step">—</span>
  </div>
  <div class="stat action">
    <span class="k">Last Action</span>
    <span class="v" id="stat-action">—</span>
  </div>
  <div class="stat lat">
    <span class="k">LLM Latency</span>
    <span class="v" id="stat-lat">—</span>
  </div>
  <div class="stat memory">
    <span class="k">Spatial Memory</span>
    <span class="v" id="stat-memory">—</span>
  </div>
  <div class="stat obstacle" id="stat-obstacle-box">
    <span class="k">Obstacle Replan</span>
    <span class="v" id="stat-obstacle">—</span>
  </div>
</div>

<div class="ctrl">
  <span class="lbl">Instruction</span>
  <input id="instr-input" type="text"
         placeholder="e.g. walk to the door and stop">
  <button class="btn-set" id="set-btn" onclick="setInstruction()">&#x2713; Set</button>
  <button class="btn-voice" id="voice-btn" onclick="toggleVoice()">&#127908; Voice</button>
  <button class="btn-pause" id="pause-btn" onclick="togglePause()">&#x23F8; Pause</button>
  <button class="btn-estop" id="estop-btn" onclick="emergencyStop()">&#x26D4; Emergency Stop</button>
  <button class="btn-restart" onclick="restartSession()">&#x21BA; Restart</button>
</div>

<div class="voice-panel" id="voice-panel">
  <div class="voice-row">
    <span class="voice-dot" id="voice-dot"></span>
    <span class="voice-lbl" id="voice-status">Click Voice or press Space</span>
  </div>
  <div class="voice-text" id="voice-transcript"></div>
</div>

<div class="layout">
  <div class="cam-panel">
    <div class="cam-wrap">
      <img id="cam-img" src="/video_feed" alt="Camera feed">
      <div class="cam-overlay">
        <span class="cam-tag">ActiveVLN · LIVE</span>
        <span class="cam-tag" id="cam-tag">—</span>
      </div>
    </div>
  </div>
  <div class="log-panel">
    <div class="log-header">
      <span class="log-title">Model output &amp; actions</span>
      <button class="flt-btn on" id="flt-btn" onclick="toggleFilter()">Key only</button>
      <button class="flt-btn" onclick="clearLog()">Clear</button>
    </div>
    <div style="position:relative;flex:1;display:flex;overflow:hidden;">
      <div id="log"></div>
      <button class="scroll-btn" id="scroll-btn" onclick="scrollBottom()">&darr; Live</button>
    </div>
  </div>
</div>

<div class="history-panel">
  <div class="history-header">
    <span class="log-title">Observation History</span>
    <span class="hist-count" id="hist-count">0 frames</span>
  </div>
  <div class="history-strip" id="hist-strip">
    <span class="hist-empty" id="hist-empty">No observations yet — navigation will populate this strip</span>
  </div>
</div>

<div class="lightbox" id="lightbox">
  <div class="lb-inner" onclick="event.stopPropagation()">
    <button class="lb-close" onclick="closeLightbox()">&#x2715;</button>
    <img id="lb-img" src="" alt="observation">
    <div class="lb-caption">
      <div class="lb-obs-num" id="lb-obs-num"></div>
      <div class="lb-act-txt" id="lb-act-txt"></div>
      <div class="lb-raw-txt" id="lb-raw-txt"></div>
    </div>
  </div>
  <button class="lb-prev" id="lb-prev" onclick="lbNav(-1)">&#8592;</button>
  <button class="lb-next" id="lb-next" onclick="lbNav(+1)">&#8594;</button>
</div>

<script>
const $log          = document.getElementById('log');
const $status       = document.getElementById('status');
const $statInstr    = document.getElementById('stat-instr');
const $statStep     = document.getElementById('stat-step');
const $statAct      = document.getElementById('stat-action');
const $statLat      = document.getElementById('stat-lat');
const $statMemory   = document.getElementById('stat-memory');
const $statObstacle = document.getElementById('stat-obstacle');
const $statObstBox  = document.getElementById('stat-obstacle-box');
const $input        = document.getElementById('instr-input');
const $setBtn       = document.getElementById('set-btn');
const $pauseBtn     = document.getElementById('pause-btn');
const $estopBtn     = document.getElementById('estop-btn');
const $camTag       = document.getElementById('cam-tag');
const $scrollBtn    = document.getElementById('scroll-btn');

let paused = false, autoScroll = true;
const MAX_LINES = 600;

const ts = () => new Date().toTimeString().slice(0,8);

function setBadge(txt, cls) {
  $status.textContent = txt;
  $status.className = 'badge ' + cls;
}

function setPaused(p) {
  paused = p;
  document.body.classList.toggle('is-paused', p);
  $pauseBtn.innerHTML = p ? '&#9654; Resume' : '&#x23F8; Pause';
  setBadge(p ? 'Paused' : 'Live', p ? 'paused' : 'live');
}

function setEStopUI(active) {
  $estopBtn.disabled = active;
  $estopBtn.innerHTML = active ? '&#x26D4; E-Stop Active' : '&#x26D4; Emergency Stop';
  if (active) setBadge('E-Stop', 'offline');
}

function classify(t) {
  if (t.includes('EMERGENCY STOP'))                        return 'err';
  if (/\*\*\*|RESTART|INSTRUCTION/.test(t))                return 'system';
  if (t.includes('OBSTACLE REPLANNING') || t.includes('Obstacle')) return 'warn';
  if (t.includes('[Memory]') || t.includes('Landmark'))    return 'instr';
  if (t.includes('Inference complete') && t.includes('waiting for')) return 'warn';
  if (t.includes('[Step'))                                 return 'step';
  if (t.includes('[Go1]') || t.includes('DRY'))            return 'action';
  if (/Instruction\s*:/.test(t))                           return 'instr';
  if (t.includes('Error') || t.includes('Traceback'))      return 'err';
  if (/warn/i.test(t))                                     return 'warn';
  return 'info';
}

function parseStep(t) {
  const m = t.match(/\[Step\s+(\d+)\]\s+([\d.]+)s\s*[→\->]+\s*([A-Z][^(]*?)(?:\s{2,}|$)/);
  return m ? {step:m[1], lat:parseFloat(m[2]), action:m[3].trim()} : null;
}

function addLine(text) {
  if (!text) return;
  const cls = classify(text);
  const div = document.createElement('div');
  div.className = 'line ' + cls;
  div.innerHTML = '<span class="ts"></span><span class="txt"></span>';
  div.children[0].textContent = ts();
  div.children[1].textContent = text;
  $log.appendChild(div);
  if ($log.children.length > MAX_LINES) $log.removeChild($log.firstChild);
  if (autoScroll) $log.scrollTop = $log.scrollHeight;

  const s = parseStep(text);
  if (s) {
    $statStep.textContent = '#' + s.step;
    $statAct.textContent  = s.action;
    $statLat.textContent  = s.lat > 0 ? s.lat.toFixed(1)+'s' : '<0.1s';
    $camTag.textContent   = 'STEP ' + s.step + ' · ' + s.action;
  }
  const ga = text.match(/\[Go1\]\s+(.+?)\s+\(/);
  if (ga) $statAct.textContent = ga[1].trim();

  const mi = text.match(/Instruction\s*:\s*(.+)/);
  if (mi) $statInstr.textContent = mi[1].trim();
  const ni = text.match(/NEW INSTRUCTION.*?[→\->]+\s*(.+)/);
  if (ni) $statInstr.textContent = ni[1].trim();

  if (text.includes('PAUSED'))  setPaused(true);
  if (text.includes('EMERGENCY STOP')) {
    const cleared = text.includes('CLEARED');
    setEStopUI(!cleared);
    setPaused(!cleared);
  }
  if (text.includes('RESUMED') || text.includes('NEW INSTRUCTION')) setPaused(false);
}

function setInstruction() {
  const val = $input.value.trim();
  if (!val) { $input.focus(); return; }
  $setBtn.disabled = true;
  fetch('/set_instruction', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({instruction: val})
  }).then(r=>r.json()).then(d => {
    $statInstr.textContent = d.instruction;
    $statStep.textContent = $statAct.textContent = $statLat.textContent = '—';
    $camTag.textContent = '—';
    addLine('[INSTRUCTION] → ' + d.instruction);
    $input.value = '';
    setPaused(false);
  }).finally(() => { $setBtn.disabled = false; $input.focus(); });
}

function togglePause() {
  const next = !paused;
  fetch('/pause', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({paused: next})
  }).then(() => {
    if (!next) setEStopUI(false);
    setPaused(next);
    addLine(next ? '[PAUSED] by user' : '[RESUMED] by user');
  });
}

function emergencyStop() {
  if (!confirm('Emergency stop will halt current motion and block future actions until Resume. Continue?')) return;
  fetch('/emergency_stop', {
    method:'POST',
    headers:{'Content-Type':'application/json'}
  }).then(() => {
    setEStopUI(true);
    setPaused(true);
    addLine('[EMERGENCY STOP] current and future actions halted');
  });
}

function restartSession() {
  if (!confirm('Restart navigation? Step counter resets.')) return;
  fetch('/restart', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({instruction:''})
  }).then(() => {
    clearLog();
    $statStep.textContent = $statAct.textContent = $statLat.textContent = '—';
    $camTag.textContent = '—';
    addLine('[RESTART] Session reset');
    setPaused(true);
    $input.focus();
  });
}

function toggleFilter() {
  const on = !document.getElementById('flt-btn').classList.contains('on');
  document.getElementById('flt-btn').classList.toggle('on', on);
  document.body.classList.toggle('filter-on', on);
}
function clearLog()   { $log.innerHTML = ''; }
function scrollBottom() {
  autoScroll = true;
  $log.scrollTop = $log.scrollHeight;
  $scrollBtn.classList.remove('show');
}

$log.addEventListener('scroll', () => {
  const atBottom = $log.scrollHeight - $log.scrollTop - $log.clientHeight < 40;
  autoScroll = atBottom;
  $scrollBtn.classList.toggle('show', !atBottom);
});
$input.addEventListener('keydown', e => { if (e.key==='Enter') setInstruction(); });
document.body.classList.add('filter-on');

fetch('/get_instruction').then(r=>r.json()).then(d => {
  if (d.instruction) $statInstr.textContent = d.instruction;
});

const es = new EventSource('/events');
es.onopen  = () => { if (!paused) setBadge('Live','live'); };
es.onerror = () => setBadge('Offline','offline');
es.onmessage = e => addLine(e.data);

// ── Voice (Web Speech API) ────────────────────────────────────────────────────
let recog = null, isListening = false;
const $voiceBtn  = document.getElementById('voice-btn');
const $voicePanel= document.getElementById('voice-panel');
const $voiceDot  = document.getElementById('voice-dot');
const $voiceStat = document.getElementById('voice-status');
const $voiceTxt  = document.getElementById('voice-transcript');

function initRecog() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { alert('Speech recognition not supported. Use Chrome or Edge.'); return false; }
  recog = new SR();
  recog.continuous = true; recog.interimResults = true; recog.lang = 'en-US';

  recog.onstart = () => {
    isListening = true;
    $voiceBtn.classList.add('listening');
    $voiceBtn.textContent = '⏹ Stop';
    $voicePanel.classList.add('active');
    $voiceDot.className = 'voice-dot listening';
    $voiceStat.textContent = 'Listening…';
  };

  recog.onresult = e => {
    let interim='', final='';
    for (let i=e.resultIndex; i<e.results.length; i++) {
      const t = e.results[i][0].transcript;
      e.results[i].isFinal ? (final += t+' ') : (interim += t);
    }
    $voiceTxt.textContent = final || interim;
    if (final) {
      $voiceDot.className = 'voice-dot processing';
      $voiceStat.textContent = 'Processing…';
      sendVoiceInstr(final.trim());
    }
  };

  recog.onerror = e => {
    if (e.error === 'not-allowed') { alert('Microphone access denied.'); stopVoice(); }
    else if (e.error !== 'no-speech') addLine('[VOICE ERROR] ' + e.error);
  };

  recog.onend = () => {
    if (isListening) { try { recog.start(); } catch(_) {} }
  };
  return true;
}

function toggleVoice() {
  if (!recog && !initRecog()) return;
  isListening ? stopVoice() : recog.start();
}

function stopVoice() {
  isListening = false;
  recog && recog.stop();
  $voiceBtn.classList.remove('listening');
  $voiceBtn.innerHTML = '&#127908; Voice';
  $voiceDot.className = 'voice-dot';
  $voiceStat.textContent = 'Click Voice or press Space';
}

function sendVoiceInstr(text) {
  addLine('[VOICE] "' + text + '"');
  fetch('/set_instruction', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({instruction: text})
  }).then(r=>r.json()).then(d => {
    $statInstr.textContent = d.instruction;
    $statStep.textContent = $statAct.textContent = $statLat.textContent = '—';
    $camTag.textContent = '—';
    addLine('[INSTRUCTION] → ' + d.instruction);
    setPaused(false);
    if (window.speechSynthesis) {
      const u = new SpeechSynthesisUtterance('Instruction set: ' + text);
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(u);
    }
    setTimeout(() => {
      if (isListening) { $voiceDot.className='voice-dot listening'; $voiceStat.textContent='Listening…'; }
    }, 1200);
  });
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.code === 'Space')  { e.preventDefault(); toggleVoice(); }
  if (e.code === 'Escape' && isListening) { e.preventDefault(); stopVoice(); }
});

// ── Observation history ───────────────────────────────────────────────────────
let histData = [], lbIdx = 0;
const $histStrip = document.getElementById('hist-strip');
const $histEmpty = document.getElementById('hist-empty');
const $histCount = document.getElementById('hist-count');
const $lightbox  = document.getElementById('lightbox');

function fetchHistory() {
  fetch('/history').then(r => r.json()).then(data => {
    const prevLen = histData.length;
    if (data.length === prevLen) return;
    const reset = data.length < prevLen; // cleared on restart
    histData = data;
    renderHistory(reset);
  }).catch(() => {});
}

function renderHistory(reset) {
  $histCount.textContent = histData.length + (histData.length === 1 ? ' frame' : ' frames');
  if (histData.length === 0) {
    $histStrip.innerHTML = '';
    $histStrip.appendChild($histEmpty);
    return;
  }
  if (reset) {
    const cards = $histStrip.querySelectorAll('.obs-card');
    cards.forEach(c => c.remove());
  }
  const existing = $histStrip.querySelectorAll('.obs-card').length;
  for (let i = existing; i < histData.length; i++) {
    const d = histData[i];
    const card = document.createElement('div');
    card.className = 'obs-card';
    card.dataset.idx = i;
    card.onclick = () => openLightbox(i);
    const actTxt = d.actions.join(' → ') || 'STOP';
    card.innerHTML =
      '<img class="obs-thumb" src="/obs_frame/' + d.n + '" loading="lazy" alt="">' +
      '<div class="obs-meta">' +
        '<span class="obs-num">Obs ' + (i + 1) + '</span>' +
        '<span class="obs-act">' + actTxt + '</span>' +
      '</div>';
    $histStrip.appendChild(card);
  }
  $histStrip.scrollLeft = $histStrip.scrollWidth;
}

function openLightbox(idx) {
  lbIdx = idx;
  const d = histData[idx];
  document.getElementById('lb-img').src = '/obs_frame/' + d.n;
  document.getElementById('lb-obs-num').textContent = 'Obs ' + (idx + 1) + ' of ' + histData.length;
  document.getElementById('lb-act-txt').textContent = d.actions.join(' → ') || 'STOP';
  document.getElementById('lb-raw-txt').textContent = d.raw || '';
  document.getElementById('lb-prev').disabled = (idx === 0);
  document.getElementById('lb-next').disabled = (idx === histData.length - 1);
  $histStrip.querySelectorAll('.obs-card').forEach((c, i) =>
    c.classList.toggle('lb-active', i === idx));
  $lightbox.classList.add('open');
}

function closeLightbox() {
  $lightbox.classList.remove('open');
  $histStrip.querySelectorAll('.obs-card.lb-active').forEach(c => c.classList.remove('lb-active'));
}

function lbNav(dir) {
  const next = lbIdx + dir;
  if (next >= 0 && next < histData.length) openLightbox(next);
}

$lightbox.addEventListener('click', e => { if (e.target === $lightbox) closeLightbox(); });

document.addEventListener('keydown', e => {
  if (!$lightbox.classList.contains('open')) return;
  if (e.key === 'ArrowLeft')  { e.preventDefault(); lbNav(-1); }
  if (e.key === 'ArrowRight') { e.preventDefault(); lbNav(+1); }
  if (e.key === 'Escape')     closeLightbox();
});

setInterval(fetchHistory, 2000);
fetchHistory();

// ── Spatial memory + obstacle replan status ───────────────────────────────
function fetchStatusExtra() {
  fetch('/status_extra').then(r => r.json()).then(d => {
    // Memory panel
    const mem = d.memory;
    if (mem && mem.steps > 0) {
      const lmPart = mem.visited_landmarks && mem.visited_landmarks.length
        ? ' | ' + mem.visited_landmarks.join(', ')
        : '';
      $statMemory.textContent = `${mem.steps}stp ${mem.dist_m}m ${mem.heading_deg}°${lmPart}`;
    } else {
      $statMemory.textContent = 'inactive';
    }

    // Obstacle panel
    if (d.obstacle_active) {
      $statObstacle.textContent = '⚠ OBSTACLE';
      $statObstBox.classList.add('alert');
    } else {
      $statObstacle.textContent = mem && mem.steps > 0 ? 'clear' : '—';
      $statObstBox.classList.remove('alert');
    }
  }).catch(() => {});
}

setInterval(fetchStatusExtra, 1000);
fetchStatusExtra();
</script>
</body>
</html>"""


# ── webcam reader (shared frame written by go1_nav.py) ───────────────────────

def _frame_reader():
    global _latest_frame
    while True:
        try:
            if os.path.exists(_shared_frame):
                with open(_shared_frame, 'rb') as f:
                    data = f.read()
                if data:
                    with _frame_lock:
                        _latest_frame = data
        except Exception:
            pass
        time.sleep(1 / 15)


def _mjpeg():
    _placeholder = None
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            if _placeholder is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(img, "Waiting for camera…", (100, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)
                _, buf = cv2.imencode('.jpg', img)
                _placeholder = buf.tobytes()
            frame = _placeholder
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        time.sleep(1 / 15)


# ── SSE log stream ────────────────────────────────────────────────────────────

def _sse():
    while not os.path.exists(_log_path):
        yield "data: [UI] waiting for go1_nav.py to start…\n\n"
        time.sleep(1)
    with open(_log_path, 'r') as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            clean = _ANSI_RE.sub('', line.rstrip())
            if not clean or _is_noise(clean):
                continue
            yield f"data: {clean}\n\n"


# ── routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/events')
def events():
    return Response(_sse(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/set_instruction', methods=['POST'])
def set_instruction():
    instr = (request.get_json() or {}).get('instruction', '').strip()
    if instr:
        open(_restart_flag, 'w').write(instr)
        open(_instr_file, 'w').write(instr)
        try:
            os.remove(_estop_flag)
        except FileNotFoundError:
            pass
        try:
            os.remove(_pause_flag)
        except FileNotFoundError:
            pass
    return jsonify(instruction=instr)

@app.route('/get_instruction')
def get_instruction():
    instr = open(_instr_file).read().strip() if os.path.exists(_instr_file) else ''
    return jsonify(instruction=instr)

@app.route('/pause', methods=['POST'])
def pause():
    want = bool((request.get_json() or {}).get('paused', True))
    if want: open(_pause_flag, 'w').close()
    else:
        try: os.remove(_estop_flag)
        except FileNotFoundError: pass
        try: os.remove(_pause_flag)
        except FileNotFoundError: pass
    return jsonify(paused=want)

@app.route('/restart', methods=['POST'])
def restart():
    instr = (request.get_json() or {}).get('instruction', '').strip()
    open(_restart_flag, 'w').write(instr)
    open(_instr_file, 'w').write(instr)
    open(_pause_flag, 'w').write("1")
    return jsonify(instruction=instr)

@app.route('/emergency_stop', methods=['POST'])
def emergency_stop():
    open(_estop_flag, 'w').write(str(time.time()))
    open(_pause_flag, 'w').write("1")
    return jsonify(ok=True, estop=True)

@app.route('/status_extra')
def status_extra():
    """Return spatial memory state and obstacle-replan flag for the UI."""
    mem = {}
    try:
        if os.path.exists(_memory_file):
            with open(_memory_file) as f:
                mem = json.load(f)
    except Exception:
        pass
    obstacle_active = os.path.exists(_obstacle_file)
    return jsonify(memory=mem, obstacle_active=obstacle_active)

@app.route('/history')
def history():
    if not os.path.exists(_hist_file):
        return jsonify([])
    try:
        with open(_hist_file, 'r') as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return jsonify(data)
    except Exception:
        pass
    return jsonify([])

@app.route('/obs_frame/<int:n>')
def obs_frame(n: int):
    path = os.path.join(_obs_dir, f"{n:04d}.jpg")
    if not os.path.exists(path):
        abort(404)
    try:
        with open(path, 'rb') as fh:
            return Response(fh.read(), mimetype='image/jpeg')
    except OSError:
        abort(404)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=5000)
    p.add_argument('--log',  default='/tmp/activevln.log',
                   help='Path to go1_nav.py log file (tail -f style)')
    args = p.parse_args()
    _log_path = args.log

    threading.Thread(target=_frame_reader, daemon=True).start()

    host_ip = os.popen("hostname -I 2>/dev/null | awk '{print $1}'").read().strip() or '127.0.0.1'
    print(f"[UI] Open  http://{host_ip}:{args.port}  in your browser")
    app.run(host='0.0.0.0', port=args.port, threaded=True)
