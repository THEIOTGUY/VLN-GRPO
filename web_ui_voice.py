#!/usr/bin/env python3
"""
VLN Web UI with VOICE — live webcam feed + real-time inference log + voice/text instruction input.
Run alongside infer.py (infer.py writes to /tmp/infer_live.log).

NEW FEATURES:
- Voice input using Web Speech API (browser-based, no server dependencies)
- Voice output using Web Speech Synthesis
- Push-to-talk and continuous listening modes
- Real-time transcription display
"""

import argparse
import os
import re
import time
import threading
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify

app = Flask(__name__)

_log_path     = "/tmp/infer_live.log"
_shared_frame = "/tmp/vln_latest_frame.jpg"
_instr_file   = "/tmp/vln_instruction.txt"
_frame_lock   = threading.Lock()
_latest_frame = None

# strip ANSI color codes from log lines before streaming to browser
_ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# lines we never want to show in the UI — noisy Flask + torchvision warnings
_NOISE_PATTERNS = (
    re.compile(r'^\s*\d+\.\d+\.\d+\.\d+ - - \['),     # Werkzeug request log
    re.compile(r'Press CTRL\+C to quit'),
    re.compile(r'Serving Flask app'),
    re.compile(r'Debug mode: off'),
    re.compile(r'Running on'),
    re.compile(r'This is a development server'),
    re.compile(r'^\s*warn\(', re.IGNORECASE),
    re.compile(r'UserWarning'),
    re.compile(r'torchvision'),
    re.compile(r'Loading checkpoint shards:'),
)

def _is_noise(line: str) -> bool:
    return any(p.search(line) for p in _NOISE_PATTERNS)


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>VLN — Live Inference</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0a0d12;
    --panel: #0d1117;
    --panel-2: #141820;
    --border: #2a2f3a;
    --border-2: #30363d;
    --muted: #8b949e;
    --text: #e6edf3;
    --blue: #58a6ff;
    --green: #56d364;
    --orange: #f0883e;
    --yellow: #e3b341;
    --red: #f85149;
    --purple: #bc8cff;
  }
  html, body { height: 100%; }
  body { background: var(--bg); color: var(--text);
         font-family: -apple-system, 'Segoe UI', system-ui, sans-serif;
         display: flex; flex-direction: column; height: 100vh; overflow: hidden;
         font-size: 14px; }

  /* ── header ─────────────────────────────────────────────── */
  header { background: linear-gradient(180deg, #1a1f26 0%, #141820 100%);
           padding: 12px 22px; border-bottom: 1px solid var(--border);
           display: flex; align-items: center; gap: 14px; flex-shrink: 0; }
  header .brand { display: flex; align-items: center; gap: 10px; }
  header .logo { width: 28px; height: 28px; border-radius: 8px;
                 background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 100%);
                 display: flex; align-items: center; justify-content: center;
                 font-weight: 700; font-size: .85rem; color: #0a0d12; }
  header h1 { font-size: 1rem; font-weight: 600; letter-spacing: 0.3px; }
  header .spacer { flex: 1; }
  .badge { font-size: .68rem; padding: 4px 10px; border-radius: 99px;
           font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px;
           display: inline-flex; align-items: center; gap: 6px;
           transition: background 0.2s; }
  .badge::before { content: ''; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
  .badge.live      { background: rgba(35,134,54,0.18);  color: #4ac26b; }
  .badge.live::before { animation: pulse 1.4s ease-out infinite; }
  .badge.paused    { background: rgba(227,179,65,0.18); color: #e3b341; }
  .badge.offline   { background: rgba(182,35,36,0.18);  color: #f85149; }
  .badge.connecting{ background: rgba(139,148,158,0.18);color: #8b949e; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

  /* ── stats row ───────────────────────────────────────────── */
  .stats { display: grid; grid-template-columns: repeat(4, 1fr);
           gap: 1px; background: var(--border); flex-shrink: 0; }
  .stat { background: var(--panel-2); padding: 10px 18px;
          display: flex; flex-direction: column; gap: 2px; }
  .stat .k { font-size: .66rem; color: var(--muted);
             text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600; }
  .stat .v { font-size: 1.02rem; font-weight: 600; color: var(--text);
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .stat.action .v { color: var(--green); font-family: 'SF Mono', Menlo, Consolas, monospace; }
  .stat.instr  .v { color: var(--orange); }
  .stat.step   .v { color: var(--blue); }
  .stat.lat    .v { color: var(--purple); }

  /* ── control bar ─────────────────────────────────────────── */
  .ctrl-bar { background: var(--panel-2); border-top: 1px solid var(--border);
              border-bottom: 1px solid var(--border);
              padding: 10px 20px; display: flex; gap: 10px; align-items: center;
              flex-shrink: 0; }
  .ctrl-bar .label { font-size: .7rem; color: var(--muted);
                     text-transform: uppercase; letter-spacing: 0.6px;
                     font-weight: 600; white-space: nowrap; }
  #instr-input { flex: 1; background: var(--panel); border: 1px solid var(--border-2);
                 color: var(--text); padding: 9px 13px; border-radius: 7px;
                 font-size: .92rem; font-family: inherit; transition: all 0.15s; }
  #instr-input::placeholder { color: #586069; }
  #instr-input:focus { outline: none; border-color: var(--blue);
                       box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.18); }
  button { border: none; padding: 9px 16px; border-radius: 7px;
           cursor: pointer; font-size: .85rem; font-weight: 600;
           transition: all 0.15s; font-family: inherit;
           display: inline-flex; align-items: center; gap: 6px; }
  button:hover:not(:disabled) { transform: translateY(-1px); }
  button:active:not(:disabled) { transform: translateY(0); }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  button.set      { background: #238636; color: #fff; }
  button.set:hover:not(:disabled)      { background: #2ea043; }
  button.voice    { background: #8250df; color: #fff; }
  button.voice:hover:not(:disabled)    { background: #9a6dea; }
  button.voice.listening { background: #f85149; animation: pulse-btn 1.5s ease-in-out infinite; }
  @keyframes pulse-btn { 0%,100% { opacity: 1; } 50% { opacity: 0.7; } }
  button.pause    { background: #30363d; color: var(--text); }
  button.pause:hover:not(:disabled)    { background: #3d444d; }
  button.restart  { background: #b62324; color: #fff; }
  button.restart:hover:not(:disabled)  { background: #da3633; }

  /* ── voice panel ─────────────────────────────────────────── */
  .voice-panel { background: var(--panel-2); border-bottom: 1px solid var(--border);
                 padding: 10px 20px; display: none; flex-direction: column; gap: 8px;
                 flex-shrink: 0; }
  .voice-panel.active { display: flex; }
  .voice-status { display: flex; align-items: center; gap: 10px; }
  .voice-indicator { width: 12px; height: 12px; border-radius: 50%;
                     background: #30363d; transition: all 0.3s; }
  .voice-indicator.listening { background: #f85149; animation: pulse 1.5s ease-in-out infinite; }
  .voice-indicator.processing { background: #e3b341; }
  .voice-text { font-size: .75rem; color: var(--muted); font-weight: 600;
                text-transform: uppercase; letter-spacing: 0.5px; }
  .voice-transcript { font-size: .9rem; color: var(--text); padding: 8px 12px;
                      background: var(--panel); border-radius: 6px;
                      border: 1px solid var(--border-2); min-height: 40px;
                      font-family: inherit; }
  .voice-transcript:empty::before { content: 'Your speech will appear here...';
                                    color: #586069; font-style: italic; }

  /* ── main layout ─────────────────────────────────────────── */
  .layout { display: grid; grid-template-columns: 1fr 460px;
            flex: 1; overflow: hidden; gap: 1px; background: var(--border); }

  /* camera panel */
  .cam-panel { background: #000; display: flex; flex-direction: column;
               align-items: stretch; justify-content: center;
               position: relative; overflow: hidden; }
  .cam-panel .cam-wrap { flex: 1; display: flex; align-items: center;
                         justify-content: center; padding: 16px;
                         position: relative; overflow: hidden; }
  .cam-panel img { max-width: 100%; max-height: 100%; border-radius: 10px;
                   box-shadow: 0 4px 24px rgba(0,0,0,0.55);
                   border: 1px solid #1a1f26; }
  .cam-overlay { position: absolute; top: 24px; left: 24px; right: 24px;
                 display: flex; justify-content: space-between; pointer-events: none;
                 font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: .72rem; }
  .cam-overlay .tag { background: rgba(10,13,18,0.72); color: #c9d1d9;
                      padding: 4px 10px; border-radius: 5px;
                      backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,0.08); }

  /* log panel */
  .log-panel { background: var(--panel); display: flex; flex-direction: column;
               overflow: hidden; }
  .log-header { padding: 10px 16px; background: var(--panel-2);
                border-bottom: 1px solid var(--border);
                display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
  .log-header .title { font-size: .7rem; color: var(--muted); font-weight: 600;
                       text-transform: uppercase; letter-spacing: 0.6px; flex: 1; }
  .filter-btn { padding: 4px 10px; border-radius: 5px; font-size: .68rem;
                background: transparent; border: 1px solid var(--border-2);
                color: var(--muted); cursor: pointer; font-weight: 600;
                text-transform: uppercase; letter-spacing: 0.5px;
                transition: all 0.12s; }
  .filter-btn:hover { color: var(--text); border-color: #484f58; }
  .filter-btn.on { background: var(--blue); border-color: var(--blue);
                   color: #0a0d12; }
  #log { flex: 1; overflow-y: auto; padding: 10px 16px;
         font-family: 'SF Mono', Menlo, Consolas, monospace;
         font-size: .76rem; line-height: 1.6; scroll-behavior: smooth; }
  #log::-webkit-scrollbar { width: 7px; }
  #log::-webkit-scrollbar-track { background: transparent; }
  #log::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
  #log::-webkit-scrollbar-thumb:hover { background: #484f58; }

  .line { padding: 2px 0; word-break: break-word; display: flex; gap: 8px; }
  .line .ts { color: #484f58; flex-shrink: 0; font-size: .68rem;
              padding-top: 1px; min-width: 44px; }
  .line .txt { flex: 1; }
  .line.step   .txt { color: var(--blue); font-weight: 600; }
  .line.action .txt { color: var(--green); }
  .line.instr  .txt { color: var(--orange); font-weight: 600; }
  .line.warn   .txt { color: var(--yellow); }
  .line.err    .txt { color: var(--red); font-weight: 500; }
  .line.info   .txt { color: #6e7681; }
  .line.system { margin: 4px 0; padding: 5px 10px; border-radius: 5px;
                 background: rgba(188,140,255,0.08);
                 border-left: 3px solid var(--purple); }
  .line.system .txt { color: var(--purple); font-weight: 600; }

  body.filter-on .line.info { display: none; }

  .pause-banner { display: none; margin: 0 0 0 auto;
                  background: rgba(227,179,65,0.15);
                  border: 1px solid #e3b341; color: var(--yellow);
                  padding: 4px 10px; border-radius: 6px; font-size: .72rem;
                  font-weight: 600; align-items: center; gap: 6px; }
  body.is-paused .pause-banner { display: inline-flex; }

  /* scroll-to-bottom button */
  .scroll-btn { position: absolute; right: 16px; bottom: 14px;
                background: var(--blue); color: #0a0d12;
                border: none; padding: 6px 11px; border-radius: 20px;
                font-size: .7rem; font-weight: 700; cursor: pointer;
                display: none; box-shadow: 0 3px 10px rgba(0,0,0,0.4); }
  .scroll-btn.show { display: block; }

  /* responsive */
  @media (max-width: 980px) {
    .layout { grid-template-columns: 1fr; grid-template-rows: 1fr 1fr; }
    .stats { grid-template-columns: repeat(2, 1fr); }
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="logo">V</div>
    <h1>VLN Live Inference</h1>
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
    <span class="k">Current Action</span>
    <span class="v" id="stat-action">—</span>
  </div>
  <div class="stat lat">
    <span class="k">LLM Latency</span>
    <span class="v" id="stat-lat">—</span>
  </div>
</div>

<div class="ctrl-bar">
  <span class="label">New Instruction</span>
  <input id="instr-input" type="text"
         placeholder="e.g.  walk to the kitchen and stop at the fridge">
  <button class="set" id="set-btn" onclick="setInstruction()">&#x2713; Set</button>
  <button class="voice" id="voice-btn" onclick="toggleVoice()">🎤 Voice</button>
  <button class="pause" id="pause-btn" onclick="togglePause()">&#x23F8; Pause</button>
  <button class="restart" onclick="restartSession()">&#x21BA; Restart</button>
</div>

<div class="voice-panel" id="voice-panel">
  <div class="voice-status">
    <span class="voice-indicator" id="voice-indicator"></span>
    <span class="voice-text" id="voice-status-text">Click Voice button to start</span>
  </div>
  <div class="voice-transcript" id="voice-transcript"></div>
</div>

<div class="layout">
  <div class="cam-panel">
    <div class="cam-wrap">
      <img src="/video_feed" alt="Webcam feed">
      <div class="cam-overlay">
        <span class="tag" id="cam-tag-l">CAM · LIVE</span>
        <span class="tag" id="cam-tag-r">—</span>
      </div>
    </div>
  </div>
  <div class="log-panel">
    <div class="log-header">
      <span class="title">Model output &amp; actions</span>
      <button class="filter-btn on" id="filter-btn" onclick="toggleFilter()">Important only</button>
      <button class="filter-btn" onclick="clearLog()">Clear</button>
    </div>
    <div style="position:relative; flex:1; display:flex; overflow:hidden;">
      <div id="log"></div>
      <button class="scroll-btn" id="scroll-btn" onclick="scrollToBottom()">&darr; Live</button>
    </div>
  </div>
</div>

<script>
const log         = document.getElementById('log');
const statusEl    = document.getElementById('status');
const statInstr   = document.getElementById('stat-instr');
const statStep    = document.getElementById('stat-step');
const statAction  = document.getElementById('stat-action');
const statLat     = document.getElementById('stat-lat');
const instrInput  = document.getElementById('instr-input');
const setBtn      = document.getElementById('set-btn');
const pauseBtn    = document.getElementById('pause-btn');
const filterBtn   = document.getElementById('filter-btn');
const scrollBtn   = document.getElementById('scroll-btn');
const camTagR     = document.getElementById('cam-tag-r');

const MAX_LINES = 500;
let paused = false;
let autoScroll = true;

function ts() {
  const d = new Date();
  return d.toTimeString().slice(0,8);
}

function setBadge(text, cls) {
  statusEl.textContent = text;
  statusEl.className = 'badge ' + cls;
}

function setPaused(p) {
  paused = p;
  document.body.classList.toggle('is-paused', p);
  pauseBtn.innerHTML = p ? '&#9654; Resume' : '&#x23F8; Pause';
  if (p) setBadge('Paused', 'paused');
  else   setBadge('Live',   'live');
}

function classify(text) {
  if (text.includes('***') || text.includes('[RESTART]') || text.includes('[INSTRUCTION]'))
    return 'system';
  if (text.includes('[Step'))                           return 'step';
  if (text.includes('[Go1]') || text.includes('DRY-RUN')) return 'action';
  if (text.match(/Instruction\\s*:/))                   return 'instr';
  if (text.includes('Error') || text.includes('Traceback')) return 'err';
  if (text.toLowerCase().includes('warn'))              return 'warn';
  return 'info';
}

function parseStep(text) {
  // [Step 013] 32.2s  →  FORWARD 25cm   ('...')
  const m = text.match(/\\[Step\\s+(\\d+)\\]\\s+([\\d.]+)s\\s*[\\u2192\\->]+\\s*([A-Z][^(]*?)(?:\\s{2,}|$)/);
  if (!m) return null;
  return { step: m[1], latency: parseFloat(m[2]), action: m[3].trim() };
}

function parseGo1(text) {
  // [Go1] Turn left  15°  (0.52 s)
  const m = text.match(/\\[Go1\\]\\s+(.+?)\\s+\\(/);
  return m ? m[1].trim() : null;
}

function addLine(text) {
  if (!text) return;
  const cls = classify(text);
  const row = document.createElement('div');
  row.className = 'line ' + cls;
  row.innerHTML = '<span class="ts"></span><span class="txt"></span>';
  row.children[0].textContent = ts();
  row.children[1].textContent = text;
  log.appendChild(row);
  if (log.children.length > MAX_LINES) log.removeChild(log.firstChild);
  if (autoScroll) log.scrollTop = log.scrollHeight;

  // ── update stats ──────────────────────────────────────────
  const stepInfo = parseStep(text);
  if (stepInfo) {
    statStep.textContent   = '#' + stepInfo.step;
    statAction.textContent = stepInfo.action;
    statLat.textContent    = stepInfo.latency > 0
                           ? stepInfo.latency.toFixed(1) + 's'
                           : '< 0.1s (cached)';
    camTagR.textContent    = 'STEP ' + stepInfo.step + ' · ' + stepInfo.action;
  }
  const go1 = parseGo1(text);
  if (go1) statAction.textContent = go1;

  const mI = text.match(/Instruction\\s*:\\s*(.+)/);
  if (mI) statInstr.textContent = mI[1].trim();
  const nI = text.match(/NEW INSTRUCTION \\*\\*\\*\\s*[\\u2192\\->]+\\s*(.+)/);
  if (nI) statInstr.textContent = nI[1].trim();

  if (text.includes('PAUSED'))  setPaused(true);
  if (text.includes('RESUMED')) setPaused(false);
  if (text.includes('NEW INSTRUCTION')) setPaused(false);
}

function setInstruction() {
  const val = instrInput.value.trim();
  if (!val) { instrInput.focus(); return; }
  setBtn.disabled = true;
  fetch('/set_instruction', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({instruction: val})
  }).then(r => r.json()).then(d => {
    statInstr.textContent = d.instruction;
    statStep.textContent   = '—';
    statAction.textContent = '—';
    statLat.textContent    = '—';
    camTagR.textContent    = '—';
    addLine('[INSTRUCTION] -> ' + d.instruction);
    instrInput.value = '';
    setPaused(false);
  }).finally(() => { setBtn.disabled = false; instrInput.focus(); });
}

function togglePause() {
  const next = !paused;
  fetch('/pause', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({paused: next})
  }).then(() => {
    setPaused(next);
    addLine(next ? '[PAUSED] by user' : '[RESUMED] by user');
  });
}

function restartSession() {
  if (!confirm('Restart inference session? Step counter will reset.')) return;
  fetch('/restart', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({instruction: ''})
  }).then(() => {
    addLine('[RESTART] Session reset');
    statStep.textContent   = '—';
    statAction.textContent = '—';
    statLat.textContent    = '—';
    camTagR.textContent    = '—';
    clearLog();
    setPaused(true);
    instrInput.focus();
  });
}

function toggleFilter() {
  const on = !filterBtn.classList.contains('on');
  filterBtn.classList.toggle('on', on);
  document.body.classList.toggle('filter-on', on);
}

function clearLog() { log.innerHTML = ''; }

function scrollToBottom() {
  autoScroll = true;
  log.scrollTop = log.scrollHeight;
  scrollBtn.classList.remove('show');
}

log.addEventListener('scroll', () => {
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  autoScroll = atBottom;
  scrollBtn.classList.toggle('show', !atBottom);
});

instrInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') setInstruction();
});

// init filter state: "Important only" is ON by default
document.body.classList.add('filter-on');

fetch('/get_instruction').then(r => r.json()).then(d => {
  if (d.instruction) statInstr.textContent = d.instruction;
});

const es = new EventSource('/events');
es.onopen    = () => { if (!paused) setBadge('Live', 'live'); };
es.onerror   = () => { setBadge('Offline', 'offline'); };
es.onmessage = (e) => addLine(e.data);

// ══════════════════════════════════════════════════════════════════════════════
// VOICE CONTROL - Web Speech API (Browser-based, no server dependencies)
// ══════════════════════════════════════════════════════════════════════════════

let recognition = null;
let isListening = false;
let speechSynthesis = window.speechSynthesis;

// Initialize Speech Recognition
function initVoiceRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  
  if (!SpeechRecognition) {
    alert('Speech recognition not supported in this browser. Please use Chrome, Edge, or Safari.');
    return false;
  }
  
  recognition = new SpeechRecognition();
  recognition.continuous = true;  // Keep listening
  recognition.interimResults = true;  // Show partial results
  recognition.lang = 'en-US';
  
  recognition.onstart = () => {
    isListening = true;
    document.getElementById('voice-btn').classList.add('listening');
    document.getElementById('voice-btn').innerHTML = '⏹️ Stop';
    document.getElementById('voice-panel').classList.add('active');
    document.getElementById('voice-indicator').classList.add('listening');
    document.getElementById('voice-status-text').textContent = 'Listening...';
    addLine('[VOICE] Listening started');
  };
  
  recognition.onresult = (event) => {
    let interimTranscript = '';
    let finalTranscript = '';
    
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalTranscript += transcript + ' ';
      } else {
        interimTranscript += transcript;
      }
    }
    
    // Update transcript display
    const transcriptEl = document.getElementById('voice-transcript');
    if (finalTranscript) {
      transcriptEl.textContent = finalTranscript.trim();
      document.getElementById('voice-indicator').classList.remove('listening');
      document.getElementById('voice-indicator').classList.add('processing');
      document.getElementById('voice-status-text').textContent = 'Processing...';
      
      // Send to robot
      sendVoiceInstruction(finalTranscript.trim());
    } else if (interimTranscript) {
      transcriptEl.textContent = interimTranscript;
    }
  };
  
  recognition.onerror = (event) => {
    console.error('Speech recognition error:', event.error);
    if (event.error === 'no-speech') {
      document.getElementById('voice-status-text').textContent = 'No speech detected, still listening...';
    } else if (event.error === 'not-allowed') {
      alert('Microphone access denied. Please allow microphone access and try again.');
      stopVoice();
    } else {
      addLine('[VOICE ERROR] ' + event.error);
    }
  };
  
  recognition.onend = () => {
    if (isListening) {
      // Auto-restart if we're still supposed to be listening
      try {
        recognition.start();
      } catch (e) {
        console.log('Recognition restart failed:', e);
      }
    }
  };
  
  return true;
}

function toggleVoice() {
  if (!recognition && !initVoiceRecognition()) {
    return;
  }
  
  if (isListening) {
    stopVoice();
  } else {
    startVoice();
  }
}

function startVoice() {
  try {
    recognition.start();
  } catch (e) {
    console.log('Recognition already started or error:', e);
  }
}

function stopVoice() {
  isListening = false;
  if (recognition) {
    recognition.stop();
  }
  document.getElementById('voice-btn').classList.remove('listening');
  document.getElementById('voice-btn').innerHTML = '🎤 Voice';
  document.getElementById('voice-indicator').classList.remove('listening', 'processing');
  document.getElementById('voice-status-text').textContent = 'Click Voice button to start';
  addLine('[VOICE] Listening stopped');
}

function sendVoiceInstruction(text) {
  if (!text) return;
  
  addLine('[VOICE] Recognized: "' + text + '"');
  
  // Send to robot via existing API
  fetch('/set_instruction', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({instruction: text})
  })
  .then(r => r.json())
  .then(d => {
    statInstr.textContent = d.instruction;
    statStep.textContent   = '—';
    statAction.textContent = '—';
    statLat.textContent    = '—';
    camTagR.textContent    = '—';
    addLine('[INSTRUCTION] -> ' + d.instruction);
    setPaused(false);

    // Speak confirmation
    speakText('Instruction received: ' + text);
    
    // Reset voice UI
    setTimeout(() => {
      document.getElementById('voice-indicator').classList.remove('processing');
      if (isListening) {
        document.getElementById('voice-indicator').classList.add('listening');
        document.getElementById('voice-status-text').textContent = 'Listening...';
      }
    }, 1000);
  })
  .catch(err => {
    console.error('Error sending instruction:', err);
    addLine('[ERROR] Failed to send instruction');
    speakText('Error sending instruction');
  });
}

function speakText(text) {
  if (!speechSynthesis) return;
  
  // Cancel any ongoing speech
  speechSynthesis.cancel();
  
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  utterance.volume = 1.0;
  utterance.lang = 'en-US';
  
  utterance.onstart = () => {
    addLine('[VOICE] Speaking: "' + text + '"');
  };
  
  utterance.onerror = (event) => {
    console.error('Speech synthesis error:', event);
  };
  
  speechSynthesis.speak(utterance);
}

// Keyboard shortcut: Space bar to toggle voice
document.addEventListener('keydown', (e) => {
  // Only if not typing in input field
  if (e.target.tagName !== 'INPUT' && e.code === 'Space') {
    e.preventDefault();
    toggleVoice();
  }
  // Escape to stop voice
  if (e.code === 'Escape' && isListening) {
    e.preventDefault();
    stopVoice();
  }
});

// Speak action updates (optional - can be toggled)
let speakActions = false;  // Set to true to enable voice feedback for all actions

// Enhanced addLine to speak important updates
const originalAddLine = addLine;
function addLineWithVoice(text) {
  originalAddLine(text);
  
  // Speak important system messages
  if (speakActions) {
    if (text.includes('[Step') && text.includes('→')) {
      const action = text.match(/→\s*(.+?)(?:\s{2,}|$)/);
      if (action) {
        speakText(action[1].trim());
      }
    } else if (text.includes('[RESTART]')) {
      speakText('Session restarted');
    } else if (text.includes('[PAUSED]')) {
      speakText('Paused');
    } else if (text.includes('[RESUMED]')) {
      speakText('Resumed');
    }
  }
}

// Replace addLine with voice-enabled version
addLine = addLineWithVoice;

console.log('[VOICE] Voice control initialized. Press Space to toggle voice input.');

</script>
</body>
</html>
"""

# ── frame reader ──────────────────────────────────────────────────────────────
def _capture_loop():
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

def _mjpeg_gen():
    placeholder = None
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            if placeholder is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(img, "Waiting for inference...", (80, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                _, buf = cv2.imencode('.jpg', img)
                placeholder = buf.tobytes()
            frame = placeholder
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(1 / 15)

# ── SSE log stream ────────────────────────────────────────────────────────────
def _sse_gen():
    while not os.path.exists(_log_path):
        yield "data: waiting for inference to start…\n\n"
        time.sleep(1)
    with open(_log_path, 'r') as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.15)
                continue
            clean = _ANSI_RE.sub('', line.rstrip())
            if not clean or _is_noise(clean):
                continue
            # SSE requires each line to be prefixed with "data: ";
            # for multi-line payloads split on newlines — but we already stripped \n.
            yield f"data: {clean}\n\n"

# ── routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(_mjpeg_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/events')
def events():
    return Response(_sse_gen(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/set_instruction', methods=['POST'])
def set_instruction():
    data = request.get_json()
    instr = (data or {}).get('instruction', '').strip()
    if instr:
        with open(_instr_file, 'w') as f:
            f.write(instr)
        # setting a new instruction always resumes inference
        try: os.remove('/tmp/vln_paused.flag')
        except FileNotFoundError: pass
    return jsonify(instruction=instr)

@app.route('/get_instruction')
def get_instruction():
    instr = ''
    if os.path.exists(_instr_file):
        instr = open(_instr_file).read().strip()
    return jsonify(instruction=instr)

@app.route('/pause', methods=['POST'])
def pause():
    data = request.get_json() or {}
    want_paused = bool(data.get('paused', True))
    flag = '/tmp/vln_paused.flag'
    if want_paused:
        open(flag, 'w').close()
    else:
        try: os.remove(flag)
        except FileNotFoundError: pass
    return jsonify(paused=want_paused)

@app.route('/restart', methods=['POST'])
def restart():
    data = request.get_json()
    instr = (data or {}).get('instruction', '').strip()
    with open('/tmp/vln_restart.flag', 'w') as f:
        f.write(instr)
    if instr:
        with open(_instr_file, 'w') as f:
            f.write(instr)
    return jsonify(instruction=instr)

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--log', default='/tmp/infer_live.log')
    args = parser.parse_args()
    _log_path = args.log

    threading.Thread(target=_capture_loop, daemon=True).start()

    print(f"[UI] Open  http://<orin-ip>:{args.port}  in your browser")
    app.run(host='0.0.0.0', port=args.port, threaded=True)
