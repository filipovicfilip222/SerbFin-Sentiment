"""
Local hand-labeling tool for the gold test set (SPEC.md §4.4 — Phase 2).

Pure Python standard library only — no Flask, no pip installs. Robust on Windows.
Starts a tiny local web server and opens a browser page that shows ONE headline
at a time with positive / negative / neutral controls (buttons or keys).

Key properties (per project requirements):
  * Loads   data/processed/test_to_label.jsonl   (240 records — never modified)
  * The `silver_hint` is NEVER sent to the page until you explicitly request it,
    and the "reveal hint" control is disabled until AFTER you pick a label.
  * Every choice is written to disk immediately (resume-safe). If you quit
    half-way, relaunch and you continue exactly where you left off.
  * Progress (e.g. 47/240) + per-class running counts are always visible.
  * When all 240 are labeled, writes the final  data/processed/test.jsonl
    with  "label_source": "human"  on every record.

Launch:
    python src/label/label_tool.py
then open the printed URL (it also tries to open your browser automatically).

Keys:  1 = positive   2 = neutral   3 = negative
       ./Right = next   ,/Left = prev   h = reveal hint (after choosing)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "test_to_label.jsonl"
PARTIAL_PATH = ROOT / "data" / "processed" / "test.partial.jsonl"  # resume file
OUTPUT_PATH = ROOT / "data" / "processed" / "test.jsonl"           # final gold set

LABELS = ("positive", "negative", "neutral")
HOST, PORT = "127.0.0.1", 5050

# --- In-memory state ---------------------------------------------------------
RECORDS: list[dict] = []        # source records (incl. silver_hint), original order
ID_TO_REC: dict[str, dict] = {}
CHOICES: dict[str, str] = {}    # id -> chosen label (human)
LOCK = threading.Lock()


def load_records() -> None:
    if not INPUT_PATH.exists():
        sys.exit(f"ERROR: input not found: {INPUT_PATH}")
    with INPUT_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            RECORDS.append(obj)
            ID_TO_REC[obj["id"]] = obj
    # Resume: load any previously saved choices.
    if PARTIAL_PATH.exists():
        with PARTIAL_PATH.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("id") in ID_TO_REC and obj.get("label") in LABELS:
                    CHOICES[obj["id"]] = obj["label"]


def _record_out(rec: dict, label: str) -> dict:
    """Build an output record: original fields + human label, minus silver_hint."""
    out = {k: v for k, v in rec.items() if k != "silver_hint"}
    out["label"] = label
    out["label_source"] = "human"
    return out


def _atomic_write(path: Path, lines: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        if lines:
            fh.write("\n")
    os.replace(tmp, path)  # atomic on Windows + POSIX


def persist() -> None:
    """Rewrite the resume file (and the final file once everything is labeled)."""
    lines = [
        json.dumps(_record_out(rec, CHOICES[rec["id"]]), ensure_ascii=False)
        for rec in RECORDS
        if rec["id"] in CHOICES
    ]
    _atomic_write(PARTIAL_PATH, lines)
    if len(CHOICES) == len(RECORDS) and RECORDS:
        _atomic_write(OUTPUT_PATH, lines)


def counts() -> dict:
    c = {lab: 0 for lab in LABELS}
    for lab in CHOICES.values():
        c[lab] = c.get(lab, 0) + 1
    return c


def state_payload() -> dict:
    # NOTE: silver_hint is deliberately excluded here.
    return {
        "total": len(RECORDS),
        "labeled": len(CHOICES),
        "counts": counts(),
        "done": len(CHOICES) == len(RECORDS) and bool(RECORDS),
        "output_path": str(OUTPUT_PATH),
        "records": [
            {"id": r["id"], "text": r["text"], "source": r.get("source", ""),
             "date": r.get("date", ""), "label": CHOICES.get(r["id"])}
            for r in RECORDS
        ],
    }


# --- HTTP handler ------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence per-request logging
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/state":
            with LOCK:
                self._json(state_payload())
        elif parsed.path == "/api/hint":
            # Hint is served ONLY on explicit request (after a choice exists).
            rid = parse_qs(parsed.query).get("id", [""])[0]
            rec = ID_TO_REC.get(rid)
            if rec is None:
                self._json({"error": "unknown id"}, 404)
            else:
                self._json({"id": rid, "silver_hint": rec.get("silver_hint")})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)

        if parsed.path == "/api/label":
            rid, label = data.get("id"), data.get("label")
            if rid not in ID_TO_REC or label not in LABELS:
                return self._json({"error": "invalid id or label"}, 400)
            with LOCK:
                CHOICES[rid] = label
                persist()
                self._json(state_payload())
        elif parsed.path == "/api/unset":
            rid = data.get("id")
            with LOCK:
                CHOICES.pop(rid, None)
                persist()
                self._json(state_payload())
        else:
            self._json({"error": "not found"}, 404)


# --- Front-end (single inline HTML page) ------------------------------------
PAGE = r"""<!doctype html>
<html lang="sr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SerbFin gold labeling</title>
<style>
  :root{ --pos:#1a7f37; --neg:#cf222e; --neu:#57606a; --bg:#0d1117; --card:#161b22;
         --fg:#e6edf3; --muted:#8b949e; --line:#30363d; }
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;
       background:var(--bg);color:var(--fg);}
  header{padding:14px 22px;border-bottom:1px solid var(--line);display:flex;
         gap:22px;align-items:center;flex-wrap:wrap;position:sticky;top:0;
         background:var(--bg);z-index:5}
  .bar{flex:1;min-width:200px;height:10px;background:var(--card);border-radius:6px;overflow:hidden}
  .bar>div{height:100%;background:#2f81f7;width:0%}
  .pill{padding:3px 10px;border-radius:20px;font-size:14px;border:1px solid var(--line)}
  .pill b{font-variant-numeric:tabular-nums}
  .cpos{color:var(--pos)} .cneg{color:var(--neg)} .cneu{color:var(--neu)}
  main{max-width:880px;margin:0 auto;padding:36px 22px}
  .meta{color:var(--muted);font-size:14px;margin-bottom:10px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;
        padding:48px 40px;min-height:200px;display:flex;align-items:center;justify-content:center}
  .headline{font-size:34px;line-height:1.35;font-weight:600;text-align:center}
  .btns{display:flex;gap:14px;margin-top:28px;justify-content:center;flex-wrap:wrap}
  button{font-size:18px;padding:16px 26px;border-radius:12px;border:2px solid var(--line);
         background:#21262d;color:var(--fg);cursor:pointer;transition:.08s}
  button:hover{filter:brightness(1.15)}
  button:disabled{opacity:.4;cursor:not-allowed}
  .b-pos{border-color:var(--pos)} .b-neg{border-color:var(--neg)} .b-neu{border-color:var(--neu)}
  .b-pos.sel{background:var(--pos)} .b-neg.sel{background:var(--neg)} .b-neu.sel{background:var(--neu)}
  .kbd{display:inline-block;min-width:20px;padding:1px 6px;margin-right:8px;border:1px solid var(--line);
       border-radius:5px;font-size:13px;background:#0d1117}
  .nav{display:flex;gap:12px;margin-top:26px;justify-content:center;align-items:center;flex-wrap:wrap}
  .nav button{font-size:15px;padding:9px 16px}
  .hintbox{margin-top:18px;text-align:center;min-height:26px;color:var(--muted);font-size:15px}
  .hintval{font-weight:700}
  label.auto{font-size:14px;color:var(--muted);user-select:none;display:flex;gap:6px;align-items:center}
  .done{text-align:center;font-size:18px;color:var(--pos);margin-top:22px}
  a{color:#2f81f7}
</style></head>
<body>
<header>
  <div style="font-weight:700">SerbFin gold labeling</div>
  <div class="bar"><div id="prog"></div></div>
  <div class="pill">Progress <b id="count">0/0</b></div>
  <div class="pill cpos">pos <b id="cpos">0</b></div>
  <div class="pill cneu">neu <b id="cneu">0</b></div>
  <div class="pill cneg">neg <b id="cneg">0</b></div>
</header>
<main>
  <div class="meta"><span id="pos-idx"></span> · <span id="src"></span> · <span id="date"></span></div>
  <div class="card"><div class="headline" id="headline">Loading…</div></div>

  <div class="btns">
    <button class="b-pos" id="bpos" onclick="choose('positive')"><span class="kbd">1</span>positive</button>
    <button class="b-neu" id="bneu" onclick="choose('neutral')"><span class="kbd">2</span>neutral</button>
    <button class="b-neg" id="bneg" onclick="choose('negative')"><span class="kbd">3</span>negative</button>
  </div>

  <div class="hintbox" id="hintbox">
    <button id="hintbtn" onclick="revealHint()" disabled><span class="kbd">h</span>reveal hint</button>
  </div>

  <div class="nav">
    <button onclick="go(-1)"><span class="kbd">,</span>prev</button>
    <button onclick="nextUnlabeled()">next unlabeled</button>
    <button onclick="go(1)">next<span class="kbd" style="margin:0 0 0 8px">.</span></button>
    <label class="auto"><input type="checkbox" id="auto"> auto-advance after choosing</label>
  </div>

  <div class="done" id="done"></div>
</main>
<script>
let S = null;        // server state
let i = 0;           // current index

async function api(path, opts){ const r = await fetch(path, opts); return r.json(); }

function render(){
  const r = S.records[i];
  document.getElementById('headline').textContent = r.text;
  document.getElementById('pos-idx').textContent = 'Item ' + (i+1) + ' / ' + S.total;
  document.getElementById('src').textContent = r.source || '';
  document.getElementById('date').textContent = r.date || '';
  for (const [lab,id] of [['positive','bpos'],['neutral','bneu'],['negative','bneg']])
    document.getElementById(id).classList.toggle('sel', r.label === lab);
  // Hint reveal only allowed once a choice exists for THIS item.
  const hb = document.getElementById('hintbtn');
  hb.disabled = !r.label;
  document.getElementById('hintbox').querySelectorAll('.hintval').forEach(e=>e.remove());
  updateStats();
}

function updateStats(){
  document.getElementById('count').textContent = S.labeled + '/' + S.total;
  document.getElementById('cpos').textContent = S.counts.positive;
  document.getElementById('cneu').textContent = S.counts.neutral;
  document.getElementById('cneg').textContent = S.counts.negative;
  document.getElementById('prog').style.width = (100*S.labeled/S.total).toFixed(1)+'%';
  document.getElementById('done').textContent = S.done
     ? '✓ All ' + S.total + ' labeled — written to ' + S.output_path : '';
}

async function choose(label){
  const r = S.records[i];
  S = await api('/api/label', {method:'POST', headers:{'Content-Type':'application/json'},
                               body: JSON.stringify({id:r.id, label})});
  if (document.getElementById('auto').checked && i < S.total-1){ i++; render(); }
  else render();
}

async function revealHint(){
  const r = S.records[i];
  if (!r.label) return;            // guard: only after choosing
  const h = await api('/api/hint?id='+encodeURIComponent(r.id));
  const box = document.getElementById('hintbox');
  box.querySelectorAll('.hintval').forEach(e=>e.remove());
  const span = document.createElement('span');
  span.className = 'hintval';
  span.textContent = '  silver hint: ' + (h.silver_hint || '—') +
                     (h.silver_hint===r.label ? '  (match)' : '  (differs)');
  span.style.color = h.silver_hint===r.label ? 'var(--pos)' : 'var(--neg)';
  box.appendChild(span);
}

function go(d){ const n=i+d; if(n>=0 && n<S.total){ i=n; render(); } }
function nextUnlabeled(){
  for (let k=0;k<S.total;k++){ const j=(i+1+k)%S.total; if(!S.records[j].label){ i=j; render(); return; } }
}

document.addEventListener('keydown', e=>{
  if (e.key==='1') choose('positive');
  else if (e.key==='2') choose('neutral');
  else if (e.key==='3') choose('negative');
  else if (e.key==='h') revealHint();
  else if (e.key==='.' || e.key==='ArrowRight') go(1);
  else if (e.key===',' || e.key==='ArrowLeft') go(-1);
});

(async function init(){
  S = await api('/api/state');
  // Start at first unlabeled item.
  i = S.records.findIndex(r=>!r.label);
  if (i < 0) i = 0;
  render();
})();
</script>
</body></html>"""


def main() -> None:
    load_records()
    done = len(CHOICES)
    print(f"Loaded {len(RECORDS)} records from {INPUT_PATH.name}")
    if done:
        print(f"Resuming — {done}/{len(RECORDS)} already labeled.")
    if len(CHOICES) == len(RECORDS) and RECORDS:
        persist()
        print(f"All records already labeled. {OUTPUT_PATH} is up to date.")
    url = f"http://{HOST}:{PORT}/"
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"\n  Labeling UI:  {url}")
    print("  Keys: 1=positive  2=neutral  3=negative  |  ,/. = prev/next  |  h = reveal hint")
    print("  Choices save automatically. Press Ctrl+C here to stop.\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Progress saved to", PARTIAL_PATH.name,
              "— relaunch to resume." if len(CHOICES) < len(RECORDS) else "")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
