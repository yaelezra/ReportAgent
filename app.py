"""
ReportAgent Dashboard
Usage:
    pip install -r requirements.txt
    python app.py
Open http://localhost:5000
"""

import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_from_directory, render_template_string
import scipy.io as sio
import numpy as np
import os, traceback, uuid, threading, ctypes
from smolagents import tool, CodeAgent, InferenceClientModel

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

session = {
    "file_paths":    [],
    "hf_token":      "",
    "anthropic_key": "",
    "model_provider":"huggingface",
    "model_id":      "meta-llama/Llama-3.3-70B-Instruct",
    "chat_history":  [],
}

agent_state = {"thread_id": None}

DEFAULT_INSTRUCTIONS = """\
- You are a senior Data Scientist.
- You know how to analyze big data, give insights, conclusions, and make graphs and reports.
- Always call list_all_features first to understand what fields are available.
- If you are not sure of a field name, use find_feature_by_description before trying to load it.
- In your final answer, tell which tools you used to answer the question.
- Elaborate your final answer as much as you can.
- If you encounter an error that you can't resolve, stop and print the error as your response.
- If the user asks you for a report on something you should make it in HTML format and save it.
- If you make plots, graphs or reports, show them.
- Always debug your code before executing.\
"""

REPORT_PROMPT = """\
Generate a comprehensive data science report for all loaded files. Include:
1. Overview of all parameters and their descriptions
2. Key statistics (mean, std, min, max) for every parameter
3. Correlation analysis between parameters — include a correlation matrix plot
4. If multiple files are loaded: compare the same parameters across files with overlay plots
5. A written summary of the most important findings and patterns
Make a separate plot for each insight. Save each one individually.\
"""

class AgentStopped(Exception):
    pass

def _stop_agent_thread():
    tid = agent_state.get("thread_id")
    if tid:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid), ctypes.py_object(AgentStopped)
        )
        agent_state["thread_id"] = None

_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder

# ── Tools (matching notebook) ─────────────────────────────────────────────────

@tool
def load_mat_file(file_path: str) -> np.array:
    """Loads a MATLAB .mat file and returns the sequence_data struct.
    Args:
      file_path: Path to the .mat file.
    """
    mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    return mat["sequence_data"]

@tool
def load_sequence(file_path: str, param_name: str) -> np.array:
    """Returns the array for a specific parameter from the .mat file.
    Args:
      file_path: Path to the .mat file.
      param_name: The parameter name to extract.
    """
    mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    return getattr(mat["sequence_data"], param_name)

@tool
def load_doc(file_path: str) -> dict:
    """Returns documentation for all parameters in the .mat file.
    Args:
      file_path: Path to the .mat file.
    """
    mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    seq = mat["sequence_data"]
    return {f: str(getattr(seq.documentation, f)) for f in seq.documentation._fieldnames}

@tool
def list_all_features(file_path: str) -> dict:
    """Returns all numeric field names and their descriptions. Call this first.
    Args:
      file_path: Path to the .mat file.
    """
    mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    seq = mat["sequence_data"]
    doc = {f: str(getattr(seq.documentation, f)) for f in seq.documentation._fieldnames}
    return {
        f: doc.get(f, "")
        for f in seq._fieldnames
        if f != "documentation"
        and isinstance(getattr(seq, f), np.ndarray)
        and np.issubdtype(getattr(seq, f).dtype, np.number)
    }

@tool
def find_feature_by_description(file_path: str, query: str, top_k: int = 3) -> dict:
    """Finds the most semantically similar features to the query.
    Args:
      file_path: Path to the .mat file.
      query: Natural language description of the feature.
      top_k: Number of top matches to return (default 3).
    """
    from sentence_transformers import util as st_util
    mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    seq = mat["sequence_data"]
    fields, corpus = [], []
    for f in seq._fieldnames:
        if f == "documentation":
            continue
        desc = str(getattr(seq.documentation, f)) if f in seq.documentation._fieldnames else ""
        fields.append(f)
        corpus.append(f"{f}: {desc}")
    emb   = get_embedder()
    q_emb = emb.encode(query, convert_to_tensor=True)
    c_emb = emb.encode(corpus, convert_to_tensor=True)
    scores = st_util.cos_sim(q_emb, c_emb)[0]
    top    = scores.topk(min(top_k, len(fields)))
    return {
        fields[idx]: {"description": corpus[idx], "similarity": round(float(s), 3)}
        for s, idx in zip(top.values, top.indices)
    }

@tool
def find_feature_in_files(file_paths: list, param_name: str) -> list:
    """Find the same feature across different files.
    Args:
      file_paths: List of .mat file paths.
      param_name: The parameter name to extract.
    """
    results = []
    for path in file_paths:
        mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
        results.append(getattr(mat["sequence_data"], param_name).tolist())
    return results

# ── Agent ─────────────────────────────────────────────────────────────────────

def build_model():
    if session["model_provider"] == "anthropic":
        from smolagents import AnthropicModel
        os.environ["ANTHROPIC_API_KEY"] = session["anthropic_key"]
        return AnthropicModel(model_id=session["model_id"] or "claude-sonnet-4-5")
    return InferenceClientModel(
        model_id=session["model_id"] or "meta-llama/Llama-3.3-70B-Instruct",
        token=session["hf_token"],
    )

def run_prompt(question, instructions, plot_dir):
    file_paths = session["file_paths"]
    agent_state["thread_id"] = threading.current_thread().ident
    try:
        agent = CodeAgent(
            tools=[load_mat_file, load_sequence, load_doc, list_all_features,
                   find_feature_by_description, find_feature_in_files],
            model=build_model(),
            additional_authorized_imports=[
                "numpy", "pandas", "seaborn", "matplotlib", "matplotlib.pyplot", "math", "os",
            ],
        )
        history = session["chat_history"][-6:]  # last 6 exchanges
        history_str = "\n".join(f"Q: {h['q']}\nA: {h['a']}" for h in history) if history else "None."

        prompt = f"""
You are given a list of files — each file contains a struct where each field is a vector of values in time.
File paths: {file_paths}

CONVERSATION HISTORY (use this to answer follow-up questions):
{history_str}

SYSTEM INSTRUCTIONS TO FOLLOW STRICTLY:
{instructions}

PLOTTING RULES — follow exactly:
- Start every code block that makes plots with:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os
- Save each plot right after creating it:
    plt.savefig(os.path.join("{plot_dir}", "plot_N.png"), dpi=120, bbox_inches="tight")
  where N is 1, 2, 3, ... (increment for each plot)
- NEVER call plt.show()
- Call plt.close() after every savefig

HTML REPORT RULES:
- If asked for an HTML report, save it to: os.path.join("{plot_dir}", "report.html")
- The HTML report should be self-contained and well styled.

USER REQUEST:
{question}
"""
        response = agent.run(prompt)
        run_id = os.path.basename(plot_dir)
        files  = os.listdir(plot_dir)
        plots  = sorted([
            f"/plots/{run_id}/{f}"
            for f in files
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg"))
        ])
        html_reports = sorted([
            f"/plots/{run_id}/{f}"
            for f in files
            if f.lower().endswith(".html")
        ])
        return {"answer": str(response), "plots": plots, "html_reports": html_reports}
    except AgentStopped:
        return {"stopped": True}
    except Exception:
        return {"error": traceback.format_exc()}
    finally:
        agent_state["thread_id"] = None

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files"}), 400
    added = []
    for f in files:
        path = os.path.join(UPLOAD_DIR, f.filename)
        f.save(path)
        if path not in session["file_paths"]:
            session["file_paths"].append(path)
        try:
            mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
            seq = mat["sequence_data"]
            doc = {field: str(getattr(seq.documentation, field)) for field in seq.documentation._fieldnames}
            params = [
                {"name": field, "description": doc.get(field, "")}
                for field in seq._fieldnames
                if field != "documentation"
                and isinstance(getattr(seq, field), np.ndarray)
                and np.issubdtype(getattr(seq, field).dtype, np.number)
            ]
            added.append({"filename": f.filename, "path": path, "params": params})
        except Exception:
            added.append({"filename": f.filename, "path": path, "params": [], "error": "Could not parse"})
    return jsonify({"ok": True, "added": added, "total": len(session["file_paths"])})

@app.route("/remove_file", methods=["POST"])
def remove_file():
    path = (request.json or {}).get("path")
    if path in session["file_paths"]:
        session["file_paths"].remove(path)
    return jsonify({"ok": True, "total": len(session["file_paths"])})

@app.route("/configure", methods=["POST"])
def configure():
    d = request.json or {}
    session.update({
        "model_provider": d.get("model_provider", "huggingface"),
        "hf_token":       d.get("hf_token", ""),
        "anthropic_key":  d.get("anthropic_key", ""),
        "model_id":       d.get("model_id", ""),
    })
    return jsonify({"ok": True})

@app.route("/ask", methods=["POST"])
def ask():
    d = request.json or {}
    q = d.get("question", "").strip()
    if not session["file_paths"]: return jsonify({"error": "Upload at least one file first"}), 400
    if not q:                      return jsonify({"error": "Question is empty"}), 400
    run_id   = uuid.uuid4().hex[:8]
    plot_dir = os.path.join(PLOTS_DIR, run_id)
    os.makedirs(plot_dir, exist_ok=True)
    result = run_prompt(q, d.get("instructions", DEFAULT_INSTRUCTIONS), plot_dir)
    if "answer" in result:
        session["chat_history"].append({"q": q, "a": result["answer"]})
    return jsonify(result)

@app.route("/report", methods=["POST"])
def report():
    d = request.json or {}
    if not session["file_paths"]: return jsonify({"error": "Upload at least one file first"}), 400
    history = "\n".join(f"Q: {h['q']}\nA: {h['a']}" for h in session["chat_history"]) or "No prior conversation."
    q = REPORT_PROMPT + f"\n\nContext from prior conversation:\n{history}"
    run_id   = uuid.uuid4().hex[:8]
    plot_dir = os.path.join(PLOTS_DIR, run_id)
    os.makedirs(plot_dir, exist_ok=True)
    result = run_prompt(q, d.get("instructions", DEFAULT_INSTRUCTIONS), plot_dir)
    if "answer" in result:
        session["chat_history"].append({"q": "📊 Full Report", "a": result["answer"]})
    return jsonify(result)

@app.route("/stop", methods=["POST"])
def stop():
    _stop_agent_thread()
    return jsonify({"ok": True})

@app.route("/plots/<run_id>/<filename>")
def serve_plot(run_id, filename):
    return send_from_directory(os.path.join(PLOTS_DIR, run_id), filename)

# ── HTML Dashboard ────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ReportAgent</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --paper:#faf9f5;--bg:#f2ede6;--bg2:#ebe5dc;
  --ink:#111;--ink2:#444;--muted:#999;
  --line:#e2dbd0;--line-s:#cec7bc;
  --accent:#d86e3b;--blue:#5b7cdb;--green:#3fa76c;--red:#d94f38;
  --r:10px;
}
html,body{height:100%;font-family:'Geist',system-ui,sans-serif;font-size:14px;color:var(--ink);background:var(--bg)}
.shell{display:grid;grid-template-rows:52px 1fr;height:100vh;overflow:hidden}

/* ── Top bar ── */
.topbar{
  display:flex;align-items:center;gap:14px;
  padding:0 20px;background:var(--paper);
  border-bottom:1px solid var(--line);z-index:20;
}
.logo{font-size:16px;font-weight:700;letter-spacing:-.02em;margin-right:4px}
.logo em{color:var(--accent);font-style:normal}
.divider{width:1px;height:20px;background:var(--line);margin:0 4px}
.stat-chip{
  display:flex;align-items:center;gap:5px;
  padding:3px 10px;border-radius:99px;
  border:1px solid var(--line-s);background:var(--bg);
  font-size:11px;color:var(--ink2);
}
.stat-chip .dot{width:6px;height:6px;border-radius:50%;background:var(--muted)}
.stat-chip .dot.on{background:var(--green)}
.topbar-right{margin-left:auto;display:flex;gap:8px}
.tb-btn{
  appearance:none;border:1px solid var(--line-s);border-radius:var(--r);
  background:var(--paper);color:var(--ink2);padding:6px 14px;
  font:inherit;font-size:12px;cursor:pointer;transition:background .15s;
}
.tb-btn:hover{background:var(--bg2)}
.tb-btn.accent{background:var(--accent);color:#fff;border-color:var(--accent)}
.tb-btn.accent:hover{opacity:.88}
.tb-btn.accent:disabled{opacity:.45;cursor:default}
.tb-btn.danger{border-color:var(--red);color:var(--red);display:none}
.tb-btn.danger:hover{background:#fdf0ee}
.tb-btn.danger.visible{display:block}

/* ── Body grid ── */
.body{display:grid;grid-template-columns:260px 1fr 0px;overflow:hidden;transition:grid-template-columns .25s ease}
.body.panel-open{grid-template-columns:260px 1fr 320px}

/* ── Sidebar ── */
.side{border-right:1px solid var(--line);background:var(--paper);display:flex;flex-direction:column;overflow:hidden}
.side-hd{padding:14px 16px 10px;border-bottom:1px solid var(--line)}
.side-hd h3{font-size:11px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}

/* Upload */
.upload-zone{
  border:1.5px dashed var(--line-s);border-radius:var(--r);
  padding:12px;text-align:center;cursor:pointer;
  transition:border-color .15s,background .15s;position:relative;
}
.upload-zone:hover{border-color:var(--accent);background:var(--bg)}
.upload-zone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%}
.upload-zone .u-ico{font-size:18px;color:var(--muted);margin-bottom:3px}
.upload-zone .u-lbl{font-size:12px;color:var(--ink2);font-weight:500}
.upload-zone .u-sub{font-size:11px;color:var(--muted)}

/* File chips */
.file-chips{display:flex;flex-direction:column;gap:5px;margin-top:8px}
.file-chip{
  display:flex;align-items:center;gap:7px;
  padding:6px 9px;border-radius:8px;background:var(--bg);
  font-size:11.5px;color:var(--ink2);border:1px solid var(--line);
}
.file-chip .fc-dot{width:5px;height:5px;border-radius:50%;background:var(--green);flex-shrink:0}
.file-chip .fc-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:'Geist Mono',monospace}
.file-chip .fc-rm{appearance:none;border:0;background:transparent;color:var(--muted);cursor:pointer;font-size:13px;flex-shrink:0;line-height:1;padding:0}
.file-chip .fc-rm:hover{color:var(--red)}

/* Param browser */
.param-browser{flex:1;overflow:hidden;display:flex;flex-direction:column}
.param-search-wrap{padding:10px 14px 6px}
.param-search{
  width:100%;border:1px solid var(--line-s);border-radius:8px;
  background:var(--bg);padding:7px 10px;font:inherit;font-size:12px;
  color:var(--ink);outline:none;
}
.param-search:focus{border-color:var(--accent)}
.param-search::placeholder{color:var(--muted)}
.param-list{flex:1;overflow-y:auto;padding:2px 0 8px}
.param-group-hd{
  padding:8px 14px 4px;font-size:10px;font-weight:600;
  letter-spacing:.08em;text-transform:uppercase;color:var(--muted);
  position:sticky;top:0;background:var(--paper);
}
.param-row{
  display:flex;flex-direction:column;gap:1px;
  padding:6px 14px;cursor:default;transition:background .1s;
}
.param-row:hover{background:var(--bg)}
.param-row .pr-name{font-size:11.5px;font-weight:500;color:var(--ink2);font-family:'Geist Mono',monospace}
.param-row .pr-desc{font-size:11px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.param-empty{padding:20px 14px;font-size:12px;color:var(--muted);text-align:center;line-height:1.6}

/* ── Center ── */
.center{display:grid;grid-template-rows:1fr auto;overflow:hidden;background:var(--paper)}

/* Graph viewer */
.viewer{position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden;background:var(--paper)}
.ph{display:flex;flex-direction:column;align-items:center;gap:12px;color:var(--muted);text-align:center;padding:40px;pointer-events:none}
.ph-ico{font-size:48px;opacity:.3}
.ph p{font-size:13px;line-height:1.7;max-width:280px}

.graph-wrap{display:none;width:100%;height:100%;flex-direction:column}
.graph-wrap.on{display:flex}

.graph-toolbar{
  display:flex;align-items:center;gap:10px;
  padding:10px 20px;border-bottom:1px solid var(--line);
  background:var(--paper);flex-shrink:0;
}
.graph-toolbar .g-label{
  font-size:13px;font-weight:500;color:var(--ink);flex:1;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.graph-toolbar .g-counter{font-size:12px;color:var(--muted);font-family:'Geist Mono',monospace}
.graph-toolbar .g-tag{
  font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  color:var(--accent);padding:2px 8px;border-radius:99px;
  border:1px solid color-mix(in srgb, var(--accent) 30%, transparent);
  background:color-mix(in srgb, var(--accent) 6%, transparent);
}

.graph-stage{flex:1;position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden;padding:16px 60px 44px}
.graph-stage img{max-width:100%;max-height:100%;object-fit:contain;border-radius:8px;box-shadow:0 2px 16px rgba(0,0,0,.07)}

.nav-btn{
  position:absolute;top:50%;transform:translateY(-50%);
  width:38px;height:38px;border-radius:50%;
  background:var(--paper);border:1px solid var(--line-s);
  color:var(--ink2);font-size:18px;display:grid;place-items:center;
  cursor:pointer;box-shadow:0 2px 10px rgba(0,0,0,.08);
  appearance:none;z-index:4;transition:transform .15s,background .15s;
}
.nav-btn:hover{background:var(--bg);transform:translateY(-50%) scale(1.08)}
.nav-btn:disabled{opacity:.2;pointer-events:none}
.nav-btn.left{left:12px}
.nav-btn.right{right:12px}

.pager{position:absolute;bottom:12px;left:0;right:0;display:flex;justify-content:center;gap:5px;pointer-events:none}
.pager-dot{
  appearance:none;border:0;height:3px;border-radius:99px;
  background:rgba(0,0,0,.12);padding:0;width:18px;
  pointer-events:auto;cursor:pointer;transition:all .15s;
}
.pager-dot.on{background:var(--accent);width:30px}
.pager-dot:hover:not(.on){background:rgba(0,0,0,.28)}

/* ── Chat bar ── */
.chat-bar{border-top:1px solid var(--line);background:var(--paper)}
.chat-history{max-height:140px;overflow-y:auto;padding:10px 20px;display:flex;flex-direction:column;gap:8px}
.msg{display:flex;flex-direction:column;gap:2px}
.msg .who{font-size:10px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
.msg.user .who{color:var(--accent)}
.msg .text{font-size:13px;color:var(--ink2);line-height:1.5}
.thinking{display:inline-flex;gap:3px;align-items:center}
.thinking i{width:4px;height:4px;border-radius:50%;background:var(--muted);animation:blink 1.1s infinite;font-style:normal}
.thinking i:nth-child(2){animation-delay:.15s}.thinking i:nth-child(3){animation-delay:.3s}
@keyframes blink{0%,80%,100%{opacity:.15}40%{opacity:1}}

.composer{display:flex;align-items:center;gap:8px;padding:10px 16px 14px}
.composer input{
  flex:1;border:1px solid var(--line-s);border-radius:var(--r);
  background:var(--bg);padding:10px 14px;font:inherit;font-size:13px;
  color:var(--ink);outline:none;transition:border-color .15s;
}
.composer input:focus{border-color:var(--accent)}
.composer input::placeholder{color:var(--muted)}
.c-btn{
  appearance:none;border-radius:var(--r);font:inherit;font-size:13px;
  font-weight:500;cursor:pointer;padding:10px 16px;white-space:nowrap;
  transition:opacity .15s,background .15s;
}
.c-btn.send{border:0;background:var(--accent);color:#fff}
.c-btn.send:hover{opacity:.88}
.c-btn.send:disabled{opacity:.4;cursor:default}
.c-btn.report{border:1px solid var(--blue);background:transparent;color:var(--blue)}
.c-btn.report:hover{background:color-mix(in srgb, var(--blue) 8%, transparent)}
.c-btn.report:disabled{opacity:.4;cursor:default}
.c-btn.stop{border:1px solid var(--red);background:transparent;color:var(--red);display:none}
.c-btn.stop:hover{background:color-mix(in srgb, var(--red) 6%, transparent)}
.c-btn.stop.visible{display:block}

/* ── Right panel ── */
.rpanel{
  border-left:1px solid var(--line);background:var(--paper);
  display:flex;flex-direction:column;overflow:hidden;
  min-width:0;
}
.rpanel-hd{
  display:flex;align-items:center;justify-content:space-between;
  padding:14px 16px;border-bottom:1px solid var(--line);flex-shrink:0;
}
.rpanel-hd h3{font-size:11px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.rpanel-close{appearance:none;border:0;background:transparent;color:var(--muted);font-size:16px;cursor:pointer}
.rpanel-close:hover{color:var(--ink)}
.rpanel-body{flex:1;overflow-y:auto;padding:16px}
.rpanel-empty{color:var(--muted);font-size:12px;text-align:center;padding:30px 16px;line-height:1.6}
.answer-block{font-size:13px;line-height:1.65;color:var(--ink2)}
.answer-block + .answer-block{margin-top:16px;padding-top:16px;border-top:1px solid var(--line)}
.answer-q{font-size:11px;font-weight:600;color:var(--accent);margin-bottom:6px;letter-spacing:.03em}

/* ── Settings overlay ── */
.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:100;align-items:center;justify-content:center}
.overlay.open{display:flex}
.sbox{background:var(--paper);border-radius:16px;padding:28px;width:460px;max-height:90vh;overflow-y:auto;box-shadow:0 24px 64px rgba(0,0,0,.18)}
.sbox h3{font-size:16px;font-weight:600;margin-bottom:20px}
.field{display:flex;flex-direction:column;gap:5px;margin-bottom:14px}
.field label{font-size:10.5px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.field input,.field select,.field textarea{
  border:1px solid var(--line-s);border-radius:8px;background:var(--bg);
  padding:9px 12px;font:inherit;font-size:13px;color:var(--ink);outline:none;
  transition:border-color .15s;
}
.field input:focus,.field select:focus,.field textarea:focus{border-color:var(--accent)}
.field textarea{resize:vertical;min-height:100px;line-height:1.5}
.prov-hf,.prov-ant{display:none}
.prov-hf.show,.prov-ant.show{display:contents}
.sbox-actions{display:flex;justify-content:flex-end;gap:10px;margin-top:20px}
.sbox-cancel{appearance:none;border:1px solid var(--line-s);border-radius:8px;background:transparent;padding:8px 16px;font:inherit;font-size:13px;color:var(--ink2);cursor:pointer}
.sbox-save{appearance:none;border:0;border-radius:8px;background:var(--accent);color:#fff;padding:8px 20px;font:inherit;font-size:13px;font-weight:500;cursor:pointer}
.sbox-save:hover{opacity:.88}

::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--line-s);border-radius:99px}
</style>
</head>
<body>
<div class="shell">

<!-- ── Top bar ── -->
<header class="topbar">
  <div class="logo">Report<em>Agent</em></div>
  <div class="divider"></div>
  <div class="stat-chip"><span class="dot" id="files-dot"></span><span id="files-label">No files loaded</span></div>
  <div class="stat-chip" id="model-chip" style="display:none"><span class="dot on"></span><span id="model-label">—</span></div>
  <div class="topbar-right">
    <button class="tb-btn danger" id="stop-tb" onclick="stopAgent()">⏹ Stop</button>
    <button class="tb-btn accent" id="report-tb" onclick="generateReport()" disabled>📊 Full Report</button>
    <button class="tb-btn" onclick="openSettings()">⚙ Settings</button>
  </div>
</header>

<!-- ── Body ── -->
<div class="body" id="body">

  <!-- Sidebar -->
  <aside class="side">
    <div class="side-hd">
      <h3>Files</h3>
      <label class="upload-zone">
        <input type="file" id="file-input" accept=".mat" multiple onchange="uploadFiles(this)">
        <div class="u-ico">📂</div>
        <div class="u-lbl">Upload .mat files</div>
        <div class="u-sub">Click or drag — multiple OK</div>
      </label>
      <div class="file-chips" id="file-chips"></div>
    </div>
    <div class="param-browser">
      <div class="param-search-wrap">
        <h3 style="font-size:11px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin-bottom:8px">Parameters</h3>
        <input class="param-search" id="param-search" placeholder="Search parameters…" oninput="filterParams()">
      </div>
      <div class="param-list" id="param-list">
        <div class="param-empty">Upload a file to browse parameters</div>
      </div>
    </div>
  </aside>

  <!-- Center -->
  <div class="center">
    <!-- Graph viewer -->
    <div class="viewer" id="viewer">
      <div class="ph" id="ph">
        <div class="ph-ico">📊</div>
        <p>Load your files and ask the agent anything.<br>Graphs will appear here with arrow navigation.</p>
      </div>
      <div class="graph-wrap" id="graph-wrap">
        <div class="graph-toolbar">
          <span class="g-tag">Plot</span>
          <span class="g-label" id="g-label"></span>
          <span class="g-counter" id="g-counter"></span>
        </div>
        <div class="graph-stage">
          <img id="graph-img" src="" alt="">
          <button class="nav-btn left"  id="nav-prev" onclick="navigate(-1)">‹</button>
          <button class="nav-btn right" id="nav-next" onclick="navigate(1)">›</button>
          <div class="pager" id="pager"></div>
        </div>
      </div>
    </div>

    <!-- Chat bar -->
    <div class="chat-bar">
      <div class="chat-history" id="chat-history"></div>
      <div class="composer">
        <input id="q-input" type="text"
          placeholder="Ask about your data… e.g. "Show correlation matrix""
          onkeydown="if(event.key==='Enter')sendQuestion()">
        <button class="c-btn report" id="report-btn" onclick="generateReport()" disabled>📊 Report</button>
        <button class="c-btn send"   id="send-btn"   onclick="sendQuestion()" disabled>Send →</button>
        <button class="c-btn stop"   id="stop-btn"   onclick="stopAgent()">⏹ Stop</button>
      </div>
    </div>
  </div>

  <!-- Right panel -->
  <div class="rpanel" id="rpanel">
    <div class="rpanel-hd">
      <h3>Analysis</h3>
      <button class="rpanel-close" onclick="closePanel()">✕</button>
    </div>
    <div class="rpanel-body" id="rpanel-body">
      <div class="rpanel-empty">Agent responses will appear here</div>
    </div>
  </div>

</div>
</div>

<!-- Settings overlay -->
<div class="overlay" id="overlay" onclick="if(event.target===this)closeSettings()">
  <div class="sbox">
    <h3>Settings</h3>
    <div class="field">
      <label>Model Provider</label>
      <select id="model-provider" onchange="toggleProv()">
        <option value="huggingface">Hugging Face</option>
        <option value="anthropic">Anthropic (Claude)</option>
      </select>
    </div>
    <div class="prov-hf show" id="prov-hf">
      <div class="field"><label>Hugging Face Token</label><input type="password" id="hf-token" placeholder="hf_…"></div>
      <div class="field"><label>Model ID</label><input type="text" id="hf-model" value="meta-llama/Llama-3.3-70B-Instruct"></div>
    </div>
    <div class="prov-ant" id="prov-ant">
      <div class="field"><label>Anthropic API Key</label><input type="password" id="ant-key" placeholder="sk-ant-…"></div>
      <div class="field"><label>Model ID</label><input type="text" id="ant-model" value="claude-sonnet-4-5"></div>
    </div>
    <div class="field">
      <label>System Instructions</label>
      <textarea id="instructions">- You are a senior Data Scientist.
- You know how to analyze big data, give insights, conclusions, and make graphs and reports.
- Always call list_all_features first to understand what fields are available.
- If you are not sure of a field name, use find_feature_by_description before trying to load it.
- In your final answer, tell which tools you used to answer the question.
- Elaborate your final answer as much as you can.
- If you encounter an error that you can't resolve, stop and print the error as your response.
- If the user asks you for a report on something you should make it in HTML format and save it.
- If you make plots, graphs or reports, show them.
- Always debug your code before executing.</textarea>
    </div>
    <div class="sbox-actions">
      <button class="sbox-cancel" onclick="closeSettings()">Cancel</button>
      <button class="sbox-save" onclick="saveSettings()">Save</button>
    </div>
  </div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────────
let plots   = [];
let current = 0;
let allParams = [];   // [{file, name, desc}]

// ── Settings ───────────────────────────────────────────────────────────────────
function openSettings(){ document.getElementById('overlay').classList.add('open') }
function closeSettings(){ document.getElementById('overlay').classList.remove('open') }
function toggleProv(){
  const v = document.getElementById('model-provider').value;
  document.getElementById('prov-hf').classList.toggle('show', v==='huggingface');
  document.getElementById('prov-ant').classList.toggle('show', v==='anthropic');
}
async function saveSettings(){
  const prov = document.getElementById('model-provider').value;
  const mid  = prov==='anthropic'
    ? document.getElementById('ant-model').value.trim()
    : document.getElementById('hf-model').value.trim();
  await fetch('/configure',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model_provider:prov,hf_token:document.getElementById('hf-token').value.trim(),
      anthropic_key:document.getElementById('ant-key').value.trim(),model_id:mid})});
  const chip = document.getElementById('model-chip');
  chip.style.display='flex';
  document.getElementById('model-label').textContent = mid.split('/').pop();
  closeSettings();
}

// ── Files ──────────────────────────────────────────────────────────────────────
async function uploadFiles(input){
  if(!input.files.length) return;
  const fd = new FormData();
  for(const f of input.files) fd.append('files',f);
  const res  = await fetch('/upload',{method:'POST',body:fd});
  const data = await res.json();
  if(data.error){alert(data.error);return;}
  data.added.forEach(f=>{
    addFileChip(f.filename, f.path);
    if(f.params?.length) addParams(f.filename, f.params);
  });
  updateFilesStatus(data.total);
  input.value='';
}
function addFileChip(name, path){
  const el = document.createElement('div');
  el.className='file-chip'; el.dataset.path=path;
  el.innerHTML=`<span class="fc-dot"></span><span class="fc-name" title="${name}">${name}</span><button class="fc-rm" onclick="removeFile('${path}',this.parentElement)">✕</button>`;
  document.getElementById('file-chips').appendChild(el);
}
async function removeFile(path, el){
  await fetch('/remove_file',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path})});
  el.remove();
  allParams = allParams.filter(p=>p.file!==path);
  renderParams(allParams);
  const total = document.querySelectorAll('.file-chip').length;
  updateFilesStatus(total);
}
function updateFilesStatus(n){
  document.getElementById('files-dot').className='dot'+(n>0?' on':'');
  document.getElementById('files-label').textContent = n===0?'No files loaded': n+' file'+(n!==1?'s':'')+' loaded';
  const hasFiles = n>0;
  document.getElementById('send-btn').disabled   = !hasFiles;
  document.getElementById('report-btn').disabled = !hasFiles;
  document.getElementById('report-tb').disabled  = !hasFiles;
}
function addParams(filename, params){
  params.forEach(p=>allParams.push({file:filename, name:p.name, desc:p.description}));
  renderParams(allParams);
}
function renderParams(params){
  const list  = document.getElementById('param-list');
  const query = document.getElementById('param-search').value.toLowerCase();
  const filtered = query ? params.filter(p=>p.name.toLowerCase().includes(query)||p.desc.toLowerCase().includes(query)) : params;
  if(!filtered.length){list.innerHTML='<div class="param-empty">No parameters match</div>';return;}
  // group by file
  const groups = {};
  filtered.forEach(p=>{ if(!groups[p.file]) groups[p.file]=[]; groups[p.file].push(p); });
  list.innerHTML='';
  Object.entries(groups).forEach(([file, ps])=>{
    const hd=document.createElement('div'); hd.className='param-group-hd'; hd.textContent=file; list.appendChild(hd);
    ps.forEach(p=>{
      const row=document.createElement('div'); row.className='param-row';
      row.innerHTML=`<span class="pr-name">${p.name}</span><span class="pr-desc" title="${p.desc}">${p.desc||'—'}</span>`;
      list.appendChild(row);
    });
  });
}
function filterParams(){ renderParams(allParams); }

// ── Agent ──────────────────────────────────────────────────────────────────────
function setRunning(on){
  document.getElementById('send-btn').disabled   = on;
  document.getElementById('report-btn').disabled = on;
  document.getElementById('report-tb').disabled  = on;
  document.getElementById('stop-btn').classList.toggle('visible', on);
  document.getElementById('stop-tb').classList.toggle('visible', on);
}
function addChatMsg(who, text){
  const h = document.getElementById('chat-history');
  const m = document.createElement('div'); m.className='msg '+who;
  m.innerHTML=`<span class="who">${who==='user'?'You':'Agent'}</span><div class="text">${text}</div>`;
  h.appendChild(m); h.scrollTop=h.scrollHeight; return m;
}
function addThinking(){
  const h = document.getElementById('chat-history');
  const m = document.createElement('div'); m.className='msg agent';
  m.innerHTML=`<span class="who">Agent</span><div class="text"><div class="thinking"><i></i><i></i><i></i></div></div>`;
  h.appendChild(m); h.scrollTop=h.scrollHeight; return m;
}
async function callAgent(label, endpoint, body){
  setRunning(true);
  addChatMsg('user', label);
  const t = addThinking();
  const res  = await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data = await res.json();
  t.remove(); setRunning(false);
  if(data.stopped){ addChatMsg('agent','⏹ Stopped.'); return; }
  if(data.error)  { addChatMsg('agent','⚠ Error — check console.'); console.error(data.error); return; }
  const short = data.answer.length>120 ? data.answer.slice(0,120)+'… (see panel →)' : data.answer;
  addChatMsg('agent', short);
  addToPanel(label, data.answer, data.html_reports || []);
  if(data.plots?.length) showPlots(data.plots, label);
}
async function sendQuestion(){
  const input=document.getElementById('q-input');
  const q=input.value.trim(); if(!q) return;
  input.value='';
  await callAgent(q,'/ask',{question:q,instructions:document.getElementById('instructions').value});
}
async function generateReport(){
  await callAgent('📊 Full Report','/report',{instructions:document.getElementById('instructions').value});
}
async function stopAgent(){
  await fetch('/stop',{method:'POST'});
}

// ── Right panel ────────────────────────────────────────────────────────────────
function openPanel(){ document.getElementById('body').classList.add('panel-open'); }
function closePanel(){ document.getElementById('body').classList.remove('panel-open'); }
function addToPanel(q, answer, htmlReports){
  const body=document.getElementById('rpanel-body');
  const empty=body.querySelector('.rpanel-empty');
  if(empty) empty.remove();
  const block=document.createElement('div'); block.className='answer-block';
  let html=`<div class="answer-q">${q}</div>${answer.replace(/\n/g,'<br>')}`;
  if(htmlReports && htmlReports.length){
    html += '<div style="margin-top:10px;display:flex;flex-direction:column;gap:6px">';
    htmlReports.forEach(r=>{
      const name=r.split('/').pop();
      html+=`<a href="${r}" target="_blank" style="display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--blue);text-decoration:none;padding:6px 10px;border:1px solid var(--blue);border-radius:7px;width:fit-content">📄 Open ${name}</a>`;
    });
    html += '</div>';
  }
  block.innerHTML=html;
  body.appendChild(block); body.scrollTop=body.scrollHeight;
  openPanel();
}

// ── Graph viewer ───────────────────────────────────────────────────────────────
function showPlots(newPlots, label){
  plots=newPlots; current=0;
  document.getElementById('ph').style.display='none';
  document.getElementById('graph-wrap').classList.add('on');
  document.getElementById('g-label').textContent=label.length>60?label.slice(0,57)+'…':label;
  renderSlide();
}
function renderSlide(){
  document.getElementById('graph-img').src=plots[current];
  document.getElementById('g-counter').textContent=(current+1)+' / '+plots.length;
  document.getElementById('nav-prev').disabled=current===0;
  document.getElementById('nav-next').disabled=current===plots.length-1;
  buildDots();
}
function navigate(dir){ current=Math.max(0,Math.min(plots.length-1,current+dir)); renderSlide(); }
function buildDots(){
  const pg=document.getElementById('pager'); pg.innerHTML='';
  plots.forEach((_,i)=>{ const d=document.createElement('button'); d.className='pager-dot'+(i===current?' on':''); d.onclick=()=>{current=i;renderSlide()}; pg.appendChild(d); });
}
document.addEventListener('keydown',e=>{
  if(document.activeElement===document.getElementById('q-input')) return;
  if(e.key==='ArrowLeft') navigate(-1);
  if(e.key==='ArrowRight') navigate(1);
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)
