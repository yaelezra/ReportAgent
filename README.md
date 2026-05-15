# ReportAgent

An AI-powered data science agent that analyzes MATLAB `.mat` files through natural language questions. Upload your files, ask anything, and get back charts, statistics, comparisons, and full HTML reports — no coding required.

---

## What it does

ReportAgent wraps a **ReAct CodeAgent** (from [smolagents](https://github.com/huggingface/smolagents)) around your `.mat` data files. You ask a question in plain English; the agent reasons through it, writes and executes Python analysis code, and returns plots, insights, and reports.

It supports:
- Single-file analysis (statistics, plots, correlations)
- Multi-file comparison (overlay plots, side-by-side stats)
- Semantic feature search — ask for "happiness level" and it finds the right field even if it's called `happinessScore`
- HTML report generation
- Conversation memory — follow-up questions work naturally

---

## Agent architecture

```
User question
      │
      ▼
  run_custom_agent()
      │
      ├── Conversation history (last 6 exchanges)
      ├── File paths
      ├── System instructions
      └── User request
              │
              ▼
         CodeAgent  ◄──── LLM (Llama / Claude)
              │
        Thinks → Acts → Observes → Repeats
              │
      ┌───────┴────────────────────────────┐
      │         Tools available            │
      ├── load_mat_file                    │
      ├── load_sequence                    │
      ├── load_doc                         │
      ├── list_all_features               │
      ├── find_feature_by_description (RAG)│
      └── find_feature_in_files           │
              │
              ▼
      Final answer + plots + HTML report
```

---

## Tools

| Tool | What it does |
|------|-------------|
| `load_mat_file` | Loads a `.mat` file and returns the full `sequence_data` struct |
| `load_sequence` | Returns the time-series array for a specific parameter by exact name |
| `load_doc` | Returns the documentation dictionary for all parameters in a file |
| `list_all_features` | Returns all numeric field names with descriptions — agent calls this first |
| `find_feature_by_description` | **RAG tool** — finds the closest matching field using semantic embedding similarity (`all-MiniLM-L6-v2`) |
| `find_feature_in_files` | Extracts the same parameter across multiple files for comparison |

---

## RAG: semantic feature search

`find_feature_by_description` uses a sentence embedding model to find the right field even when you don't know the exact name.

**How it works:**
1. Loads the documentation for every field in the file
2. Embeds each `"field_name: description"` into a vector
3. Embeds your query into a vector
4. Returns the top-K fields ranked by **cosine similarity**

**Example:**
- You ask: *"show me the happiness level"*
- The field is actually called `happinessScore`
- The tool finds it because *"happiness level"* and *"happinessScore: Self-reported happiness on a 1–10 scale"* are semantically close

---

## Setup

**1. Install dependencies**
```bash
pip install smolagents huggingface_hub sentence-transformers scipy numpy pandas matplotlib seaborn
```

**2. Add your token to Colab secrets**

Go to the 🔑 key icon in Colab and add:
- `HF_TOKEN` — your Hugging Face token (from huggingface.co/settings/tokens)

Or for Anthropic (Claude):
- `ANTHROPIC_API_KEY` — from console.anthropic.com

**3. Run the notebook cells in order**

---

## Usage

```python
my_rules = """
- You are a senior Data Scientist.
- Always call list_all_features first to understand what fields are available.
- If you are not sure of a field name, use find_feature_by_description before trying to load it.
- Elaborate your final answer as much as you can.
- If the user asks for a report, make it in HTML format and save it.
- Always debug your code before executing.
"""

my_question = "Compare happiness level between both files and make an HTML report."

file_paths = [
    '/content/daily_life_analytics1.mat',
    '/content/daily_life_analytics2.mat'
]

response = run_custom_agent(file_paths, instructions=my_rules, user_prompt=my_question)
```

**The agent has memory** — follow-up questions work across calls:
```python
# First question
run_custom_agent(file_paths, my_rules, "Plot correlation matrix for file 1")

# Follow-up — agent remembers the previous answer
run_custom_agent(file_paths, my_rules, "Which tools did you use in your previous answer?")

# Reset memory when you want a fresh start
conversation_history = []
```

---

## Web Dashboard

A Flask dashboard is available in `app.py` for a full browser-based interface:

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

**Features:**
- Multi-file upload with parameter browser
- Graph viewer with ← → arrow navigation
- Chat interface with conversation memory
- One-click **📊 Full Report** button
- **⏹ Stop** button to interrupt the agent mid-run
- Right panel showing full text responses and HTML report links
- Supports both Hugging Face and Anthropic (Claude) models

---

## Data format

Your `.mat` file must contain a `sequence_data` struct where:
- Each field is a **numeric vector** (time-series values)
- A `documentation` sub-struct maps each field name to a description string

```
sequence_data
├── happinessScore      → [7.2, 6.8, 8.1, ...]
├── sleepHours          → [7.5, 6.0, 8.0, ...]
├── stepsPerDay         → [8200, 5400, 11000, ...]
└── documentation
    ├── happinessScore  → "Self-reported happiness on a 1–10 scale"
    ├── sleepHours      → "Total sleep duration in hours"
    └── stepsPerDay     → "Daily step count from wearable device"
```
