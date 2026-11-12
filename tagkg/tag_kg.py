# %% [markdown]
# # Read once, tag once: a Gemma 4 tag layer over the competition code graphs
#
# Gemma 4 reads an outline of every module in every repository snapshot exactly once and writes short
# behaviour tags. It then reads each bug report, states the faulty behaviour in one line, and tags it in the
# same style. Output: `kg_tags.json` = {"modules": {task: {module: tags}}, "issues": {task: {...}}, "meta": {...}}.
# Evaluation is done by `analysis/localize.py`, which reads this file.
#
# Kaggle: GPU T4 x2, internet on (to update transformers), inputs = competition data + Gemma 4 model.

# %%
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate"], check=False)

# %%
import glob
import hashlib
import json
import os
import re
import time
from collections import defaultdict

T0 = time.time()
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
INPUT = os.environ.get("DATA_DIR") or "/kaggle/input"
LIMIT_TASKS = int(os.environ.get("LIMIT_TASKS", "0"))  # 0 = all
DRY_RUN = os.environ.get("DRY_RUN") == "1"  # no model: fake tags, for local testing

tasks_path = sorted(glob.glob(os.path.join(INPUT, "**", "tasks.jsonl"), recursive=True))[0]
tasks = [json.loads(l) for l in open(tasks_path) if l.strip()]
if LIMIT_TASKS:
    tasks = tasks[:LIMIT_TASKS]
GIDX = {}
for p in glob.glob(os.path.join(INPUT, "**", "*.json"), recursive=True):
    GIDX.setdefault(os.path.basename(p)[:-5], p)


def graph_path(t):
    iid = t["instance_id"]
    for k in [iid, f"{t['repo'].split('/')[-1].lower()}_{t['base_commit']}", f"{iid.rsplit('_', 1)[0]}_{t['base_commit']}"]:
        if k in GIDX:
            return GIDX[k]
    return None


print(f"tasks={len(tasks)} graphs found={sum(graph_path(t) is not None for t in tasks)}")

# %% [markdown]
# ## 1. Module outlines (the "read once" unit)

# %%
PKG = {"fastapi": "fastapi", "rich": "rich", "requests": "requests", "httpx": "httpx"}
DEF_RE = re.compile(r"^\s*(async\s+def|def|class)\s")
DOC_RE = re.compile(r'(?:"""|\'\'\')\s*(.*?)\s*(?:\n|"""|\'\'\')', re.S)


def is_test_node(nid):
    parts = nid.split(".")
    return any(p in ("tests", "test", "testing") or p.startswith("test_") for p in parts[:-1]) or parts[0] in ("tests", "docs", "docs_src")


def first_code_line(text):
    for l in text.splitlines():
        s = l.strip()
        if s and not s.startswith("@") and not s.startswith("#"):
            return l
    return ""


def is_symbol(text):
    return bool(DEF_RE.match(first_code_line(text)))


def module_of(nid, node_ids):
    """Graphs hold only classes and functions, so a symbol's module is the longest dotted prefix that is not a node."""
    parts = nid.split(".")
    mod = None
    for i in range(1, len(parts)):
        p = ".".join(parts[:i])
        if p in node_ids:
            break
        mod = p
    return mod


def outlines(graph):
    """module id -> outline text (signature + first docstring line of each symbol in the module)."""
    text = {n["id"]: (n.get("text") or "") for n in graph["nodes"]}
    node_ids = set(text)
    members = defaultdict(list)
    pkg = PKG.get(graph.get("graph", {}).get("repo_name", "").split("_")[0], None)
    for i in text:
        if is_test_node(i):
            continue
        if pkg and i.split(".")[0] != pkg:  # library code only (all reference fixes live there)
            continue
        m = module_of(i, node_ids)
        if m:
            members[m].append(i)
    out = {}
    for m in sorted(members):
        lines = []
        for i in sorted(members[m], key=lambda x: (x.count("."), x))[:60]:  # top-level symbols first
            sig = first_code_line(text[i]).strip()[:110]
            dm = DOC_RE.search(text[i][:600])
            doc = (dm.group(1).strip().splitlines() or [""])[0][:90] if dm else ""
            lines.append(f"- {i[len(m) + 1:]}: {sig}" + (f"  # {doc}" if doc else ""))
        out[m] = "\n".join(lines)[:3500]
    return out


task_modules, unique = {}, {}
for t in tasks:
    p = graph_path(t)
    if not p:
        continue
    g = json.load(open(p))
    om = outlines(g)
    task_modules[t["instance_id"]] = {}
    for m, body in om.items():
        h = hashlib.sha1(f"{t['repo']}|{m}|{body}".encode()).hexdigest()[:16]
        task_modules[t["instance_id"]][m] = h
        unique.setdefault(h, (t["repo"], m, body))
print(f"modules per task (median) = {sorted(len(v) for v in task_modules.values())[len(task_modules)//2]}, "
      f"unique module outlines to tag = {len(unique)}  t={time.time()-T0:.0f}s")

# %% [markdown]
# ## 2. Gemma 4 (greedy decoding, thinking off)

# %%
MODULE_PROMPT = """You are indexing the Python library {repo} so that bug reports can be routed to the right code.
Below is an outline of the module `{module}` (its functions and classes, with signatures and docstring first lines).

{body}

Write 8 to 12 short tags, each 2 to 4 lowercase words, naming the behaviours, concepts and features this module implements
(for example: "cookie parsing", "redirect handling", "progress bar rendering"). Prefer plain words over code names.
Answer with one line only, in this format:
TAGS: tag one, tag two, tag three"""

ISSUE_PROMPT = """You are triaging a bug report for the Python library {repo}. You cannot see the code.

<report>
{report}
</report>

First, say in one sentence what behaviour is wrong or missing. Then write 8 to 12 short tags, each 2 to 4 lowercase words,
naming the behaviours, concepts and features of {repo} that the fix will most likely touch (for example: "cookie parsing",
"redirect handling", "progress bar rendering"). Prefer plain words over code names.
Answer with exactly two lines, in this format:
BUG: <one sentence>
TAGS: tag one, tag two, tag three"""

MODEL_ID = None
if not DRY_RUN:
    import torch
    import kagglehub
    from transformers import AutoModelForCausalLM, AutoProcessor

    n_gpu = torch.cuda.device_count()
    mem = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n_gpu)) / 2**30
    MODEL_ID = "google/gemma-4/transformers/gemma-4-e4b-it" if mem > 20 else "google/gemma-4/transformers/gemma-4-e2b-it"
    print(f"gpus={n_gpu} total_mem={mem:.1f}GiB -> {MODEL_ID}")
    path = kagglehub.model_download(MODEL_ID)
    processor = AutoProcessor.from_pretrained(path)
    tok = getattr(processor, "tokenizer", processor)
    tok.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map="auto")
    model.eval()
    print("model loaded", f"dtype={dtype}", f"t={time.time()-T0:.0f}s")


def generate(prompts, max_new_tokens=96, bs=12):
    if DRY_RUN:
        return [f"BUG: fake\nTAGS: {' '.join(re.findall(r'[a-z]{4,}', p.lower())[-6:-3])}, fake tag" for p in prompts]
    outs = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        texts = [processor.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True).strip())
        if (i // bs) % 10 == 0:
            print(f"  {i + len(chunk)}/{len(prompts)}  t={time.time()-T0:.0f}s", flush=True)
    return outs


def parse_tags(s):
    m = re.search(r"TAGS:\s*(.*)", s)
    raw = m.group(1) if m else s.splitlines()[-1] if s else ""
    return [x.strip().strip(".").lower() for x in raw.split(",") if x.strip()][:15]


def parse_bug(s):
    m = re.search(r"BUG:\s*(.*)", s)
    return m.group(1).strip() if m else ""


# %% [markdown]
# ## 3. Tag every unique module outline once, then every bug report

# %%
keys = sorted(unique, key=lambda h: len(unique[h][2]))  # similar lengths per batch -> less padding
mod_raw = generate([MODULE_PROMPT.format(repo=unique[h][0], module=unique[h][1], body=unique[h][2]) for h in keys])
mod_tags = {h: parse_tags(r) for h, r in zip(keys, mod_raw)}
print(f"tagged {len(mod_tags)} modules, empty={sum(1 for v in mod_tags.values() if not v)}  t={time.time()-T0:.0f}s")

iss_raw = generate([ISSUE_PROMPT.format(repo=t["repo"], report=(t.get("problem_statement") or "")[:6000]) for t in tasks],
                   max_new_tokens=160, bs=8)
issues = {t["instance_id"]: {"bug": parse_bug(r), "tags": parse_tags(r), "raw": r} for t, r in zip(tasks, iss_raw)}
print(f"tagged {len(issues)} issues, empty={sum(1 for v in issues.values() if not v['tags'])}  t={time.time()-T0:.0f}s")

# %%
out = {
    "meta": {"model": MODEL_ID or "DRY_RUN", "unique_modules": len(unique), "n_tasks": len(task_modules),
             "runtime_s": round(time.time() - T0, 1), "decoding": "greedy, thinking off",
             "module_prompt": MODULE_PROMPT, "issue_prompt": ISSUE_PROMPT},
    "modules": {iid: {m: mod_tags[h] for m, h in mm.items()} for iid, mm in task_modules.items()},
    "module_raw_examples": {unique[h][1]: r for h, r in list(zip(keys, mod_raw))[:20]},
    "issues": issues,
}
json.dump(out, open(os.path.join(OUT_DIR, "kg_tags.json"), "w"))
print("wrote kg_tags.json", out["meta"])
for iid in list(issues)[:3]:
    print(iid, issues[iid]["bug"], issues[iid]["tags"])
