# %% [markdown]
# # Where do the fixes live? Code-graph fault localization on the Gemma 4 Developer Agent tasks
#
# Training-free study of 129 public tasks: how far (in call-graph hops) the reference fix is
# from the symbols the issue mentions, and how well cheap lexical / graph / embedding rankers
# localize the edited function. Writes `results.json` (small) to the working directory.
#
# Kaggle: attach the competition data "gemma-4-developer-agent", then Run All.
# Local: `DATA_DIR=/path/to/data python analysis/localize.py`

# %%
import glob
import json
import math
import os
import re
import time
from collections import Counter, defaultdict, deque

import numpy as np

T0 = time.time()
SEARCH_ROOTS = [os.environ.get("DATA_DIR", ""), "/kaggle/input", "./data"]
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
KS = [1, 3, 5, 10, 20, 50, 100]
RRF_K = 60  # standard reciprocal-rank-fusion constant (Cormack et al., 2009); not tuned


def find_tasks_file():
    for root in SEARCH_ROOTS:
        if root and os.path.isdir(root):
            hits = sorted(glob.glob(os.path.join(root, "**", "tasks.jsonl"), recursive=True))
            if hits:
                return hits[0]
    raise FileNotFoundError("tasks.jsonl not found; attach the competition data or set DATA_DIR")


TASKS_PATH = find_tasks_file()
ROOT = os.path.dirname(TASKS_PATH)


import gzip
import tarfile
import zipfile

# Kaggle may mount the data with a different layout or naming than the data page describes,
# so index every graph/embedding file under the input root by its task-id stem.
DATA_ROOT = os.path.dirname(ROOT) if os.path.basename(ROOT) in ("published", "data") else ROOT
if DATA_ROOT in ("/kaggle/input", "/kaggle", "/"):
    DATA_ROOT = ROOT


def show_tree(root, depth=3):
    for dirpath, dirnames, filenames in os.walk(root):
        lvl = dirpath[len(root):].count(os.sep)
        if lvl >= depth:
            dirnames[:] = []
            continue
        ex = ", ".join(sorted(filenames)[:4])
        print(f"{'  ' * lvl}{os.path.basename(dirpath) or dirpath}/  ({len(filenames)} files) {ex}")


def extract_archives(root):
    """Unpack graphs/embeddings archives (if the data ships them zipped) into the working dir."""
    dest = os.path.join(OUT_DIR, "extracted")
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            low = f.lower()
            if not any(k in low for k in ("graph", "embed")):
                continue
            p = os.path.join(dirpath, f)
            if low.endswith(".zip"):
                print("extracting", p)
                zipfile.ZipFile(p).extractall(dest)
            elif low.endswith((".tgz", ".tar.gz", ".tar")):
                print("extracting", p)
                tarfile.open(p).extractall(dest)
    return dest


def index_files(roots):
    gidx, eidx = {}, {}
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                p = os.path.join(dirpath, f)
                if f.endswith(".json.gz"):
                    gidx.setdefault(f[:-8], p)
                elif f.endswith(".json") and f != "tasks.json":
                    gidx.setdefault(f[:-5], p)
                elif f.endswith(".npz"):
                    eidx.setdefault(f[:-4], p)
    return gidx, eidx


INDEX_ROOT = "/kaggle/input" if os.path.isdir("/kaggle/input") else DATA_ROOT
print("data root:", DATA_ROOT, " index root:", INDEX_ROOT)
show_tree(INDEX_ROOT, depth=4)
tasks = [json.loads(l) for l in open(TASKS_PATH) if l.strip()]
GIDX, EIDX = index_files([INDEX_ROOT])
if min(sum(t["instance_id"] in GIDX for t in tasks), sum(t["instance_id"] in EIDX for t in tasks)) < len(tasks) // 2:
    GIDX2, EIDX2 = index_files([extract_archives(INDEX_ROOT)])
    GIDX2.update(GIDX)
    EIDX2.update(EIDX)
    GIDX, EIDX = GIDX2, EIDX2
n_g = sum(t["instance_id"] in GIDX for t in tasks)
n_e = sum(t["instance_id"] in EIDX for t in tasks)
print(f"tasks={len(tasks)}  graphs found={n_g}  embeddings found={n_e}")
if n_g < len(tasks) // 2:
    raise RuntimeError("Graph files not found for most tasks. Copy the folder tree printed above and send it.")

# %% [markdown]
# ## 1. Parse reference patches into hunks

# %%
TRIVIAL = {"pass", "else:", "try:", "return", "finally:", "continue", "break", '"""', "'''", ")", "]", "}", "),", "):"}


def parse_patch(patch):
    """Return list of dicts: file, is_new, hunks[{header, removed, context}]."""
    files, cur, hunk = [], None, None
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            cur = {"file": None, "is_new": False, "hunks": []}
            files.append(cur)
            hunk = None
        elif cur is None:
            continue
        elif line.startswith("--- "):
            if line.strip() == "--- /dev/null":
                cur["is_new"] = True
        elif line.startswith("+++ "):
            p = line[4:].strip()
            cur["file"] = p[2:] if p.startswith("b/") else p
        elif line.startswith("@@"):
            m = re.match(r"@@[^@]*@@(.*)", line)
            hunk = {"header": (m.group(1) if m else "").strip(), "removed": [], "context": [], "added": []}
            cur["hunks"].append(hunk)
        elif hunk is not None:
            if line.startswith("-"):
                hunk["removed"].append(line[1:])
            elif line.startswith("+"):
                hunk["added"].append(line[1:])
            elif line.startswith(" "):
                hunk["context"].append(line[1:])
    return [f for f in files if f["file"]]


def useful(lines):
    out = []
    for l in lines:
        s = l.strip()
        if len(s) >= 4 and s not in TRIVIAL and not s.startswith("#"):
            out.append(s)
    return out


def is_test_path(p):
    p = p.lower()
    return "/tests/" in "/" + p or os.path.basename(p).startswith("test_") or p.startswith("docs/")


def path_to_module(path):
    p = path[:-3] if path.endswith(".py") else path
    if p.startswith("src/"):
        p = p[4:]
    p = p.replace("/", ".")
    if p.endswith(".__init__"):
        p = p[: -len(".__init__")]
    return p


# %% [markdown]
# ## 2. Graph + embedding loading, gold-node mapping, issue anchors

# %%
def load_graph(iid):
    if iid not in GIDX:
        raise FileNotFoundError(iid)
    p = GIDX[iid]
    g = json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))
    ids = [n["id"] for n in g["nodes"]]
    text = {n["id"]: (n.get("text") or "") for n in g["nodes"]}
    adj = defaultdict(set)
    n_edges = 0
    etypes = Counter()
    links = g.get("edges", g.get("links", []))
    for e in links:
        s, t = e["source"], e["target"]
        n_edges += 1
        etypes[e.get("type", "?")] += 1
        if s == t:
            continue
        adj[s].add(t)
        adj[t].add(s)
    return ids, text, adj, n_edges, etypes


def load_emb(iid, ids):
    if iid not in EIDX:
        return {}, 0
    z = np.load(EIDX[iid])
    keys = set(z.files)
    vec, has = {}, 0
    for i in ids:
        k = i if i in keys else (i + ".npy" if i + ".npy" in keys else None)
        if k is not None:
            v = z[k].astype(np.float32).ravel()
            n = np.linalg.norm(v)
            vec[i] = v / n if n > 0 else v
            has += 1
    return vec, has


def is_test_node(nid):
    parts = nid.split(".")
    return any(p in ("tests", "test", "testing") or p.startswith("test_") for p in parts[:-1]) or parts[0] in ("tests", "docs", "docs_src")


def map_gold(files, ids, text):
    """Map each edited hunk to the most specific graph node whose source contains the edited lines."""
    line_sets = {i: set(l.strip() for l in text[i].splitlines()) for i in ids}
    gold, stats = set(), Counter()
    for f in files:
        if is_test_path(f["file"]) or not f["file"].endswith(".py"):
            continue
        stats["src_files"] += 1
        if f["is_new"]:
            stats["new_files"] += 1
            continue
        mod = path_to_module(f["file"])
        cands = [i for i in ids if i == mod or i.startswith(mod + ".")]
        if not cands:  # module naming mismatch: fall back to all nodes
            cands = ids
            stats["module_fallback"] += 1
        for h in f["hunks"]:
            stats["hunks"] += 1
            probe = useful(h["removed"]) or useful(h["context"])
            hm = re.search(r"\b(?:def|class)\s+([A-Za-z_]\w*)", h["header"])
            hname = hm.group(1) if hm else None
            best, best_key = None, None
            for i in cands:
                c = sum(1 for s in probe if s in line_sets[i])
                bonus = 0.5 if hname and i.rsplit(".", 1)[-1] == hname else 0.0
                score = c + bonus if c > 0 else 0.0
                if score <= 0:
                    continue
                key = (score, -len(text[i]))
                if best_key is None or key > best_key:
                    best, best_key = i, key
            if best is None:
                stats["hunks_unmapped"] += 1
            else:
                gold.add(best)
    return gold, stats


IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
STOP = {"self", "None", "True", "False", "return", "import", "from", "class", "def", "with", "this", "that",
        "when", "then", "should", "would", "could", "error", "Error", "value", "values", "data", "name", "type",
        "test", "tests", "python", "Python", "issue", "example", "using", "code", "file", "line", "version",
        "__init__", "__call__", "main", "args", "kwargs", "request", "response", "get", "set", "run"}


def issue_anchors(issue, ids):
    toks = set(IDENT.findall(issue))
    names = defaultdict(list)
    for i in ids:
        names[i.rsplit(".", 1)[-1]].append(i)
    anchors = set()
    for t in toks:
        # dotted mention matching a node id suffix exactly
        if "." in t:
            for i in ids:
                if i == t or i.endswith("." + t):
                    anchors.add(i)
        for part in t.split("."):
            if len(part) >= 4 and part not in STOP and 0 < len(names.get(part, [])) <= 10:
                anchors.update(names[part])
    return anchors


def bfs(adj, sources, max_d=50):
    dist = {s: 0 for s in sources}
    q = deque(sources)
    while q:
        u = q.popleft()
        if dist[u] >= max_d:
            continue
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


# %% [markdown]
# ## 3. Rankers: TF-IDF (lexical), graph hops from issue anchors, embedding similarity, RRF hybrid

# %%
def ident_tokens(s):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if len(t) >= 2]


def tfidf_scores(issue, docs):
    """Cosine of sublinear TF-IDF vectors; idf computed over this repo's nodes."""
    toks = [Counter(ident_tokens(d)) for d in docs]
    df = Counter()
    for c in toks:
        df.update(c.keys())
    N = len(docs)
    idf = {t: math.log((1 + N) / (1 + d)) + 1 for t, d in df.items()}
    q = Counter(ident_tokens(issue))
    qv = {t: (1 + math.log(c)) * idf[t] for t, c in q.items() if t in idf}
    qn = math.sqrt(sum(v * v for v in qv.values())) or 1.0
    out = np.zeros(N, dtype=np.float64)
    for j, c in enumerate(toks):
        dot, nn = 0.0, 0.0
        for t, cnt in c.items():
            w = (1 + math.log(cnt)) * idf[t]
            nn += w * w
            if t in qv:
                dot += w * qv[t]
        out[j] = dot / (qn * (math.sqrt(nn) or 1.0))
    return out


def rank_of_first_gold(order, gold):
    for r, i in enumerate(order, 1):
        if i in gold:
            return r
    return None


def order_by(cands, score, tiebreak):
    # higher score first, then higher tiebreak, then id for determinism
    return sorted(cands, key=lambda i: (-score[i], -tiebreak[i], i))


def rrf(orders):
    s = defaultdict(float)
    for o in orders:
        for r, i in enumerate(o, 1):
            s[i] += 1.0 / (RRF_K + r)
    return s


# %% [markdown]
# ## 4. Run over all tasks

# %%
rows, agg_stats, edge_types = [], Counter(), Counter()
for ti, t in enumerate(tasks):
    iid = t["instance_id"]
    try:
        ids, text, adj, n_edges, et = load_graph(iid)
    except FileNotFoundError:
        agg_stats["missing_graph"] += 1
        continue
    edge_types.update(et)
    files = parse_patch(t.get("patch", ""))
    gold, st = map_gold(files, ids, text)
    agg_stats.update(st)
    issue = t.get("problem_statement", "") or ""
    cands = [i for i in ids if not is_test_node(i)]
    cset = set(cands)
    gold &= cset
    row = {"instance_id": iid, "repo": t.get("repo"), "n_nodes": len(ids), "n_edges": n_edges,
           "n_cands": len(cands), "n_gold": len(gold), "issue_chars": len(issue),
           "n_src_files_edited": sum(1 for f in files if f["file"].endswith(".py") and not is_test_path(f["file"]))}
    if not gold:
        row["status"] = "no_gold"
        rows.append(row)
        continue

    anchors = issue_anchors(issue, ids) & cset
    row["n_anchors"] = len(anchors)
    row["gold_is_anchor"] = bool(gold & anchors)
    row["gold_ids"] = sorted(gold)

    # graph distance from anchors to nearest gold (undirected call/reference graph)
    if anchors:
        dist = bfs(adj, list(anchors))
        dg = [dist[g] for g in gold if g in dist]
        row["min_hop"] = min(dg) if dg else None  # None = unreachable
        row["hood"] = {str(k): sum(1 for i, d in dist.items() if d <= k and i in cset) for k in (0, 1, 2, 3)}
    else:
        dist = {}
        row["min_hop"] = "no_anchor"

    # gold degree vs repo median (are fixes in hubs or leaves?)
    degs = [len(adj.get(i, ())) for i in cands]
    row["gold_degree"] = max(len(adj.get(g, ())) for g in gold)
    row["median_degree"] = float(np.median(degs)) if degs else 0.0

    # rankers
    docs = [i.replace(".", " ") + " " + text[i] for i in cands]
    tf = dict(zip(cands, tfidf_scores(issue, docs)))
    o_tfidf = order_by(cands, tf, {i: 0 for i in cands})

    hop_score = {i: -dist.get(i, 10**6) for i in cands}
    o_graph = order_by(cands, hop_score, tf)  # hop distance, ties broken lexically

    vec, n_emb = load_emb(iid, ids)
    row["emb_coverage"] = n_emb / max(1, len(ids))
    seeds = [i for i in anchors if i in vec] or [i for i in o_tfidf[:5] if i in vec]
    if seeds and vec:
        S = np.stack([vec[s] for s in seeds])
        emb = {i: float(np.max(S @ vec[i])) if i in vec else -2.0 for i in cands}
    else:
        emb = {i: 0.0 for i in cands}
    o_emb = order_by(cands, emb, tf)

    fused = rrf([o_tfidf, o_graph, o_emb])
    o_hybrid = order_by(cands, fused, tf)
    fused2 = rrf([o_tfidf, o_graph])
    o_tg = order_by(cands, fused2, tf)

    row["rank"] = {
        "tfidf": rank_of_first_gold(o_tfidf, gold),
        "graph": rank_of_first_gold(o_graph, gold),
        "embedding": rank_of_first_gold(o_emb, gold),
        "tfidf+graph": rank_of_first_gold(o_tg, gold),
        "hybrid": rank_of_first_gold(o_hybrid, gold),
    }
    # random baseline: expected hit@k with g gold among N candidates (hypergeometric)
    N, g = len(cands), len(gold)
    row["random_hit"] = {str(k): 1 - (math.comb(N - g, k) / math.comb(N, k) if k <= N - g else 0.0) for k in KS}
    # file-level: does the top-1 node live in an edited module?
    gold_mods = {path_to_module(f["file"]) for f in files if f["file"].endswith(".py") and not is_test_path(f["file"])}
    row["top1_module_hit"] = {m: any(o[0] == gm or o[0].startswith(gm + ".") for gm in gold_mods)
                              for m, o in [("tfidf", o_tfidf), ("graph", o_graph), ("embedding", o_emb),
                                           ("tfidf+graph", o_tg), ("hybrid", o_hybrid)]}
    row["status"] = "ok"
    rows.append(row)
    if ti % 10 == 0:
        print(f"[{ti+1}/{len(tasks)}] {iid} nodes={len(ids)} gold={len(gold)} anchors={len(anchors)} "
              f"hop={row['min_hop']} ranks={row['rank']}  t={time.time()-T0:.0f}s")

# %% [markdown]
# ## 5. Aggregate and save

# %%
ok = [r for r in rows if r["status"] == "ok"]
methods = ["tfidf", "graph", "embedding", "tfidf+graph", "hybrid"]


def hit_at(m, k, subset):
    return sum(1 for r in subset if r["rank"][m] is not None and r["rank"][m] <= k) / max(1, len(subset))


def mrr(m, subset):
    return sum(1.0 / r["rank"][m] for r in subset if r["rank"][m]) / max(1, len(subset))


summary = {
    "n_tasks": len(tasks), "n_rows": len(rows), "n_ok": len(ok),
    "n_no_gold": sum(1 for r in rows if r["status"] == "no_gold"),
    "mapping_stats": dict(agg_stats), "edge_types": dict(edge_types),
    "repos": dict(Counter(r["repo"] for r in rows)),
    "hit_at_k": {m: {str(k): hit_at(m, k, ok) for k in KS} for m in methods},
    "mrr": {m: mrr(m, ok) for m in methods},
    "random_hit_at_k": {str(k): float(np.mean([r["random_hit"][str(k)] for r in ok])) if ok else 0 for k in KS},
    "top1_module_hit": {m: float(np.mean([r["top1_module_hit"][m] for r in ok])) if ok else 0 for m in methods},
    "per_repo": {},
    "min_hop_hist": dict(Counter(str(r["min_hop"]) for r in ok)),
    "gold_is_anchor": sum(1 for r in ok if r["gold_is_anchor"]),
    "median_nodes": float(np.median([r["n_nodes"] for r in rows])) if rows else 0,
    "median_cands": float(np.median([r["n_cands"] for r in ok])) if ok else 0,
    "median_gold": float(np.median([r["n_gold"] for r in ok])) if ok else 0,
    "runtime_s": round(time.time() - T0, 1),
}
for repo in sorted(set(r["repo"] for r in ok)):
    sub = [r for r in ok if r["repo"] == repo]
    summary["per_repo"][repo] = {"n": len(sub), **{m: {"hit@1": hit_at(m, 1, sub), "hit@10": hit_at(m, 10, sub),
                                                         "mrr": mrr(m, sub)} for m in methods}}

out = os.path.join(OUT_DIR, "results.json")
json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
print("wrote", out, f"in {time.time()-T0:.0f}s")
