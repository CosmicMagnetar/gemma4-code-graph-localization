"""Build paper figures from results/results.json (produced by analysis/localize.ipynb on Kaggle).

Usage: python figures/make_figures.py [results/results.json]
Writes figures/fig1_hit_at_k.png and figures/fig2_hop_distance.png, and prints the tables used in WRITEUP.md.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "results.json")
d = json.load(open(path))
S, rows = d["summary"], d["rows"]
ok = [r for r in rows if r.get("status") == "ok"]
methods = ["tfidf", "embedding", "graph", "tfidf+graph", "hybrid"]
labels = {"tfidf": "TF-IDF (issue text)", "embedding": "Embedding (provided vectors)",
          "graph": "Graph hops from issue anchors", "tfidf+graph": "RRF: TF-IDF + graph",
          "hybrid": "RRF: all three"}
colors = {"tfidf": "#4C72B0", "embedding": "#8172B2", "graph": "#55A868", "tfidf+graph": "#C44E52",
          "hybrid": "#DD8452"}

# Figure 1: hit@k curve, k = 1..100
ks = list(range(1, 101))
fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
for m in methods:
    ys = [sum(1 for r in ok if r["rank"][m] and r["rank"][m] <= k) / len(ok) for k in ks]
    ax.plot(ks, ys, label=labels[m], color=colors[m], lw=2)
rk = sorted(int(k) for k in S["random_hit_at_k"])
ax.plot(rk, [S["random_hit_at_k"][str(k)] for k in rk], "--", color="gray", lw=1.2, label="Random (expected)")
ax.set_xscale("log")
ax.set_xlabel("Candidate functions/classes read (k, log scale)")
ax.set_ylabel("Tasks with an edited symbol in top-k")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7.5, loc="upper left", frameon=False)
ax.set_title(f"Fault localization on {len(ok)} Gemma 4 agent tasks", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig1_hit_at_k.png"))

# Figure 2: hop distance from issue-mentioned symbols to nearest edited symbol
order = ["0", "1", "2", "3", "4+", "unreachable", "no anchor"]
counts = dict.fromkeys(order, 0)
for r in ok:
    h = r["min_hop"]
    if h == "no_anchor":
        counts["no anchor"] += 1
    elif h is None:
        counts["unreachable"] += 1
    elif int(h) >= 4:
        counts["4+"] += 1
    else:
        counts[str(int(h))] += 1
fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=160)
bars = ax.bar(order, [counts[o] for o in order], color=["#55A868"] * 5 + ["#999999"] * 2)
for b, o in zip(bars, order):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, str(counts[o]), ha="center", fontsize=8)
ax.set_xlabel("Call-graph hops from issue-mentioned symbols to nearest edited symbol")
ax.set_ylabel("Tasks")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig2_hop_distance.png"))

# Tables for the writeup
print(f"tasks={S['n_tasks']} analysed={S['n_ok']} no_gold={S['n_no_gold']} runtime_s={S['runtime_s']}")
print("mapping:", S["mapping_stats"])
print("repos:", S["repos"])
print(f"median nodes={S['median_nodes']} median candidates={S['median_cands']} median gold={S['median_gold']}")
print("gold symbol named in issue:", S["gold_is_anchor"], "/", S["n_ok"])
print("hop counts:", counts)
print("\n| Method | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR | Top-1 in edited module |")
print("|---|---|---|---|---|---|---|")
for m in methods:
    h = S["hit_at_k"][m]
    print(f"| {labels[m]} | {h['1']:.3f} | {h['5']:.3f} | {h['10']:.3f} | {h['20']:.3f} | {S['mrr'][m]:.3f} | "
          f"{S['top1_module_hit'][m]:.3f} |")
r = S["random_hit_at_k"]
print(f"| Random (expected) | {r['1']:.3f} | {r['5']:.3f} | {r['10']:.3f} | {r['20']:.3f} | - | - |")
print("\n| Repo | n | " + " | ".join(f"{labels[m]} Hit@10" for m in methods) + " |")
print("|---|---|" + "---|" * len(methods))
for repo, v in S["per_repo"].items():
    print(f"| {repo} | {v['n']} | " + " | ".join(f"{v[m]['hit@10']:.3f}" for m in methods) + " |")
hood = [r["hood"] for r in ok if isinstance(r.get("hood"), dict)]
if hood:
    import statistics as st
    print("\nmedian k-hop neighbourhood size from anchors:",
          {k: st.median(h[k] for h in hood) for k in ("0", "1", "2", "3")})
deg = [(r["gold_degree"], r["median_degree"]) for r in ok]
print("gold degree > repo median degree in", sum(1 for g, m in deg if g > m), "/", len(deg), "tasks")
