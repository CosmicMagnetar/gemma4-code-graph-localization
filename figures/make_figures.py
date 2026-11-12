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
bars = ax.bar(range(len(order)), [counts[o] for o in order], color=["#55A868"] * 5 + ["#999999"] * 2)
ax.set_xticks(range(len(order)), ["0", "1", "2", "3", "4+", "unreach-\nable", "no anchor\nin issue"])
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

# Subset analysis: does the issue name any symbol in the graph?
import statistics as st

def hits(sub, m, k):
    return sum(1 for r in sub if r["rank"][m] and r["rank"][m] <= k) / max(1, len(sub))

with_a = [r for r in ok if r["min_hop"] != "no_anchor"]
no_a = [r for r in ok if r["min_hop"] == "no_anchor"]
print(f"\n| Subset | n | " + " | ".join(f"{labels[m]} H@1/H@10" for m in methods) + " |")
print("|---|---|" + "---|" * len(methods))
for name, sub in [("issue names >=1 graph symbol", with_a), ("issue names no graph symbol", no_a)]:
    print(f"| {name} | {len(sub)} | " + " | ".join(f"{hits(sub, m, 1):.3f}/{hits(sub, m, 10):.3f}" for m in methods) + " |")
near = [r for r in with_a if isinstance(r["min_hop"], int) and r["min_hop"] <= 2]
print("gold within 2 hops of an anchor:", len(near), "/", len(ok))
print("median degree of candidates (per task) median:", st.median(r["median_degree"] for r in ok),
      " median max-gold degree:", st.median(r["gold_degree"] for r in ok))
print("median edges/nodes:", st.median(r["n_edges"] / r["n_nodes"] for r in ok))
print("median candidate share of nodes:", st.median(r["n_cands"] / r["n_nodes"] for r in ok))
print("tasks with >1 gold symbol:", sum(1 for r in ok if r["n_gold"] > 1), " max gold:", max(r["n_gold"] for r in ok))
print("edge types:", S["edge_types"])
print("TF-IDF hit@67 (cost of a median 2-hop walk):", round(hits(ok, "tfidf", 67), 3))
print("hit@k for k=1..6:", {k: {m: round(hits(ok, m, k), 3) for m in ("tfidf", "graph", "tfidf+graph")} for k in range(1, 7)})
reach = [r for r in ok if isinstance(r["min_hop"], int)]
print("reachable:", len(reach), " of which >=3 hops:", sum(1 for r in reach if r["min_hop"] >= 3))

# ---------------- TagTrail (Gemma 4 read-once tag layer) ----------------
if "kg-tags" in S["hit_at_k"]:
    kgm = ["tfidf", "tfidf-expanded", "kg-tags", "tfidf+graph", "tfidf+graph+kg"]
    kl = {"tfidf": "TF-IDF", "tfidf-expanded": "TF-IDF + Gemma query\n(no code tags)", "kg-tags": "TagTrail tags\nonly",
          "tfidf+graph": "RRF TF-IDF\n+ graph", "tfidf+graph+kg": "RRF TF-IDF + graph\n+ TagTrail"}
    print("\n| Method | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR | Top-1 in edited module |")
    print("|---|---|---|---|---|---|---|")
    for m in kgm:
        h = S["hit_at_k"][m]
        print(f"| {kl[m].replace(chr(10), ' ')} | {h['1']:.3f} | {h['5']:.3f} | {h['10']:.3f} | {h['20']:.3f} | "
              f"{S['mrr'][m]:.3f} | {S['top1_module_hit'][m]:.3f} |")
    na = S["no_anchor"]
    print(f"\nNo-anchor subset n={na['n']}")
    print("| Method | Hit@1 | Hit@10 | Hit@20 | MRR |\n|---|---|---|---|---|")
    for m in kgm:
        print(f"| {kl[m].replace(chr(10), ' ')} | {na[m]['hit@1']:.3f} | {na[m]['hit@10']:.3f} | {na[m]['hit@20']:.3f} | {na[m]['mrr']:.3f} |")
    wa = [r for r in ok if r["min_hop"] != "no_anchor"]
    print(f"\nWith-anchor subset n={len(wa)}")
    for m in kgm:
        print(f"| {kl[m].replace(chr(10), ' ')} | {hits(wa, m, 1):.3f} | {hits(wa, m, 10):.3f} | {hits(wa, m, 20):.3f} |")
    ml, mln = S["module_level"], S["module_level_no_anchor"]
    print(f"\nModule level (n={ml['n']}, median modules={ml['median_modules']}):", {m: ml[m] for m in ("kg-tags", "tfidf")})
    print(f"Module level, no anchor (n={mln['n']}):", {m: mln[m] for m in ("kg-tags", "tfidf")})
    print("tagging meta:", ml["kg_meta"])
    # paired comparison on all tasks: wins/losses of tfidf+graph+kg vs tfidf+graph by rank
    w = sum(1 for r in ok if (r["rank"]["tfidf+graph+kg"] or 10**9) < (r["rank"]["tfidf+graph"] or 10**9))
    l = sum(1 for r in ok if (r["rank"]["tfidf+graph+kg"] or 10**9) > (r["rank"]["tfidf+graph"] or 10**9))
    print(f"paired ranks, +TagTrail vs without: better in {w}, worse in {l}, tied {len(ok)-w-l}")
    w = sum(1 for r in ok if (r["rank"]["kg-tags"] or 10**9) < (r["rank"]["tfidf-expanded"] or 10**9))
    l = sum(1 for r in ok if (r["rank"]["kg-tags"] or 10**9) > (r["rank"]["tfidf-expanded"] or 10**9))
    print(f"paired ranks, tags-only vs Gemma-expanded TF-IDF: better in {w}, worse in {l}")

    # Figure 3: hit@10 by subset for the key methods
    fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=160)
    subs = [("All tasks", ok), ("Issue names a symbol", wa), ("Issue names no symbol", [r for r in ok if r["min_hop"] == "no_anchor"])]
    bw = 0.16
    cols = ["#4C72B0", "#937860", "#55A868", "#C44E52", "#DD8452"]
    for j, m in enumerate(kgm):
        ys = [hits(sub, m, 10) for _, sub in subs]
        xs = [i + (j - 2) * bw for i in range(len(subs))]
        ax.bar(xs, ys, bw, label=kl[m].replace("\n", " "), color=cols[j])
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.01, f"{y:.2f}", ha="center", fontsize=6)
    ax.set_xticks(range(len(subs)), [f"{n}\n(n={len(s)})" for n, s in subs])
    ax.set_ylabel("Hit@10")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig3_tagtrail_subsets.png"))

    # Figure 4: module-level hit@k, tags vs TF-IDF, all and no-anchor
    fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=160)
    kk = [1, 3, 5, 10]
    for m, c, ls in [("kg-tags", "#DD8452", "-"), ("tfidf", "#4C72B0", "-")]:
        ax.plot(kk, [ml[m][f"hit@{k}"] for k in kk], ls, color=c, marker="o", label=f"{'TagTrail tags' if m=='kg-tags' else 'TF-IDF'} (all)")
        ax.plot(kk, [mln[m][f"hit@{k}"] for k in kk], "--", color=c, marker="s", label=f"{'TagTrail tags' if m=='kg-tags' else 'TF-IDF'} (no anchor)")
    ax.set_xticks(kk)
    ax.set_xlabel("Modules opened (k)")
    ax.set_ylabel("Tasks with an edited module in top-k")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig4_module_level.png"))
