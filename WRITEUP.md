# Where Do the Fixes Live? Code-Graph Fault Localization for Small Local Coding Agents

**Subtitle:** A training-free study of the 129 public Gemma 4 Developer Agent tasks. How far are reference fixes from the symbols an issue mentions, and which cheap rankers find them before a model reads any code?

## Abstract

A small model running on one accelerator cannot afford to read a whole repository. Before Gemma 4 can draft a fix it must decide *where to look*, using a few tool calls and a limited context window. We measure how hard that first step is on the 129 public tasks of the Gemma 4 Developer Agent competition, using only what the competition ships: AST call graphs, 256-dimensional node embeddings and the issue text. We map each reference patch to the graph nodes it edits, extract the symbols each issue names, and measure two things. The first is the call-graph distance between named and edited symbols. The second is how high four training-free rankers place an edited symbol: identifier-aware TF-IDF, graph hops from issue anchors, similarity in the provided embedding space, and reciprocal-rank fusion.

Of 115 tasks we could map, the issue names an edited symbol in only 28 (24%) and names no graph symbol at all in 34 (30%). Only 43 have an edited symbol within two hops of a named one. The graph is sparse, with a median of 0.68 `calls` edges per node, and 8 fixes cannot be reached from the issue at all. Graph hops give the best first guess (hit@1 0.165). TF-IDF is strongest once the agent can read five or more symbols (hit@10 0.496, against 0.029 for random). Fusing the two gives the best MRR (0.257). The provided embeddings, used node-to-node, are the weakest signal, and adding them to the fusion lowers every metric. Everything runs on a Kaggle CPU notebook in two minutes.

## 1. Introduction

<!-- REWRITE IN MY VOICE: motivation. Why localization on everyday hardware matters to you; what you saw or expected when thinking about small agents in big repos. -->
Agents built on large hosted models can afford to explore, open many files and recover from wrong turns. A model that fits on a laptop has no such slack. Every file it opens costs context, and a wrong first guess about where the bug lives is often the whole failure: the agent edits the wrong function, tests fail, and the remaining turns go to undoing it.

The Gemma 4 Developer Agent competition anticipates this. With each repository snapshot it ships a code graph and a dense embedding per node. It also provides tools built on these, such as `get_code_neighbors`, `get_code_subgraph` and `search_similar_code`. What is not yet known is how useful these structures are for an agent's first decision. Are the functions that need editing named in the issue? If not, are they one call away, or far away? Does the embedding space place them near the named symbols?

We answer these questions with measurements rather than a new agent. Our contributions:

1. **A gold localization map** of the public split. Each reference-patch hunk is mapped to the most specific graph node containing its edited lines, and we report how often this succeeds.
2. **A distance profile of real fixes:** the call-graph distance from issue-named symbols to the nearest edited symbol.
3. **A training-free localization baseline.** Lexical, graph, embedding and fused rankers are compared under one protocol against an exactly computed random baseline.
4. **A reproducible pipeline.** One Kaggle CPU notebook produces every number, and one script turns its JSON output into every table and figure.

## 2. Related Work

**Benchmarks and agents.** SWE-bench (Jimenez et al., 2024) made resolving real GitHub issues, verified by repository tests, the standard task for software-engineering agents. The competition tasks share its format: problem statement, base commit, reference patch and test patch. SWE-agent (Yang et al., 2024) showed that the agent–computer interface strongly affects success. OpenHands (Wang et al., 2024) is an open platform for such agents. AutoCodeRover (Zhang et al., 2024) added structure-aware search over classes and methods, which is close in spirit to the competition's graph tools.

**Localization as a separate stage.** Agentless (Xia et al., 2024) showed that a fixed pipeline of hierarchical localization, repair and validation competes with open-ended agents, which makes localization a major lever. LocAgent (Chen et al., 2025) lets an LLM agent traverse a heterogeneous graph of files, classes and functions. RepoGraph (Ouyang et al., 2024) plugs a line-level repository graph into existing agents. Our work is complementary. Rather than proposing another graph-guided agent, we measure, on the exact graphs and embeddings given to Gemma 4 participants, how far fixes are from the issue and how much model-free use of the graph already buys.

**IR bug localization.** Before LLMs, bug localization was framed as retrieving source files for a bug report. BLUiR (Saha et al., 2013) showed that using code structure, such as class, method and identifier fields, improves over flat text retrieval. Our TF-IDF ranker is a function-level descendant of that work. We combine rankers with reciprocal rank fusion (Cormack et al., 2009) at its standard constant, so nothing is tuned on the evaluation tasks.

**Gemma.** Gemma models (Gemma Team, 2025) are open-weight models designed for single accelerators and consumer hardware. That is the setting behind our budget-centred evaluation.

## 3. Method

### 3.1 Data and gold localization

We use the public split: `tasks.jsonl` (129 tasks with problem statement, base commit and reference `patch`), one NetworkX node-link graph per repository snapshot, and one `.npz` of 256-d float32 node vectors per snapshot. In a graph, node `id` is the fully qualified symbol and `text` is its source. We do not use snapshots, test patches or any model.

Patches change *lines*, but agent tools work on *symbols*. We parse each diff into files and hunks. We skip tests, documentation examples (`docs_src/`) and new files, since new files have no node at the base commit. For each hunk we take its removed lines, or its context lines for pure additions, and drop trivial lines such as `pass`, brackets, comments and anything under four characters. Candidate nodes are restricted to the edited module (`src/pkg/a/b.py` → `pkg.a.b`). Each candidate scores one point per probe line that appears verbatim in its source, plus 0.5 if its name matches the function in the hunk header. The top score wins, and ties go to the shortest source, so a method beats its enclosing class. A task's gold set is the union over its hunks.

### 3.2 Issue anchors

An agent's first clue is the identifiers in the issue. We extract identifier and dotted-identifier tokens from the problem statement. A node is an *anchor* if the issue mentions its full id or a dotted suffix of it. A node is also an anchor if its short name equals a mentioned identifier component that:
- is at least four characters long,
- is not a generic word (`self`, `request`, `value`, and so on), and
- names at most ten nodes in the repository, so that common names do not flood the set.

Hints are not used.

### 3.3 Rankers

All rankers order the same candidates: every graph node outside test and documentation packages.

- **TF-IDF.** Issue text against node id plus source. Identifiers are split on non-alphanumerics, underscores and camelCase, with sublinear term frequency and per-repository IDF.
- **Graph.** Breadth-first search over the *undirected* call graph from all anchors, ordered by hop distance with ties broken by TF-IDF. Without anchors it falls back to TF-IDF.
- **Embedding.** The maximum cosine similarity between a node and any seed, in the provided space. Seeds are the anchors, or the top five TF-IDF nodes when there are none. No natural-language query encoder ships with the vectors, so we use them node-to-node, as `search_similar_code` would from a symbol the agent already holds.
- **Fusion.** Reciprocal rank fusion with k = 60 (Cormack et al., 2009), once for TF-IDF + graph and once for all three rankers. The constant is the published default and is not tuned.

### 3.4 Metrics

For each task and ranker we record the rank of the first gold node. We report three metrics:
- **hit@k:** the share of tasks with a gold node in the top k.
- **MRR:** mean reciprocal rank of the first gold node.
- **File-level top-1:** whether the top-ranked node lies in an edited module.

The random baseline is exact: with N candidates and g gold nodes, hit@k = 1 − C(N−g, k)/C(N, k). For tasks with anchors we also record the minimum hop distance to gold and the size of the 0–3-hop neighbourhood, which is how much code a graph walk must read to reach that radius.

## 4. Results

Every number below is printed by `figures/make_figures.py` from the `results.json` written by the Kaggle notebook (120.5 s on CPU).

**Coverage.** The tasks come from fastapi (67), rich (48), requests (13) and httpx (1). Their patches touch 244 non-test Python files, 28 of them new. Of 632 hunks, 455 (72%) map to a node. 115 tasks have at least one gold node among the candidates and form the evaluation set. The other 14 only add files, only edit documentation examples, or have no mappable hunk. Graphs have a median of 3,250 nodes, of which 1,155 (39%) are non-test candidates. The median task has one gold node, but 49 tasks edit more than one (maximum 43).

### 4.1 Localization accuracy

| Method | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR | Top-1 in edited module |
|---|---|---|---|---|---|---|
| TF-IDF (issue text) | 0.113 | **0.348** | **0.496** | **0.600** | 0.228 | 0.330 |
| Embedding (provided vectors) | 0.104 | 0.235 | 0.278 | 0.339 | 0.164 | 0.261 |
| Graph hops from issue anchors | **0.165** | 0.296 | 0.365 | 0.426 | 0.235 | 0.365 |
| RRF: TF-IDF + graph | 0.157 | 0.330 | 0.426 | 0.557 | **0.257** | **0.383** |
| RRF: all three | 0.148 | 0.296 | 0.383 | 0.522 | 0.234 | 0.348 |
| Random (expected) | 0.003 | 0.015 | 0.029 | 0.056 | – | – |

*Table 1. Function-level localization on 115 tasks.*

All rankers are far above random, and which one is best depends on the reading budget (Figure 1). At k = 1, graph hops win. For k = 2–4, the TF-IDF + graph fusion leads. From k = 5 to 100, TF-IDF leads or ties, finding 60% of fixes within 20 symbols. The fusion has the best MRR and file-level top-1. Adding embeddings to the fusion lowers every column.

![Hit@k vs. reading budget](figures/fig1_hit_at_k.png)
*Figure 1. Share of tasks with an edited symbol in the top k (log x-axis).*

### 4.2 How far are fixes from the issue?

![Hop distance from issue anchors to edited symbol](figures/fig2_hop_distance.png)
*Figure 2. Shortest undirected call-graph distance from any issue anchor to the nearest edited symbol.*

The distribution has two peaks. In 28 tasks the issue names an edited symbol outright, and hops 1 and 2 add only 15 more. Of the 73 reachable fixes, 30 (41%) are three or more hops away. Another 8 are unreachable, and 34 issues name no graph symbol at all. Walks grow fast: the median neighbourhood around the anchors holds 3, 13, 67 and 116 nodes at radius 0–3. Reading the whole 2-hop neighbourhood therefore costs about as much as reading the top 67 TF-IDF results. The walk covers 43/115 tasks, while TF-IDF at 67 covers 0.774 of them.

The graph has a single edge type, `calls` (460,018 edges summed over the 129 task graphs). It has a median of 0.68 edges per node and a median candidate degree of 1. Edited symbols are better connected than typical nodes: the most-connected edited symbol has a median degree of 4 and exceeds its repository's median in 88/115 tasks.

**Anchors decide which signal works.**

| Subset | n | TF-IDF H@1 / H@10 | Embedding H@1 / H@10 | Graph H@1 / H@10 | RRF TF-IDF+graph H@1 / H@10 |
|---|---|---|---|---|---|
| Issue names ≥1 graph symbol | 81 | 0.136 / **0.642** | 0.148 / 0.358 | **0.210** / 0.457 | 0.198 / 0.543 |
| Issue names no graph symbol | 34 | 0.059 / 0.147 | 0.000 / 0.088 | 0.059 / 0.147 | 0.059 / 0.147 |

*Table 2. Without anchors the graph ranker falls back to TF-IDF, so those rows coincide by construction.*

When anchors exist, graph hops give the best first guess, but TF-IDF still finds more within ten reads. When they don't, every method is weak, and none places the edit in its top ten for more than 15% of tasks. Here the bottleneck is understanding the prose issue, which none of these rankers attempts.

### 4.3 Per-repository breakdown

| Repo | n | TF-IDF | Embedding | Graph | RRF TF-IDF+graph | RRF all |
|---|---|---|---|---|---|---|
| fastapi/fastapi | 59 | 0.424 | 0.169 | 0.254 | 0.356 | 0.288 |
| Textualize/rich | 43 | 0.512 | 0.302 | 0.395 | 0.419 | 0.395 |
| psf/requests | 12 | 0.750 | 0.667 | 0.750 | 0.750 | 0.750 |
| encode/httpx | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

*Table 3. Hit@10 per repository. The httpx row is a single task.*

TF-IDF has the best hit@10 in every repository. fastapi is the hardest, and the embedding ranker falls to 0.169 there.

## 5. Discussion: what this means for a Gemma 4 agent

<!-- REWRITE IN MY VOICE: key insight. In your own words, what should someone building a small local agent do differently after seeing these numbers? -->
These results suggest three concrete changes for a small agent working within a fixed context budget.

**Use the issue's identifiers as a first guess, not a search frontier.** When the issue names symbols, opening the anchors and their neighbours is the best single guess (hit@1 0.210 on those tasks). Expanding hop by hop from there wastes context. Radius 2 already costs a median of 67 symbols, and 41% of reachable fixes lie three or more hops out.

**Back the guess with lexical retrieval.** Identifier-aware TF-IDF finds half of all fixes within ten symbols and costs milliseconds and zero tokens. Unweighted RRF with graph distance gives the best overall ranking without tuning. A sensible default is to give the model a fused top-20 list before its first file read.

**Do not use the provided embeddings as symbol-to-symbol search for localization.** Used this way they are the weakest signal, and they hurt the fusion. They may still serve clustering or duplicate detection, or work with a matched query encoder.

The 34 tasks without anchors are where a language model is actually needed: turning a prose issue into likely identifiers or files. We expect Gemma 4 to add the most over these baselines there.

## 6. Limitations

<!-- REWRITE IN MY VOICE: limitations. Say plainly what you would do with more time and what you are unsure about. -->
- **No model in the loop.** We measure the localization step, not resolve rate. We did not run Gemma 4, so we cannot claim that better rankings yield more fixes. The next experiment is to give each ranker's top-k to a Gemma 4 agent and measure resolved tasks.
- **Small public split.** 115 usable tasks from four libraries, dominated by fastapi and rich, with a single httpx task. The hidden test set may differ.
- **Heuristic gold mapping.** 28% of hunks did not map to a node. Identical lines in several functions can mislead the mapping, and new files are excluded entirely.
- **One reference fix.** Other correct fixes may edit other functions, so hit@k is a lower bound on useful localization.
- **Anchor heuristic.** Name matching misses symbols described only in prose and can over-match. The ten-node ambiguity cap is a judgment call, not a tuned value.
- **Embeddings used node-to-node.** Without a matched text encoder we cannot test issue-to-code embedding retrieval.

## 7. Reproduce

- **Hardware:** Kaggle CPU notebook, with no GPU and no internet; the competition data is attached. Figures were made on a laptop (Python 3, numpy, matplotlib).
- **Notebook:** https://www.kaggle.com/code/cosmicmagnetar/gemma-4-code-graph-localization
- **Code:** REPO-LINK (code, `results.json`, figure script).

```bash
# Run the analysis on Kaggle (or open the notebook above and click Run All)
kaggle kernels push -p analysis            # set "id" in analysis/kernel-metadata.json to your username first
kaggle kernels output <user>/gemma-4-code-graph-localization -p results
# Rebuild every table and figure in this paper
python figures/make_figures.py results/results.json
```

## References

- Chen, Z., Tang, X., Deng, G., et al. (2025). *LocAgent: Graph-Guided LLM Agents for Code Localization.* arXiv:2503.09089.
- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). *Reciprocal rank fusion outperforms Condorcet and individual rank learning methods.* SIGIR. doi:10.1145/1571941.1572114
- Gemma Team (2025). *Gemma 3 Technical Report.* arXiv:2503.19786.
- Jimenez, C. E., Yang, J., Wettig, A., et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR. arXiv:2310.06770.
- Markowitz, E., Perozzi, B., Rózemberczki, B., et al. (2026). *Google – The Gemma 4 Developer Agent Paper Track.* Kaggle.
- Ouyang, S., Yu, W., Ma, K., et al. (2024). *RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph.* arXiv:2410.14684.
- Saha, R. K., Lease, M., Khurshid, S., & Perry, D. E. (2013). *Improving bug localization using structured information retrieval.* ASE. doi:10.1109/ASE.2013.6693093
- Wang, X., Li, B., Song, Y., et al. (2024). *OpenHands: An Open Platform for AI Software Developers as Generalist Agents.* arXiv:2407.16741.
- Xia, C. S., Deng, Y., Dunn, S., & Zhang, L. (2024). *Agentless: Demystifying LLM-based Software Engineering Agents.* arXiv:2407.01489.
- Yang, J., Jimenez, C. E., Wettig, A., et al. (2024). *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.* arXiv:2405.15793.
- Zhang, Y., Ruan, H., Fan, Z., & Roychoudhury, A. (2024). *AutoCodeRover: Autonomous Program Improvement.* arXiv:2404.05427.
