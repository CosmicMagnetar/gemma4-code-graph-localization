# Where Do the Fixes Live? Code-Graph Fault Localization for Small Local Coding Agents

Code and data products for a paper submitted to the Kaggle **Google – The Gemma 4 Developer Agent Paper Track**.
The paper itself is [WRITEUP.md](WRITEUP.md).

This is a study of the 129 public tasks in the Gemma 4 Developer Agent competition. It needs no training and makes no model calls. It measures how far each reference fix is, in call-graph hops, from the symbols named in the issue. It also measures how well four rankers put an edited function near the top of the list: issue-text TF-IDF, graph hops, the provided embeddings, and reciprocal-rank fusion of these.

## Layout

| Path | What it is |
|---|---|
| `analysis/localize.py` | The full analysis (source of truth). Runs locally or on Kaggle. |
| `analysis/localize.ipynb` | The same code as a notebook, generated from the `.py` cells (`# %%`). |
| `analysis/kernel-metadata.json` | Kaggle kernel config, so the run can be pushed with the Kaggle API. |
| `results/results.json` | Output of the Kaggle run: summary metrics and one row per task. |
| `figures/make_figures.py` | Builds `figures/*.png` and prints every table in the paper from `results.json`. |

## Reproduce

**Hardware:** Kaggle CPU notebook. No GPU and no internet needed; the competition data is attached as input. Figures can be made on any machine with Python 3, numpy and matplotlib.

**Option A: Kaggle UI**
1. Join https://www.kaggle.com/competitions/gemma-4-developer-agent.
2. Open the public notebook: NOTEBOOK-LINK-TBD. Or create a new notebook from the competition's **Code** tab and import `analysis/localize.ipynb`.
3. Click **Run All**. It writes `/kaggle/working/results.json`.

**Option B: Kaggle API**
```bash
pip install kaggle numpy matplotlib
# edit "id" in analysis/kernel-metadata.json to <your-username>/gemma-4-code-graph-localization
kaggle kernels push -p analysis
kaggle kernels status <your-username>/gemma-4-code-graph-localization
kaggle kernels output <your-username>/gemma-4-code-graph-localization -p results
```

**Figures and tables**
```bash
python figures/make_figures.py results/results.json
```

**Local run (if you have the data downloaded)**
```bash
DATA_DIR=/path/to/competition/data OUT_DIR=results python analysis/localize.py
```

Notes on the data layout. On Kaggle, graphs and embeddings are named `<repo>_<base_commit>.json` and `<repo>_<base_commit>.npz`, not by instance id. The script indexes both naming schemes.

## License

Apache 2.0.
