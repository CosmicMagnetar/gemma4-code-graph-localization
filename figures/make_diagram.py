"""Draw the TagTrail pipeline diagram (figures/fig0_tagtrail_pipeline.png). No data needed."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
fig, ax = plt.subplots(figsize=(7.6, 2.7), dpi=170)
ax.set_xlim(0, 10)
ax.set_ylim(0.75, 4.4)
ax.axis("off")


def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec="#333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2, wrap=True)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#333"))


ax.text(0.1, 4.2, "Once per repository snapshot (offline)", fontsize=8, weight="bold", color="#555")
box(0.1, 2.9, 1.9, 1.0, "Competition code graph\n(classes, functions,\n`calls` edges)", "#E8EEF7")
box(2.6, 2.9, 2.0, 1.0, "Module outlines\nsignature + first\ndocstring line", "#E8EEF7")
box(5.2, 2.9, 2.1, 1.0, "Gemma 4 reads each\nmodule once and writes\n8-12 behaviour tags", "#FBE5D6")
box(7.9, 2.9, 2.0, 1.0, "TagTrail index\nmodule -> tags\n(cached, reused)", "#FBE5D6")
arrow(2.0, 3.4, 2.6, 3.4)
arrow(4.6, 3.4, 5.2, 3.4)
arrow(7.3, 3.4, 7.9, 3.4)

ax.text(0.1, 2.35, "Per bug report (online)", fontsize=8, weight="bold", color="#555")
box(0.1, 0.9, 1.9, 1.1, "Bug report\n(often no code\nnames at all)", "#EAF4EA")
box(2.6, 0.9, 2.0, 1.1, "Gemma 4: one-line\nreading of the bug\n+ tags in the same\nvocabulary", "#FBE5D6")
box(5.2, 0.9, 2.1, 1.1, "Match issue tags to\nmodule tags (TF-IDF\nover tag words)", "#E8EEF7")
box(7.9, 0.9, 2.0, 1.1, "Fuse with TF-IDF and\ngraph hops (RRF)\n-> ranked reading list", "#EAF4EA")
arrow(2.0, 1.45, 2.6, 1.45)
arrow(4.6, 1.45, 5.2, 1.45)
arrow(7.3, 1.45, 7.9, 1.45)
arrow(8.9, 2.9, 6.6, 2.0)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig0_tagtrail_pipeline.png"))
print("wrote fig0_tagtrail_pipeline.png")
