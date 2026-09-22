# arXiv Submission Package

This folder contains the complete, self-contained LaTeX source, figures, and empirical data artifacts for the paper:

> **Distilling Lexical Product Associations into Deep Transformers: An Extreme Multi-Label Approach for Natural Language E-Commerce Search**  
> *Sunnidhya Roy and Samarpita Bhaumik (IIIT Bangalore)*

---

## Directory Structure

```
arxiv_submission/
├── main.tex                       # Primary LaTeX manuscript (arXiv-formatted)
├── figures/                       # High-resolution (300 DPI) publication figures
│   ├── fig_category_distribution.png
│   ├── fig_text_length_histogram.png
│   ├── fig_baseline_comparison.png
│   └── fig_ranking_curves.png
└── data/                          # Ground-truth numerical data & experimental logs
    ├── baseline_metrics.json      # Corrected TF-IDF teacher ranking metrics
    ├── validation_metrics.json    # DistilBERT student classification & ranking metrics
    └── qualitative_benchmark.csv  # 10 natural language query archetype evaluation
```

---

## How to Compile Locally

You can compile the manuscript using `pdflatex` directly:

```bash
pdflatex main.tex
pdflatex main.tex
```

*(Two passes are recommended to resolve cross-references and table numbering).*

---

## How to Submit to arXiv (arxiv.org)

1. When submitting to arXiv's submission portal, upload either:
   * A zip archive containing `main.tex` and the `figures/` directory, OR
   * Upload `arxiv_submission.zip` generated in the root directory.
2. Note: Do **not** upload external 100MB+ model weights or raw datasets to the arXiv LaTeX compiler.
3. arXiv will run `pdflatex` automatically on Linux and produce the published paper PDF.

---

## Ground-Truth Alignment

All values in Tables 1, 2, 3, 4, and 5 and in-text numbers are strictly synchronized with:
* `data/validation_metrics.json`
* `data/baseline_metrics.json`
* `data/qualitative_benchmark.csv`
* `rs-final-arxiv.ipynb`
