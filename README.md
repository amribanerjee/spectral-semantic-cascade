# Multi-Agent Semantic Drift & Consensus Collapse

This notebook studies how noise ("semantic drift") propagates through networks of communicating agents, and how network topology affects whether the agents converge to a shared state or collapse into disagreement. It combines graph spectral theory, information theory, a multi-agent consensus simulation, and a clinical NLP embedding experiment.

Three network topologies are compared throughout:

- **Erdős–Rényi (ER)** — random graph
- **Barabási–Albert (BA)** — scale-free / preferential-attachment graph
- **Watts–Strogatz (WS)** — small-world graph

## Contents

The notebook has 5 independent stages, each runnable on its own.

| Stage | Cell | What it does |
|---|---|---|
| 1. Spectral Analysis | `execute_stage_one()` | Builds the three topologies, computes the graph Laplacian, its eigenvalues/eigenvectors (Fiedler value, algebraic connectivity, spectral gap), and plots the Fiedler vector distribution for each network. |
| 2. Semantic Drift (single node) | `execute_stage_two()` | Starts from a random probability distribution over a 512-token vocabulary and repeatedly injects Gaussian noise, renormalizing at each step. Tracks KL divergence from the original distribution and entropy over time, for several noise variances. |
| 3. Consensus Collapse Simulation | `execute_stage_three()` | Simulates a multi-agent system updating its state via a noisy weighted-averaging (transition-matrix) process on each topology, and tracks how state variance evolves — i.e. whether agents converge or diverge under a fixed noise level. |
| 4. Clinical Embedding Drift | `execute_stage_four()` | Loads `Bio_ClinicalBERT`, embeds a sample clinical sentence, and measures how cosine similarity to the original embedding degrades as Gaussian noise of increasing variance is added — a proxy for semantic drift in a real language model's representation space. |
| 5. Combined Phase/Trajectory Analysis | `execute_stage_five()` | Repeats the consensus simulation across multiple random trials per topology, and additionally analyzes the eigenstructure of the update ("phase transition") matrix — eigenvector concentration and mean/variance trajectories with error bands. |

## Outputs

Each stage saves one or more plots as PDFs in the working directory and prints a summary table (Markdown-formatted via `pandas`) to stdout:

- `stage1_spectral_distribution.pdf`
- `stage2_semantic_divergence.pdf`
- `stage3_consensus_cascade.pdf`
- `stage4_clinical_embedding_drift.pdf`
- `stage5_graph1_phase_transition.pdf`
- `stage5_graph2_eigenvector_concentration.pdf`
- `stage5_graph3_variance_trajectories.pdf`

## Requirements

```
numpy
scipy
pandas
matplotlib
networkx
torch
transformers
tabulate   # required by pandas .to_markdown()
```

Install with:

```bash
pip install numpy scipy pandas matplotlib networkx torch transformers tabulate
```

Stage 4 downloads the `emilyalsentzer/Bio_ClinicalBERT` model from the Hugging Face Hub on first run, so an internet connection (and a few hundred MB of disk space) is needed for that stage.

## Usage

Run the whole notebook top to bottom, or execute an individual stage's function after running its cell, e.g.:

```python
execute_stage_one()
execute_stage_three()
```

All stages seed their random number generators (`seed=42`), so results are reproducible across runs.

## Notes

- Stage 1 and Stage 3/5 both build the same three topologies (`n=150` nodes) independently — if you modify graph parameters, update them in each stage.
- Stage 3 uses a fixed noise level (`critical_noise = 0.05`) to illustrate consensus collapse near a "critical" drift threshold; adjust `gamma` (agent inertia) or `critical_noise` to explore other regimes.
- Stage 4 runs fine on CPU for the single sentence used, but will be slow if extended to larger batches — a GPU is recommended for heavier use.
