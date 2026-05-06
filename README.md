# Exact Counting in Recurrent Neural Networks

**Can we make RNNs count perfectly — not approximately, but exactly — and generalise this to sequences far longer than anything they trained on?**

This repository contains the code and experiments from my MSc dissertation (City, University of London, 2024), which builds on [El-Naggar (2023)](https://openaccess.city.ac.uk/id/eprint/33010/) to investigate two approaches for inducing exact counting behaviour in RNNs.

---

## The problem

Standard RNNs (and even LSTMs) trained on sequence classification tasks like [Dyck-1](https://en.wikipedia.org/wiki/Dyck_language) tend to learn statistical approximations of counting rules, not the rules themselves. They work well within their training distribution but fail to generalise to longer sequences — a symptom of the broader problem of **systematic generalisation** in neural networks.

A Dyck-1 sequence is a string of balanced brackets, e.g. `(())()` is valid, `(()` is not. Recognising these requires a network to track a running count with exact integer semantics — something continuous-valued hidden states are not naturally suited for.

---

## Approaches

### 1. Discrete Full Range Counter (DFRC)
An extension of El-Naggar's Discrete Non-Negative Counter (DNNC). The DNNC tracks only non-negative counts; the DFRC generalises this to support the full integer range — both positive and negative — which is a natural requirement for balanced bracket recognition.

The DFRC uses a two-stack architecture: the first stack tracks the running count over the full range, and the second stack handles category classification. A key design choice is applying a `tanh` function to the second stack (taking the first stack's output as input), which maps category outputs to `[-1, 1]` and provides a smooth transition between categories rather than a hard discrete switch. This also allows gradients to flow more cleanly during backpropagation, using the `tanh` derivative `1 - tanh²` in the backward pass.

The DFRC plugs into any RNN backbone (vanilla RNN, GRU, or LSTM) and remains compatible with standard backpropagation.

### 2. Parameter Freeze Counter (PFC)
A different angle: instead of adding a module, can we *configure* a standard RNN to count exactly by pre-determining and freezing specific weight values? The PFC investigates which parameter configurations enable systematic counting and whether freezing them during training is viable.

---

## Key results

All results below are on the **large dataset** (training on sequences up to 50 tokens, evaluated on held-out sequences up to 100 tokens).

| Model | Counter | Training (50 tokens) | Short test (50 tokens) | Long test (100 tokens) |
|-------|---------|:--------------------:|:---------------------:|:----------------------:|
| RNN | DNNC | 57.6% | 54.6% | 52.8% |
| GRU | DNNC | 72.0% | 70.5% | 64.18% |
| LSTM | DNNC | 99.86% | 99.87% | 95.6% |
| RNN | DFRC | 52.4% | 50.0% | 50.0% |
| GRU | DFRC | 71.86% | 69.57% | 64.66% |
| **LSTM** | **DFRC** | **100%** | **100%** | **99.84%** |
| RNN | PFC | 53.4% | 53.4% | 51.6% |
| GRU | PFC | 52.99% | 51.03% | 49.94% |
| LSTM | PFC | 52.8% | 53.7% | 52.5% |

*Long test accuracy is the meaningful metric — it measures generalisation to sequences twice the length of anything seen during training.*

**Main finding:** LSTM + DFRC achieved near-perfect generalisation to unseen long sequences (99.84%), significantly outperforming the DNNC baseline (95.6%) on the same backbone. The PFC approach did not scale effectively to the large dataset, suggesting that fixed-parameter strategies require further architectural work to be viable at scale.

---

## Repository structure

```
├── src/
│   ├── data.py         # Dyck-1 sequence generator
│   ├── models.py       # RNN / GRU / LSTM base architectures
│   ├── dfrc.py         # Discrete Full Range Counter module
│   ├── pfc.py          # Parameter Freeze Counter logic
│   └── train.py        # Training and evaluation loop
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_baseline_experiments.ipynb
│   ├── 03_dfrc_experiments.ipynb
│   ├── 04_pfc_experiments.ipynb
│   └── 05_results_analysis.ipynb
├── results/            # Saved metrics and plots
└── requirements.txt
```

---

## Running the experiments

*Full instructions coming soon. Requires Python 3.9+, PyTorch, NumPy, Pandas, scikit-learn.*

```bash
pip install -r requirements.txt
```

See the notebooks in `notebooks/` for step-by-step walkthroughs of each experiment.

---

## Background reading

- El-Naggar, N. (2023). *What Really Counts: Theoretical and Practical Aspects of Counting Behaviour in Simple RNNs.* — the paper this work directly extends
- Lake & Baroni (2018). *Generalization without systematicity* — motivation for the systematic generalisation problem
- Gers & Schmidhuber (2001). *LSTM recurrent networks learn simple context-free and context-sensitive languages*

---

## About

MSc Data Science dissertation, City, University of London (2024). Supervised by Dr Tillman Weyde.

The broader question this work touches on — whether neural networks can be made to follow rules rather than approximate them — remains open and connects to current work on reasoning, compositionality, and interpretability in large models.
