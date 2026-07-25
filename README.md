<div align="center">

# Ad Tech for ML Practitioners

### Machine Learning and Reinforcement Learning for Online Advertising


[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
![Chapters](https://img.shields.io/badge/Chapters-13-blue)
![Pages](https://img.shields.io/badge/PDF_Pages-220-green)
![Diagrams](https://img.shields.io/badge/Mermaid_Diagrams-64-orange)

*A two-week intensive textbook for ML/RL engineers entering the ad tech domain.*

[Read the PDF](ML_RL_Bidding_Textbook.pdf) | [Read Online (Markdown)](chapter_00_preface.md) | [Combined Single File](combined.md)

</div>

---

## Who This Book Is For

You are an **ML or RL engineer** who has been asked to apply your skills to bidding and allocation in advertising technology. You understand gradient descent, MDPs, and policy gradients, but terms like "DSP," "RTB," and "GSP auction" are new to you. **This book bridges that gap.**

The book takes you from zero ad-tech knowledge to being able to design, implement, and deploy ML/RL-based bidding systems — in two weeks of focused study.

## What You'll Learn

```
Week 1: Foundations                          Week 2: ML/RL for Bidding
─────────────────────                        ──────────────────────────
 Advertising ecosystem & key players          RL formulation of bidding (MDP design)
 Auction theory (Vickrey, GSP, VCG)           DQN, TD3, SAC, CQL for bid optimization
 Real-time bidding mechanics (OpenRTB)        Budget pacing (PID, Lagrangian dual)
 CTR prediction (LR, FM, FFM)                 Multi-agent bidding & mechanism design
 Bid optimization & bid shading               Production systems at scale
 Deep learning for CTR (Wide&Deep,            Research frontiers (diffusion models,
  DeepFM, DIN, DCN-V2)                         LLMs, privacy, causal inference)
```

## Table of Contents

### Week 1: Foundations

| Day | Chapter | Topic | Key Concepts |
|:---:|---------|-------|-------------|
| — | [Preface](chapter_00_preface.md) | How to use this book | Prerequisites, conventions, references |
| 1 | [Chapter 1](chapter_01_ecosystem.md) | The Advertising Ecosystem | DSPs, SSPs, exchanges, pricing models, OpenRTB |
| 2 | [Chapter 2](chapter_02_auction_theory.md) | Auction Theory | Vickrey, first-price, GSP, VCG, Myerson, Revenue Equivalence |
| 3 | [Chapter 3](chapter_03_rtb.md) | Real-Time Bidding | Bid requests, auction mechanics, bid shading, feedback loops |
| 4 | [Chapter 4](chapter_04_ctr_prediction.md) | CTR Prediction | Logistic regression, feature hashing, FTRL, FM, FFM, calibration |
| 5 | [Chapter 5](chapter_05_bid_optimization.md) | Bid Optimization | Bidding equation, Lagrangian optimization, bid landscapes |
| 6 | [Chapter 6](chapter_06_deep_learning_ctr.md) | Deep Learning for CTR | Wide & Deep, DeepFM, DIN, DCN-V2, multi-task learning |

### Week 2: ML/RL for Bidding

| Day | Chapter | Topic | Key Concepts |
|:---:|---------|-------|-------------|
| 8 | [Chapter 7](chapter_07_rl_formulation.md) | RL Formulation | Bidding MDP, state/action/reward design, simulators |
| 9 | [Chapter 8](chapter_08_rl_algorithms.md) | RL Algorithms | DQN, TD3, SAC, constrained RL, offline RL (CQL, BCQ, IQL) |
| 10 | [Chapter 9](chapter_09_budget_pacing.md) | Budget Pacing | PID control, Lagrangian dual, traffic forecasting |
| 11 | [Chapter 10](chapter_10_multiagent_auction_design.md) | Multi-Agent & Auctions | Nash equilibrium, mechanism design, auto-bidding equilibria |
| 12 | [Chapter 11](chapter_11_production.md) | Production Systems | Model serving, feature stores, latency budgets, A/B testing |
| 13 | [Chapter 12](chapter_12_advanced_frontiers.md) | Research Frontiers | Diffusion models, LLMs, privacy, causal inference, fairness |
| 14 | [Capstone](chapter_13_capstone.md) | Build a Bidding Agent | End-to-end project with evaluation rubric |

## How to Read

Each chapter is a standalone Markdown file designed for a single day of focused study (3-5 hours). The content uses:

- **Rich narrative prose** — concepts explained like an O'Reilly textbook, not a code tutorial
- **Mermaid diagrams** — 64 architecture diagrams, flowcharts, and sequence diagrams (render natively on GitHub)
- **LaTeX math** — proper formulations for auction theory, optimization, and RL
- **Callout boxes** — Key Insights, Industry Examples, "For the RL Engineer" bridges, Historical Notes
- **Conceptual exercises** — thought experiments and analytical problems at the end of each chapter

> **Best viewing experience**: GitHub renders Mermaid diagrams natively. You can also use VS Code, Obsidian, or download the [PDF version](ML_RL_Bidding_Textbook.pdf).

## Key Formulas at a Glance

| Domain | Formula | Chapter |
|--------|---------|:-------:|
| **Bidding** | $\text{bid} = v \times \hat{p}(\text{click}) \times \hat{p}(\text{convert}) \times \frac{1}{1+\lambda}$ | 5 |
| **First-Price Equilibrium** | $b^*(v) = \frac{n-1}{n} \cdot v$ for $v \sim U[0,1]$ | 2 |
| **Optimal Reserve** | $r^* = \frac{v_{\max}}{2}$ (Myerson) | 2 |
| **FM Interaction** | $\sum_i \sum_{j>i} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$ | 4 |
| **PID Pacing** | $m_t = K_p e_t + K_i \sum e + K_d \Delta e$ | 9 |
| **Bellman (Bidding)** | $Q^*(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q^*(s', a')]$ | 8 |
| **Constrained RL** | $\mathcal{L} = \mathbb{E}[V] - \lambda(\mathbb{E}[\text{cost}] - B)$ | 8 |

## Essential References

| Paper | Year | Topic |
|-------|:----:|-------|
| Wang, Zhang, Yuan — *Display Advertising with RTB* | 2017 | The definitive RTB textbook ([arXiv:1610.03013](https://arxiv.org/abs/1610.03013)) |
| Aggarwal et al. — *Auto-bidding and Auctions Survey* | 2024 | Comprehensive survey from Google ([arXiv:2408.07685](https://arxiv.org/abs/2408.07685)) |
| McMahan et al. — *Ad Click Prediction: A View from the Trenches* | 2013 | Google's production CTR system (KDD 2013) |
| Edelman, Ostrovsky, Schwarz — *GSP Auction* | 2007 | How search ad auctions work (AER Vol 97) |
| Cai et al. — *RTB by Reinforcement Learning* | 2017 | Foundational RL for bidding ([arXiv:1701.02490](https://arxiv.org/abs/1701.02490)) |
| Conitzer et al. — *Pacing Equilibrium* | 2019 | Why first-price with pacing works ([arXiv:1811.07166](https://arxiv.org/abs/1811.07166)) |

## Tools and Datasets

| Tool | Purpose |
|------|---------|
| [AuctionGym](https://github.com/amazon-science/auction-gym) | Bidding simulation environment (Amazon Science) |
| [iPinYou Dataset](https://github.com/wnzhang/make-ipinyou-data) | Real RTB benchmark data |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel) | Game theory / multi-agent framework (DeepMind) |
| [NeurIPS AIGB Baseline](https://github.com/alimama-tech/NeurIPS_Auto_Bidding_AIGB_Track_Baseline) | Auto-bidding competition baseline (Alibaba) |

## Video Lectures

| Course | Instructor | Coverage |
|--------|-----------|----------|
| [Stanford CS364A](https://www.youtube.com/playlist?list=PLEGCF-WLh2RJBqmxvZ0_ie-mleCFhi2N4) | Tim Roughgarden | Auction theory, mechanism design |
| [Stanford CS364B](https://www.youtube.com/playlist?list=PLEGCF-WLh2RI77PL4gwLld_OU9Zh3TCX9) | Tim Roughgarden | Advanced mechanism design |

## Building the PDF

The repository includes a pre-built PDF. To rebuild it after making changes:

```bash
# Requires: pandoc, xelatex (TeX Live), mmdc (mermaid-cli)
bash build_pdf.sh
```

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Contributing

Found an error, have a suggestion, or want to add a section? Issues and pull requests are welcome.

---

<div align="center">

**If you find this book useful, please give it a star!**

</div>
