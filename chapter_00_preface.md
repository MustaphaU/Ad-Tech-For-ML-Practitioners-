# ML and RL for Bidding and Allocation in Online Advertising

### By Nima Rafiee

---

# Preface

Every second, across billions of screens worldwide, a quiet but consequential competition unfolds. A user loads a webpage, an ad slot becomes available, and within the span of a single heartbeat -- roughly 100 milliseconds -- dozens of algorithms evaluate the opportunity, compute a bid, and one emerges as the winner. The winning ad appears on the screen. The user may click; they may not. Either way, money changes hands, data is logged, and the system learns.

This is real-time bidding, and it represents one of the largest-scale applications of machine learning and decision-making under uncertainty in the world today. The global digital advertising market exceeds $600 billion annually. The infrastructure that allocates this spending processes millions of auctions per second, each one a small optimization problem demanding predictions, strategic reasoning, and real-time control. It is, in short, a playground for the technically ambitious.

This book is written for you -- the ML or RL engineer who has been asked to bring your skills to the bidding and allocation side of advertising technology. You understand stochastic gradient descent, Markov decision processes, and policy gradient methods, but terms like "DSP," "RTB," and "GSP auction" are new to you. You have deep technical foundations and zero ad-tech context. This book bridges that gap.

## What Makes Ad Tech Unique for ML/RL Engineers

The advertising ecosystem is unusual among ML application domains for several reasons that make it particularly rewarding:

**Scale and speed.** A major demand-side platform (DSP) like The Trade Desk or Google's DV360 may evaluate hundreds of thousands of bid requests per second. Each decision must be made in under 50 milliseconds. This is not batch inference -- it is real-time prediction and decision-making at a scale that few other domains demand.

**Sequential decision-making under constraints.** An advertiser has a daily budget of, say, $10,000 and a goal to maximize conversions. How should that budget be spread across the day's auctions? Spend too fast and you exhaust the budget by noon, missing valuable evening impressions. Spend too slowly and you leave money on the table. This is a textbook reinforcement learning problem -- literally.

**Strategic interaction.** Your bidding algorithm does not operate in isolation. It competes against other algorithms in auctions, creating a multi-agent game-theoretic environment. Your optimal strategy depends on what others do, and what others do depends on you.

**Rich prediction problems.** Predicting whether a user will click an ad (CTR prediction) is one of the most studied problems in industrial ML. It combines sparse categorical features, sequential user behavior, contextual signals, and extreme class imbalance -- all at massive scale.

**Direct economic feedback.** Unlike many ML applications where the loss function is a proxy, in ad tech the model's predictions directly determine how much real money is spent. A 1% improvement in bid efficiency can translate to millions of dollars saved or earned.

> **For the RL Engineer**: If you have worked on robotics, game-playing agents, or recommendation systems, you will find that bidding shares deep structural similarities. A bidding agent observes a state (the bid request, budget remaining, time of day), takes an action (the bid amount), receives a reward (a click or conversion, minus the cost), and transitions to a new state. The key complications are partial observability (you only see outcomes for auctions you win), non-stationarity (competitors change their strategies), and hard constraints (budgets, ROI targets).

## How This Book Is Organized

The book is structured as a two-week intensive curriculum, with each chapter designed for a single day of focused study. The first week builds foundational knowledge of the advertising ecosystem, auction theory, and the core ML problems. The second week applies RL and advanced ML to bidding and allocation, culminating in production system design and research frontiers.

### Week 1: Foundations

| Day | Chapter | Topic | Hours |
|-----|---------|-------|-------|
| 1 | Ch 1 | The Online Advertising Ecosystem | 3--4 |
| 2 | Ch 2 | Auction Theory Foundations | 4--5 |
| 3 | Ch 3 | How Real-Time Bidding Works | 3--4 |
| 4 | Ch 4 | CTR Prediction: The Core ML Problem | 4--5 |
| 5 | Ch 5 | Bid Optimization and Pricing Models | 3--4 |
| 6 | Ch 6 | Deep Learning for User Response Prediction | 4--5 |
| 7 | -- | Review, Exercises, and iPinYou Dataset Lab | 4--6 |

### Week 2: ML/RL for Bidding

| Day | Chapter | Topic | Hours |
|-----|---------|-------|-------|
| 8 | Ch 7 | Formulating Bidding as an RL Problem | 4--5 |
| 9 | Ch 8 | RL Algorithms for Bidding | 4--5 |
| 10 | Ch 9 | Budget Pacing and Constrained Optimization | 3--4 |
| 11 | Ch 10 | Multi-Agent Bidding and Auction Design | 4--5 |
| 12 | Ch 11 | Production Systems: From Research to Deployment | 3--4 |
| 13 | Ch 12 | Advanced Topics and Research Frontiers | 4--5 |
| 14 | -- | Capstone Project: Build a Bidding Agent | 6--8 |

The progression is deliberate. Chapters 1--3 teach you the language and mechanics of the industry so that you can have a conversation with ad-tech colleagues. Chapters 4--6 develop the core ML models that underpin every bidding system. Chapters 7--10 are where your RL expertise becomes central -- formulating bidding as sequential decision-making and solving it. Chapters 11--12 connect everything to production reality and the research frontier.

## Conventions Used in This Book

Throughout the text, you will encounter several recurring elements:

**Mathematical notation** uses standard conventions. We write $v_i$ for bidder $i$'s valuation, $b_i$ for their bid, $F(\cdot)$ for cumulative distribution functions, and use boldface for vectors. Display equations are numbered when referenced later.

**Exercises** appear at the end of each chapter and are primarily conceptual -- thought experiments, derivations, and analytical problems. A smaller number involve implementation, but these are always supplementary to the main ideas.

**Callout boxes** highlight material that deserves special attention:

> **Key Insight**: These highlight non-obvious takeaways -- the kind of thing a senior engineer would tell you over coffee.

> **Industry Example**: These ground the theory in real systems at Google, Meta, Alibaba, The Trade Desk, and others.

> **For the RL Engineer**: These explicitly bridge ad-tech concepts to your existing ML/RL knowledge.

> **Historical Note**: These provide context for why the industry works the way it does.

**Diagrams** use the Mermaid format for flowcharts, sequence diagrams, and architecture diagrams, which render natively in GitHub, VS Code, and Obsidian.

## Key References

This book synthesizes ideas from several foundational sources. You do not need to read all of them upfront, but you should know they exist and return to them as your understanding deepens.

**Wang, Zhang, and Yuan (2017)**, *Display Advertising with Real-Time Bidding and Behavioural Targeting* (arXiv:1610.03013), is the closest existing textbook to what we cover. It is freely available and provides excellent technical depth on RTB mechanics and early ML approaches. Our treatment builds on theirs with updated coverage of RL methods and modern auction formats.

**Nisan, Roughgarden, Tardos, and Vazirani (2007)**, *Algorithmic Game Theory* (Cambridge University Press), is the definitive reference for auction theory and mechanism design. Chapter 2 of this book draws heavily on their treatment.

**Aggarwal et al. (2024)**, *Auto-bidding and Auctions in Online Advertising: A Survey* (arXiv:2408.07685), is a comprehensive survey from Google Research covering the state of the art in automated bidding -- the very topic you will be working on.

**Roughgarden (2016)**, *Twenty Lectures on Algorithmic Game Theory*, provides an accessible entry point to auction theory. His Stanford CS364A video lectures are freely available on YouTube and are an excellent companion to Chapter 2.

**Hartline (2012)**, *Mechanism Design and Approximation*, is a freely available advanced reference (jasonhartline.com/MDnA/) for those who want to go deeper into optimal mechanism design.

## Recommended Software

Several open-source tools will be useful as you work through the exercises and capstone project:

- **Auction Gym** (amazon-science/auction-gym) -- A simulation environment for auction mechanisms, useful for experimenting with bidding strategies in controlled settings.
- **iPinYou Dataset Tools** (wnzhang/make-ipinyou-data) -- Scripts for preparing the iPinYou RTB benchmark dataset, one of the few publicly available real-world RTB datasets.
- **OpenSpiel** (google-deepmind/open_spiel) -- DeepMind's framework for research in games, including auction games and multi-agent learning.

A standard Python scientific computing environment (NumPy, pandas, matplotlib, scikit-learn, PyTorch, Gymnasium) is assumed throughout.

## A Note on Pace

This book is dense by design. It compresses what might otherwise take months of on-the-job osmosis into two weeks of structured study. Some chapters -- particularly Chapters 2 (auction theory) and 7 (RL formulation) -- may require multiple readings. That is expected. The goal is not to memorize every formula but to build a mental model robust enough that when you encounter a real bidding system, you can reason about it from first principles.

---

*Let's begin.*
