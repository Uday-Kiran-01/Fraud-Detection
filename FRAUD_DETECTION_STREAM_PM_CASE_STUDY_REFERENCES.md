# Personal Product Case Study: Ad Click Fraud Detection (Streaming Simulator & Production Risk Engine)

This document breaks down the product thinking, technical choices, and candid post-mortem behind **Ad Click Fraud Detection (Streaming Simulator & Production Risk Engine)**. As an aspiring Product Manager targeting AdTech, FinTech, and Data Platform systems, this project represents my personal reflection on bridging the dangerous gap between offline data science fantasies and the live operational realities of high-velocity streaming risk engineering.

---

### 1. User Discovery (The "Offline Model Fantasy" Problem)
**The Raw Problem:** In data science competitions and classroom projects, machine learning models are trained on clean, static CSV files and judged on a single offline metric like ROC-AUC. But in real-world ad networks, clicks do not sit politely in a CSV. They stream in continuously at thousands of events per second, with sudden coordinated botnet bursts striking for 20 minutes before evaporating. When an offline model is deployed into live production without streaming validation, systems frequently crash under arrival backpressure, inference latency spikes out of control, and risk teams flag thousands of legitimate users because the algorithm was never tested against dynamic traffic pacing.  
**The Questioning:** Why do machine learning models that boast 99% accuracy in offline notebooks fail so miserably in live ad tech environments? Because offline metrics optimize for mathematical perfection on static data, while live systems live and die on streaming latency, queue stability, and strict business review budgets. A model is completely useless if it takes 500 milliseconds to score a click in a high-velocity ad attribution pipeline.  
**The Authentic Goal:** I did not set out to build another Kaggle competition script. I wanted to build a production-oriented machine learning risk platform that answers a critical operational question: *How does an ad fraud model actually behave when subjected to live streaming traffic bursts, and how do we ensure that fraud flags strictly match the business's human review capacity?*

---

### 2. Expectations vs Reality (The Static CSV Illusion & The Streaming Reality Check)

#### What I Expected:
*   **The Offline Fantasy:** I initially assumed that training a high-performing LightGBM model with tuned tree depths on historical data would be the primary achievement of the project.
*   **The Default Threshold Assumption:** I assumed that using standard classification thresholds or maximizing the F1 score across the entire test set would naturally translate into business success.

#### What Reality Proved (The Operational Awakening):
*   **The Class Imbalance Trap:** In clickstream fraud, positive conversion fraud accounts for less than 0.2% of total traffic. A naive model that predicts "non-fraud" 100% of the time achieves 99.8% accuracy while letting millions of dollars in ad spend bleed into fraudulent publisher channels.
*   **Static Scoring Ignores Latency:** Testing on a static CSV gives zero indication of how feature extraction and model scoring perform under memory constraints. If feature extraction takes 10 milliseconds per click during a 1,000 click/sec traffic spike, the server runs out of memory and crashes.
*   **The Flag Budget Reality:** Real fraud operations teams do not review arbitrary lists of flagged clicks. They have fixed staff capacity: *"Our fraud analysts can only inspect 1% of total daily clicks."* If the model flags 5% of traffic, 80% of those flags are ignored. If it flags 0.1%, fraud slips through unmonitored.
*   **The Pivot to Stream Simulation:** I realized that before deploying any model, you must build a **Streaming Flight Simulator**: a tool that replays real click traffic at production pacing (e.g., 50 events/sec in 5,000-row micro-batches) to pressure-test latency, memory footprint, and flag budgets under live conditions.

---

### 3. The Automation Journey (From Standalone Scripts to a Unified Streaming Engine)

**The Initial Fragmented Approach:**
*   Like most ML projects, the codebase began as disjointed Python scripts: one script for feature generation, another for training LightGBM, and an ad-hoc Jupyter notebook for generating scatter plots.
*   Testing streaming behavior required manually editing hard-coded sleep timers and loop ranges inside a scratch script.

**Why the Fragmented Approach Failed:**
*   Without a unified CLI architecture, running an end-to-end evaluation required executing four different scripts in sequence with manual file path coordination.
*   Comparing model performance head-to-head was painful: candidate models had slightly different preprocessing logic, making fair A/B benchmarking impossible.

**The Pivot to Production Architecture:**
*   I refactored the entire system into a clean, modular CLI entrypoint (`main.py`) exposing 7 single-responsibility verbs: `eda`, `featurize`, `train`, `infer`, `simulate`, `abtest`, and `report`.
*   I built **`src/stream_simulator.py`**, enabling configurable arrival rates (`--rate 50`), micro-batch chunking (`--batch 5000`), and real-time latency and memory telemetry logging.
*   I built dynamic flag-budget logic (`--flag-rate 0.01`), allowing risk operators to command exact review volumes and having the mathematical engine compute the precise quantile cutoff on the fly.
*   I automated executive reporting via `src/report.py`, compiling candidate model metrics, threshold curves, and PR distributions into a polished Markdown dossier (`ab_report.md`).

---

### 4. What the Data Actually Taught Me (Traffic Pacing & Commercial Risk Calibration)

#### 1. Micro-Batches Defeat Queue Backpressure
*   **The Latency Bottleneck:** Scoring click events row-by-row in a Python loop resulted in massive latency overhead (over 2 milliseconds per record), creating catastrophic queue backlog.
*   **Vectorized Micro-Batching:** By batching streaming events into 5,000-row micro-chunks and scoring them through LightGBM's vectorized C++ backend, inference latency plummeted to sub-20ms per batch (under 0.004 ms per record).
*   **The PM Takeaway:** In high-throughput data platforms, architectural packaging and batch vectorization matter just as much as model algorithm selection.

#### 2. The Commercial Alignment of Flag Budgets
*   **The Fallacy of Arbitrary Cutoffs:** Defaulting to a `0.50` probability cutoff in severe class imbalance either flags excessive clean traffic or misses entire fraud rings.
*   **Quantile Flag Budgeting:** When we calibrate the decision engine to an operational 1% review budget, the model isolates the top 99th percentile highest-risk clicks. Within that tightly bounded 1% volume, the precision exceeds 90%, capturing the overwhelming majority of coordinated bot fraud without overwhelming human auditors.

---

### 5. Future Value: A Reusable Blueprint for Real-Time Risk Platforms
*   **A Pre-Deployment Verification Standard:** The stream simulator serves as a reusable architectural template for any data team preparing to transition models from offline training into live streaming Kafka or Flink pipelines.
*   **Cross-Domain Applicability:** The combination of operational flag budgeting, threshold sweeps, and automated A/B reporting transfers directly to payment fraud detection, account takeover monitoring, and credit underwriting systems.
*   **Defensible Executive Governance:** The project pairs mathematical rigor with automated executive reporting, giving business stakeholders the transparent data evidence required to defend automated fraud blocks to commercial partners.

---

### 6. Product Management Retrospective & Intuition
1.  **Production Readiness Over Theoretical Math:** A machine learning model is not a product; it is merely one component of a decision pipeline. A PM must focus on how the model fits into the operational workflow: scoring latency, memory stability, and human review limits.
2.  **Respecting Commercial Partner Relationships:** In ad tech, false positive fraud accusations can destroy publisher contracts and wipe out revenue. Calibrating models through dynamic threshold sweeps protects commercial trust.
3.  **The Empathy of Operational Stress:** Having watched data systems break under unpredicted traffic surges, I approach product engineering with defensive empathy. Building a streaming simulator that pressure-tests algorithms before live deployment protects both company capital and engineering peace of mind.
