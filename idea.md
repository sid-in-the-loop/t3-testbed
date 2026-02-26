T3: Test-Time Threading
A Training-Free Framework for Parallel Search Coverage in Agentic QA
Sidhaarth Sredharan  •  LTI, Carnegie Mellon University  •  February 2026
1. Motivation
Agentic search systems - models that iteratively issue queries to retrieve information - fail in a characteristic way: they get stuck. A ReAct-style agent given the question "Who directed the film starring the actress who won the 1994 Best Actress Oscar?" may spend its entire compute budget on variations of "1994 Oscar Best Actress", never pivoting to retrieve film or director information. The agent is anchored.

We call this failure mode anchor collapse, with two distinct manifestations:

Intra-Thread Collapse: Within a single trajectory, the agent issues semantically similar queries, retrieving overlapping documents and making no progress. Obama birthplace → where was Barack Obama born → Obama birth city → ...
Inter-Thread Collapse: When k parallel threads are spawned without diversity enforcement, they independently converge to the same query region. Thread 1: Obama birthplace | Thread 2: birth of Obama | Thread 3: where Obama was born.

Our pilot study confirms the key diagnostic: pass@k (oracle-best of k independently sampled trajectories) vastly exceeds pass@1, especially for small models. This gap means the correct solution path exists within the model's trajectory space - the bottleneck is finding it, not generating it.
Kinda plot we need to generate to see if the concept works 


Crucially, this is not a verification problem. Recent work (General AgentBench, Feb 2026) shows that even GPT-5 as an external verifier fails to select the correct trajectory from a sampled set. We argue this is because the candidates are redundant - when all k trajectories collapse onto the same answer via the same queries, no selector can save you. The bottleneck is upstream, in generation diversity.

2. Proposal: T³
T³ is a training-free inference framework that closes the pass@k gap by turning wasted parallel compute into genuine search coverage. The core mechanism is simple:

Spawn k parallel search threads, each seeded with a diversity prompt that steers it toward a distinct region of query space.
Execute each thread as an independent ReAct loop (Thought → Search → Observe → repeat).
Synthesize: each thread produces a compressed evidence summary; a single parent model call sees all summaries and produces the final answer.

The synthesis step is critical. Rather than selecting one of k trajectories (where self-choice fails), the parent model reasons jointly over all k evidence streams. This sidesteps verification entirely - diversity of evidence, not selection, drives the gain.

Connection to s1



s1
T³ (Proposed)
Problem
Collapses to short answers
Collapses to similar queries
Intervention
Append “Wait” → rethink
Seed k threads → diverse search
Scaling axis
Sequential token budget
Parallel threads × steps
Key hypothesis
pass@1 improves with budget
pass@1 improves with thread count
Training required
1K examples (SFT)
None


T³ is the conceptual analog of s1 for search: just as s1 appends “Wait” to force rethinking within a single chain, T³ forks into parallel threads to force exploration of distinct query regions.

Three Variants

T³ Fixed: k is hardcoded; diversity seeds are hand-crafted per question (e.g., direct entity lookup, alias expansion, entity-chain traversal). Proof of concept.
T³ Dynamic: The parent LM decides how many threads to spawn and what seeds to use, given the question. Tests whether the model can self-organize diversity.
T³ DPP: A Determinantal Point Process kernel replaces hand-crafted seeds. N candidate queries are sampled from the LM; k are selected to maximize diversity subject to relevance-to-missing-information scoring (see Section 4). Principled version of Fixed.

3. Positioning vs. Existing Work

Dimension
Search-o1 / ReAct
Multi-Agent (MIRAGE)
T³ (Proposed)
Parallelism
None (sequential)
Multiple agents
One agent, parallel threads
Diversity source
N/A
Agent heterogeneity (assumed)
Enforced via seeds / DPP
Failure mode studied
None
None
Intra + inter-thread collapse
Training required
Yes (RL)
Often yes
None
Pass@k as diagnostic
No
No
Yes
Verification required
No
Yes (selection)
No (synthesis sidesteps it)


Key distinction: multi-agent work assumes diversity; T³ measures its absence and enforces it.

4. DPP and Gap-Driven Relevance (T³ DPP)
The DPP variant replaces hand-crafted seeds with a formal diversity mechanism. At each spawn decision, the model samples N candidate queries and selects k via a DPP kernel:

L(i,j) = relevance(qᵢ) × K(qᵢ, qⱼ) × relevance(qⱼ)

where K is a diversity kernel (e.g., 1 - cosine similarity of query embeddings) and relevance is scored against missing information rather than the original query.
For benchmarks where decomposition is not clean (HLE, GAIA, causal chains), the gap-driven score degrades gracefully: diversity at each step still explores different entry points into the same hop, raising the probability that at least one thread finds a productive path. The decomposition is not required for T³ to work.

5. Proposed Experiments
The paper follows a diagnosis -> prescription narrative.

Exp. A: Collapse Characterization (Diagnosis)
Run single-agent ReAct trajectories across model scales. Measure ITC, ATC, QNS per model to establish whether a collapse gradient exists with scale.
Models: Llama-3B, Llama-7B, Llama-70B, GPT-4o-mini, GPT-4o
Benchmarks: HotpotQA (proof of concept), then MuSiQue, 2WikiMultihopQA, Bamboogle

Exp. B: Pass@k Gap (Oracle Ceiling)
For each model, sample k ∈ {1, 2, 4, 8, 16} independent trajectories and compute oracle accuracy. Confirm pass@k ≫ pass@1 and correlate gap size with collapse rate from Exp. A.

Exp. C: T³ Evaluation (Prescription)
Compare all T³ variants under compute-matched conditions against:
Single-thread ReAct baseline (pass@1)
Naive parallel baseline (k threads, no diversity enforcement)
s1-style sequential scaling (1 thread, k× token budget)
T³ Fixed, T³ Dynamic, T³ DPP

Primary question: does structured diversity close more of the oracle gap than sequential depth, at equal compute?

Exp. D: Generalization
Extend to short-answer benchmarks beyond multi-hop QA: Natural Questions, TriviaQA. Then long-form report generation after short-answer is complete. Optionally probe on coding tasks (SWE-bench subset) to test whether threading generalizes beyond search.

Benchmark Suite

Benchmark
Type
Hops
Priority
HotpotQA
Multi-hop QA
2
PoC (first)
MuSiQue
Multi-hop QA
2-4
Primary
2WikiMultihopQA
Multi-hop QA
2
Primary
Bamboogle
Compositional QA
2+
Primary
Natural Questions
Single-hop QA
1
Generalization
TriviaQA
Single-hop QA
1
Generalization
Long-form report gen
Open-ended
N/A
Phase 2


6. Metrics

Symbol
Name
Definition
QNS
Query Novelty Score
Avg. pairwise TF-IDF dissimilarity across all queries in a trajectory
ITC
Intra-Thread Collapse
Proportion of steps where query similarity to any prior query > τ
ATC
Inter-Thread Collapse
Proportion of thread pairs whose query centroid similarity > 0.6
Pass@k
Oracle Accuracy
EM when the best of k trajectories is oracle-selected
T³ EM
T³ Exact Match
EM from T³ synthesis vs. single-thread baseline


QNS, ITC, and ATC are standalone diagnostic contributions for characterizing search trajectory quality, independent of T³.

7. Ablation Plan
Thread count k: sweep k ∈ {1, 2, 3, 4, 6, 8} - produces the thread-count scaling curve (parallel analog of s1’s token-budget curve)
Seed strategy: hand-crafted vs. LM-generated vs. DPP-selected
Summary format: free-form vs. structured (sub-question / evidence / confidence)
Summary length budget: ablate token budget per thread summary
Collapse detector: TF-IDF similarity vs. embedding similarity
Gap-driven relevance: with vs. without sub-question tracking in DPP kernel

8. Novelty Summary
To our knowledge, no prior work has:
Formally defined intra-thread and inter-thread collapse as distinct, measurable failure modes in agentic search
Used pass@k as a diagnostic to determine whether solution paths are reachable independently of an agent’s ability to find them
Attributed the verification gap (pass@k grows but self-choice stays flat) to generation diversity rather than selector weakness
Proposed gap-driven relevance scoring - weighting candidate queries by contribution to uncovered sub-questions rather than proximity to the original query
Produced a thread-count scaling curve as the parallel analog of s1’s sequential token-budget scaling curve
Demonstrated that synthesis across diverse threads sidesteps verification, making selection unnecessary

9. Related Work to Verify
need to audit test-time-only work (no training) on top of base LLMs published recently. Key papers to check:
Search-o1 - sequential search scaling, training-free baseline
General AgentBench (Feb 2026) - documents verification gap, motivates our diversity hypothesis
DVTS (Diverse Verifier Tree Search) - comparison baseline for Exp. C
Any recent spawn-join or parallel search work without RL training

LTI, Carnegie Mellon University  •  February 2026


Proof of Concept
WebWalkerQA 
1. What We're Building
A single clean experiment that answers: given a fixed compute budget, does parallel diverse threading (T³) outperform sequential token scaling (s1)?
Dataset: WebWalkerQA, 200 questions, fixed. Metric: Exact Match. No training, inference only.
2. Compute Budget
Total budget = k × n × t tokens = k × n search calls.
k = number of threads per turn
n = number of turns (fixed at 6 for all runs)
t = tokens per thread per turn
s1 per turn
1 thread. Gets k×t tokens per turn as one long reasoning chain with k search calls embedded. Causal reasoning preserved - it sees everything sequentially.

T³ per turn
k threads in parallel. Each gets t tokens and 1 search call. Each outputs a summary of s tokens. Parent receives all k summaries (k×s tokens), decides to output <answer> or spawn another round.

Parent context at turn j = question + all summaries from rounds 1…j = grows by k×s tokens each round.

3. Experiment Matrix
Compare within groups A, B, C. Oracle is the ceiling - not compute matched, just shows how much headroom exists.
ID
Method
k
n
t (tokens/thread/turn)
Total tokens
Total search calls
A1
s1
1
6
1024
6,144
6
A2
T³
2
6
512
6,144
12
B1
s1
1
6
4096
24,576
24
B2
T³
4
6
1024
24,576
24
B3
T³
8
6
512
24,576
48
C1
s1
1
6
8192
49,152
48
C2
T³
8
6
1024
49,152
48
C3
T³
16
6
512
49,152
96
Oracle
pass@8 (uncapped)
8
6
1024
49,152×8
uncapped

4. T³ Loop
Each turn follows this structure:

Parent LM receives: question + all previous round summaries
Spawns k threads, each with a different seed prompt
Each thread: Thought → Search[query] → Observe → produces summary (s tokens)
Parent receives k summaries
Parent outputs <answer> if confident, else proceeds to turn k+1
At turn n=6 budget exhausted: parent forced to output best answer

The parent context window grows by k×s tokens each turn. With k=4, s=512, n=6: total parent context = 4×512×6 = 12,288 tokens. Well within 32k context.
5. What to Measure
Metric
Definition
Why
Exact Match (EM)
Normalized string match against ground truth
Primary accuracy metric
Pass@k (oracle)
EM when best of k threads is oracle-selected
Upper bound, shows headroom
Context richness
Total tokens in parent context at each turn
Shows T³ preserves info density
Search calls used
Actual calls made before <answer>
Efficiency metric
Turns used
How many rounds before termination
Did model use full budget?

6. Task Split

Ethan
Unified API wrapper: OpenAI + Gemini, swappable model interface
Evaluation harness: exact match with answer normalization (lowercase, strip articles)
Logging: per-question JSON — turns used, search calls, tokens used, final answer, EM
Run A1 (s1, k=1, t=1024): single thread, 6 turns, 1 search/turn
Run B1 (s1, k=1, t=4096): single thread, 6 turns, 4 searches/turn, 4096 tokens/turn
Run C1 (s1, k=1, t=8192): single thread, 6 turns, 8 searches/turn, 8192 tokens/turn
Run Oracle: 8 independent threads uncapped, oracle-select best answer per question


Sid
T³ core loop: spawn k threads, collect summaries, parent synthesis, <answer> detection
Diversity seed generation: one LLM call upfront per question, generates k distinct seed prompts
Thread summary format: structured - what I found / what I didn't find / best partial answer (s = t/2 tokens)
Run A2 (T³ k=2, t=512): 2 threads, 6 turns
Run B2 (T³ k=4, t=1024): 4 threads, 6 turns
Run B3 (T³ k=8, t=512): 8 threads, 6 turns
Run C2 (T³ k=8, t=1024): 8 threads, 6 turns
Run C3 (T³ k=16, t=512): 16 threads, 6 turns
Produce Figure 1: two side-by-side plots with pass@k curves and oracle ceiling


7. The Plot (Figure 1)
Two plots, side by side. Both share the same oracle ceiling.



Left plot — Sequential Scaling (s1)
Right plot — Parallel Scaling (T³)
X axis
k (1, 2, 4, 8, 16) — search calls per turn multiplier
k (1, 2, 4, 8, 16) — number of threads
Y axis
Exact Match on 200 WebWalkerQA questions
Exact Match on 200 WebWalkerQA questions
Lines
pass@1, pass@2, pass@4, pass@8, pass@16 + Oracle
pass@1, pass@2, pass@4, pass@8, pass@16 + Oracle
What it shows
How much sequential depth (more tokens/calls per turn) helps
How much parallel diversity (more threads) helps


Oracle (same line on both plots): best of 8 independent runs, oracle-selected. This is the ceiling both methods are trying to close.

Prediction: right plot rises faster than left. Sequential scaling plateaus early; parallel diversity keeps climbing. Oracle stays well above both - shows how much headroom remains.




