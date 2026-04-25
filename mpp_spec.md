# Multiperspective Perceptron Predictor (MPP) — Specification

**Source:** Daniel A. Jiménez, "Multiperspective Perceptron Predictor," CBP 2025, June 21, 2025, Tokyo, Japan. Texas A&M University and Barcelona Supercomputing Center.

**Purpose of this document:** Reference specification extracted from the CBP 2025 paper. Intended as a working spec for implementation/extension work in gem5 (e.g., extending speculative history update support to the SC component of `MultiperspectivePerceptronTAGE64KB`). Content is restricted to what the paper states. Items not stated in the paper are marked as such; nothing is invented.

---

## 1. Overview

- MPP is a **hashed perceptron predictor** that uses several different kinds of control-flow history information to form hashes into tables, read out weights, sum them, and threshold the sum to make a prediction.
- The sum of weights is called `yout` (from the original perceptron paper).
- Weights are **6-bit saturating** integers, updated by increment/decrement based on taken/not-taken.
- MPP is **combined with TAGE-SC-L** (the winner of the previous CBP). The paper updates the idea originally presented as the "multiperspective perceptron predictor" with and without TAGE in CBP 2016.
- Reported aggregate results (CBP 2025 organizers' script):
  - Aggregate branch mispredictions per kiloinstruction: **3.3895**
  - Cycles on the wrong path per kiloinstruction: **144.699**
  - (Section 6 separately states: arithmetic mean MPKI per workload category = **3.39**, geometric mean speedup over baseline = **1.017**, aggregate cycles on wrong path = **144.82**.)
- Hardware budget: **192 KB** total (see §10 Cost Analysis).

---

## 2. Background (as stated in the paper)

- The original perceptron branch predictor used a table of perceptron weight vectors. A vector is selected by branch address; the dot product with the bipolar global history yields the prediction. Weights are trained by perceptron learning (increment when matching, decrement when mismatching), saturating at the bit-width's max/min. A bias weight is trained on the branch outcome alone.
- Improvements cited: ahead pipelining with path information; piecewise linear decision surfaces; hashing history/path features to reduce table count.
- The **hashed perceptron predictor** (similar to Loh & Jiménez's modulo path-history; O-GEHL is a specific instance):
  - Several tables, each indexed by a different hash of branch history.
  - Tables hold saturating confidence weights.
  - Selected weights are summed; prediction is taken if sum ≥ 0, not taken otherwise.
  - On mispredict or low-confidence correct prediction, the corresponding weights are incremented (taken) or decremented (not taken).
  - This breaks the one-to-one correspondence between weights and history bits.

---

## 3. Prediction Algorithm (high level)

1. The combiner checks whether the branch has nontrivial behavior and should be filtered (Bloom-filter-based, §6). Branches observed only-taken or only-not-taken are predicted in that direction.
2. Otherwise, MPP is used. For each feature with its parameters, the relevant histories are hashed together with the branch address, taken modulo the table size, to yield an index into that feature's table; the weight at that index is read.
3. A **transfer function** is applied to each weight (§5).
4. The transformed weights are **summed** to form `yout`.
5. If `yout ≥ 0`, predicted **taken**; otherwise **not taken**.

The combined MPP / TAGE-SC-L prediction is produced by the combiner in §7.

---

## 4. Features

Training threshold is `θ`, set using a simplified version of Seznec's training algorithm from O-GEHL.

### 4.1 Traditional Features

- **GHIST** — Global history. Branch outcomes shifted into a large register (0 = not taken, 1 = taken). Hashed by bitwise XOR of multi-bit blocks. Parameters: starting and ending indices of the history register to hash.
  - **Note from paper:** GHIST is **not used in the final configuration on its own**, only in combination with other histories (see GHISTPATH, GHISTMODPATH).
- **PATH** — Path history; sequence of recent branch addresses. Addresses are truncated to **16 bits** and shifted into an array. Parameters: depth, shift. Hashed by accumulating a hash value up to the given depth.
- **LOCAL** — Local history. First-level table of per-branch shift registers, selected by a hash of the branch address; then hashed as an index into a second-level table of perceptron weights.
- **GHISTPATH** — Combination of GHIST and PATH; the two are XORed together as they are computed.
- **BIAS** — Feature value is `0`. XORed with branch address to form an index. Captures branch's overall taken/not-taken tendency independent of history.

### 4.2 Novel Features

- **IMLI** — Innermost loop iteration counter, with a twist:
  - Original (Seznec): increment on taken backward branch, reset on not-taken.
  - Alternate **forward IMLI**: increment when a forward branch is **not taken**; reset when it **is taken**. Models loops where the exit test is at the top.
  - **Only the forward formulation is used in MPP** (because backward IMLI is already incorporated into TAGE-SC-L).
- **MODHIST** — Modulo history. Only branches whose addresses are congruent to 0 modulo some modulus are recorded; incongruent branches are filtered out. Parameters: modulus (small integer), history length. Designed to combat "branch misalignment" (where a branch sometimes appears in history and sometimes does not, shifting all later positions).
- **MODPATH** — Modulo path history. Same as MODHIST but uses branch addresses instead of outcomes. Same parameters; hash computed similarly to PATH.
- **GHISTMODPATH** — Combination of MODHIST and MODPATH, analogous to GHISTPATH.
- **RECENCY** — Fixed-depth recency stack of recently encountered branches, managed with **LRU replacement**. Hashes the addresses in the stack. Parameters: depth into the stack to hash; shift to apply to the accumulator after each hash step.
- **RECENCYPOS** — Position of a branch's address in the recency stack (or "not present"). Single parameter: depth into the stack to search. Hash value = position where the address was found, or maximum table index if not found.
- **BLURRYPATH** — Coarse-granularity path history. Regions = branch addresses right-shifted by a configured amount. When a branch from a *new* region is encountered, the previous region is shifted into the history. Parameters:
  1. Amount to shift (region granularity)
  2. Depth within the history to hash
  3. Shift applied to each region number while generating the hash
- **ACYCLIC** — History register `H` of length `n`. Outcome of a branch with address `PC` is recorded at `H[PC mod n]`. Two variants:
  - `H` is an array of branch outcomes.
  - `H` is an array of hashed addresses of the corresponding branches.
  - Parameters: size of `H`, amount by which to shift hashed elements of `H`.
- **BACKPATH** — Like PATH but only records backward branches.
- **TAGE** — Incorporates the TAGE prediction and confidence shifted left by tuned parameters. Lets MPP benefit from cases where TAGE is more accurate than MPP alone.

### 4.3 Discussion (paper's own caveat)

> None of the novel features are particularly good predictors by themselves. Together with each other and with the traditional features, they provide alternate perspectives on branch history and allow finding new correlations.

---

## 5. Transfer Function

- Motivation: weights and "probability that branch is taken" may have a non-linear relationship.
- Form: `a*x / (1 - b*x²)`. Specifically the function `3.2x / (1 − x²/1600)` worked well.
- Implemented as a **lookup table indexed by the weight**.
- The same optimization was applied to the perceptron predictor (SC) inside TAGE-SC-L. The SC lookup table ranges in value from **−63 to 63** and is found in the modified TAGE-SC-L code.
- "The function has been tweaked slightly to improve accuracy" — i.e., the deployed table is not exactly the closed form.

---

## 6. Training & Update

- **Adaptive threshold (`θ`):** Tuned so that the number of updates due to mispredictions is roughly equal to the number due to low-confidence predictions. Algorithm follows Seznec (O-GEHL).
- **When to update:** A nontrivial branch resolves → update if either (a) prediction was incorrect, or (b) prediction was correct but `|yout| < θ`.
- **How to update:** Each weight used to make the prediction is incremented (if taken) or decremented (if not taken), with **saturating arithmetic**.
- A learning rate is applied to the perceptron output for threshold setting.
- Bits from the addresses of other control-flow instructions (unconditional branches, calls, returns) are also folded into the branch history.

### 6.1 Speculative Update Behavior — IMPORTANT

The paper distinguishes two kinds of speculative update:

1. **Histories** (global, path, modulo, etc.) are speculatively updated using **ground-truth taken/not-taken** information, via a hook the CBP 2025 organizers provide. Rollback to non-speculative history is therefore never needed. Organizers also allow an unlimited number of speculative history update entries.
2. **Predictor tables and the training threshold `θ`** are speculatively updated as well, but using the **combined MPP/TAGE-SC-L prediction**, not the ground truth. Because predictions can be wrong, tables/`θ` may be updated with wrong information. The training process undoes the update when a speculatively-updated mispredicted branch is resolved.
3. The predictor monitors when there are too many low-confidence branches in flight, and at that point puts MPP into a **non-speculative update mode** until some low-confidence branches commit.

> **Implementation note for gem5 work:** This second form (speculative update of *tables and θ* using the combiner's own prediction) is the part that makes "extending speculative history update support to SC" non-trivial. The paper's accounting (see §8) says removing speculative table/θ update increases MPKI by 0.033.

---

## 7. The Combiner

MPP is combined with TAGE-SC-L using a tuned linear combination of the perceptron confidence and the SC confidence.

### 7.1 Slope and bias selection

- The **slope and bias** of the linear combination are tuned **per category**, where the category is the tuple:
  - TAGE-SC-L prediction ∈ {0, 1}
  - MPP prediction ∈ {0, 1}
  - TAGE-only prediction ∈ {0, 1}
  - TAGE confidence ∈ {0, 1, 2}
- That gives **2 × 2 × 2 × 3 = 24 categories**. Most were tuned individually; rare ones were tuned grouped together.

### 7.2 Per-category bias selection

- An additional bias is added to the sum, then thresholded to make a prediction.
- The added bias is chosen to minimize the recent number of misses, **trained per category**.
- For each of the 24 categories, the combiner tracks **64 candidate bias values**.
- For each (category, bias-value) pair, a counter records how many misses that bias would have produced. When any counter saturates at **7** (3-bit counter), all 64 counters in that category are decayed.
- Total: 24 × 64 × 3 bits = **4,608 bits**.

### 7.3 Trivial-branch filtering (Bloom filters)

- Two Bloom filters: one for taken, one for not-taken.
- MPP only **trains** when a branch appears in **both** Bloom filters.
- Otherwise the single observed behavior is used to predict.
- First-time-encountered branch: predicted taken iff the **last 5 ghist bits are all true**, otherwise not taken.
- Most MPP histories are still updated even for trivial branches, **based on a per-feature mask** (tuned).
- Bloom filters cost **24 KB** total. The paper notes this would be unnecessary if a BTB were available — a single "observed not taken" bit per BTB entry in an 8K-entry BTB would suffice. The CBP 2025 framework does not expose a BTB, hence the workaround.

---

## 8. Optimizations — Quantitative Value (Figure 4)

Each row removes only **one** optimization (others kept). When applicable, hardware freed by removal is reused for extra perceptron weights.

| Optimization removed | MPKI delta vs. all-on |
|---|---|
| Filtering always-taken / never-taken branches | **+0.141** |
| Non-linear transfer function (replaced by linear of same range) | **+0.059** |
| Smart combining algorithm (replaced with: pick MPP iff `\|yout\| > 0.5·θ`) | **+0.042** |
| Speculative table/θ update (only non-speculative updates retained) | **+0.033** |
| Tuned combiner constants (slope=1, bias=0, equal weights) | **+0.027** |

---

## 9. Feature-selection process (informational)

- Features were selected via a **genetic algorithm**, followed by a **hill-climbing** phase, evaluating "hundreds of thousands" of combinations on the CBP 2025 traces.
- Trace replay was done in a **bespoke branch-predictor-only simulator** (not the full CBP 2025 simulator) for speed; the final feature set was integrated back into the CBP 2025 simulator.
- The selected feature set is encoded in the source code as an **array of `history_spec` structs**.
- A key point: **feature selection was performed against the combined MPP + TAGE-SC-L predictor**, optimizing the combined MPKI rather than MPP-alone MPKI. As a consequence, MPP's selected features lean more heavily on modulo-history and other features somewhat orthogonal to what TAGE-SC-L already covers.

---

## 10. Cost Analysis (192 KB hardware budget)

The paper counts only **mutable state**. Read-only tables (e.g., the transfer function lookup) and feature definitions are considered part of the algorithm, not storage.

| # | Component | Bits |
|---|---|---|
| 1 | TAGE-SC-L (organizers' implementation, assumed) | 524,288 (64 KB) |
| 2 | 33 tables of 6-bit weights. Table 0: 2,048 entries. Tables 1–32: 4,096 entries each. Total entries = 133,120 | 798,720 |
| 3 | 2 Bloom filters, each 3 tables × 32,768 bits | 196,608 |
| 4 | 1,280 local history registers × 34 bits | 43,520 |
| 5 | Various global history, path history, modulo history, etc. (detailed in code) | 3,323 |
| 6 | 24 tables × 64 × 3-bit miss counters (combiner) | 4,608 |
| 7 | 32-bit integer added to TAGE-SC-L to communicate TAGE prediction + confidence to MPP | 32 |
| 8 | Speculative-update entry, single copy (organizers allow many but only one is budgeted; see breakdown below) | 850 |
| 9 | Miscellaneous counters (overestimate to avoid pedantry) | 900 |
|  | **Total** | **1,572,849 bits = 192 KB** |

### 10.1 Breakdown of the 850-bit speculative-update entry (item 8)

Combined fields from the combiner and from MPP:

| Sub-field | Width |
|---|---|
| (a) Branch PC for the combiner | 32 bits |
| (b) Two pointers to prediction objects (paper notes these would be smaller in real hardware) | 2 × 64 = 128 bits |
| (c) Combiner sum (double-precision float) | 64 bits |
| (d) Branch PC for MPP | 32 bits |
| (e) Two 16-bit hashes of the PC for MPP | 32 bits |
| (f) `yout` value (32-bit integer) | 32 bits |
| (g) 33 × 16-bit indices into perceptron weight tables | 528 bits |
| (h) "Prediction has been updated" flag | 1 bit |
| (i) "Overall predictor of the combiner" record | 1 bit |
|  | **Total: 850 bits** |

The paper also notes that local histories of TAGE-SC-L and MPP are redundant and **could be shared** to reduce storage, but were left separate.

---

## 11. Implementation & Hardware Discussion (paper's own caveats)

The paper is explicit that CBP does not score for design complexity, and acknowledges MPP would be hard but not impossible to build. Items called out as challenging:

- **Sequential critical path:** read weight → apply transfer function → sum → threshold. Mitigation: ahead pipelining (Jiménez 2002; Seznec & Fraboulet 2003).
- **Transfer-function lookup:** the only step the author does not know to be already implemented in industry. Suggested mitigation: move the lookup off the critical path by applying it during *update* and storing wider weights, so update advances the weight to the next lookup-table value rather than incrementing by one.
- **Multiple predictions per fetch block** require instantaneous speculative update of global history. Possible workaround: use bits from targets and/or branch addresses, updated only on taken branches, instead of taken/not-taken bits.
- **TAGE feature** introduces serial dependency on the TAGE predictor's output (mitigate via ahead pipelining).
- **Combiner's per-category bias selection** is currently implemented with frequent linear scan of the 24 × 64 miss-counter tables; the paper believes this could be replaced with simple counters.
- **Local history** is well-known to be hard in hardware; a complexity/accuracy compromise is unavoidable.
- **Multiple speculatively updated features** would require additional rollback complexity on misprediction recovery.

---

## 12. Items Reported but Not Specified Numerically in the Paper

The following are mentioned but not given exact numeric parameters in the paper text — values must be read from the source code:

- The exact list of `history_spec` structs (which features are enabled, with which parameters) — paper says "given in the source code".
- The exact contents of the transfer-function lookup table (the closed form `3.2x / (1 − x²/1600)` is approximate; the deployed table is "tweaked slightly").
- The exact contents of the SC transfer-function lookup table (range −63..63, in the modified TAGE-SC-L code).
- Per-feature mask for "update history even on trivial branches" (mentioned as tuned per feature, value not given).
- The 24 tuned slope/bias pairs for the combiner.
- The mapping of which combiner categories were tuned individually vs. grouped.
- The exact set of bits and precise widths in item (5) of the cost table (3,323 bits, "accounted in detail in the code").

For any work that depends on these (including the gem5 SC speculative-update extension), values must be taken from the released source, not from this document.

---

## 13. References (as cited in the paper)

1. Adiga et al., "The IBM z15 High Frequency Mainframe Branch Predictor Industrial Product," ISCA 2020.
2. Akkary et al., "Perceptron-Based Branch Confidence Estimation," HPCA 2004.
3. Albericio, San Miguel, Enright Jerger, Moshovos, "Wormhole: Wisely Predicting Multidimensional Branches," MICRO-47, 2014.
4. Bhatia, Chacon, Pugsley, Teran, Gratz, Jiménez, "Perceptron-Based Prefetch Filtering," ISCA 2019.
5. Grayson et al., "Evolution of the Samsung Exynos CPU Microarchitecture," ISCA 2020.
6. Jiménez, "Reconsidering Complex Branch Predictors," HPCA-9, 2002.
7. Jiménez, "Fast Path-Based Neural Branch Prediction," MICRO-36, 2003.
8. Jiménez, "Piecewise Linear Branch Prediction," ISCA-32, 2005.
9. Jiménez and Lin, "Dynamic Branch Prediction with Perceptrons," HPCA-7, 2001.
10. Loh and Jiménez, "Reducing the Power and Complexity of Path-Based Neural Branch Prediction," WCED 2005.
11. Seznec, "Genesis of the O-GEHL Branch Predictor," JILP 2005.
12. Seznec and Fraboulet, "Effective Ahead Pipelining of Instruction Block Address Generation," ISCA 2003.
13. Seznec, San Miguel, Albericio, "The Inner Most Loop Iteration Counter: A New Dimension in Branch History," MICRO-48, 2015.
14. Shah et al., "Sparc T4: A Dynamically Threaded Server-on-a-Chip," IEEE Micro 2012.
15. Tarjan and Skadron, "Merging Path and Gshare Indexing in Perceptron Branch Prediction," TACO 2005.
16. Teran, Wang, Jiménez, "Perceptron Learning for Reuse Prediction," MICRO 2016.
