# MPP-on-gem5 Implementation Specification

**Target:** Port the Multiperspective Perceptron Predictor (Jiménez, CBP 2025) into gem5, as a new branch predictor class with full, correct speculative update of *every* history-tracking structure.

**Companion document:** `mpp_spec.md` — the algorithmic spec extracted from the paper. This document is the *gem5 implementation* layer. Wherever the paper is silent on a value, this spec marks it `TBD-from-source` and the value must be lifted from the released CBP 2025 source, not invented.

**Scope of "correct speculative update":** Unlike the CBP 2025 framework (which exposes a ground-truth speculative-update hook and allows unbounded shadow state), gem5 has only predicted directions at `lookup()` time and a `squash()`-based rollback path. This spec therefore departs from the paper in one specific, deliberate way:

> All history-tracking structures are speculatively updated with the **predicted** direction at `lookup()` time, recorded in a per-branch checkpoint, and **rolled back on `squash()`** to the exact state immediately before that `lookup()`. Predictor-table updates (weights, θ, combiner counters) remain **non-speculative** in this gem5 port — they are applied only at `update()` (commit). This avoids needing a second rollback path for table state and matches what gem5's existing TAGE-family predictors do. The paper's "speculative table/θ update" optimization (worth ≈ 0.033 MPKI per Figure 4) is therefore deliberately **not** ported in v1; see §11 for an optional v2 extension.

---

## 1. Where this lives in the gem5 tree

```
src/cpu/pred/
├── multiperspective_perceptron.{hh,cc}         # existing (CBP 2016) — leave alone
├── multiperspective_perceptron_tage.{hh,cc}    # existing — leave alone
├── multiperspective_perceptron_tage_64KB.{hh,cc}  # existing — leave alone
├── multiperspective_perceptron_tage_8KB.{hh,cc}   # existing — leave alone
├── multiperspective_perceptron_cbp2025.{hh,cc}    # NEW — MPP standalone (CBP 2025 features)
├── multiperspective_perceptron_tage_cbp2025.{hh,cc}  # NEW — combiner + plumbing
└── multiperspective_perceptron_tage_192KB.{hh,cc}    # NEW — concrete sized variant
```

`BranchPredictor.py` gains:

```python
class MultiperspectivePerceptronCBP2025(BranchPredictor):
    type = 'MultiperspectivePerceptronCBP2025'
    cxx_class = 'gem5::branch_prediction::MultiperspectivePerceptronCBP2025'
    cxx_header = "cpu/pred/multiperspective_perceptron_cbp2025.hh"
    # all sized parameters with defaults filled from source

class MultiperspectivePerceptronTAGECBP2025(LTAGE):
    type = 'MultiperspectivePerceptronTAGECBP2025'
    cxx_class = 'gem5::branch_prediction::MultiperspectivePerceptronTAGECBP2025'
    cxx_header = "cpu/pred/multiperspective_perceptron_tage_cbp2025.hh"
    speculativeHistUpdate = Param.Bool(True, ...)  # MUST default True
    # ... combiner params, MPP sub-component, statistical_corrector handle

class MultiperspectivePerceptronTAGE192KB(MultiperspectivePerceptronTAGECBP2025):
    # concrete 192 KB sizing — TAGE-SC-L 64 KB + MPP 128 KB
```

The existing `MultiperspectivePerceptronTAGE64KB` is **not** modified by this spec. The user's prior in-progress work on extending `speculativeHistUpdate` to the SC of `MultiperspectivePerceptronTAGE64KB` is a *separate* effort and is informed by §6 of this document — but the new CBP 2025 code is the cleaner place to land full SC speculative update.

---

## 2. High-level class structure

```cpp
namespace gem5::branch_prediction {

// ---------- MPP standalone (no TAGE) ----------
class MultiperspectivePerceptronCBP2025 : public BPredUnit
{
  public:
    struct BranchInfo;          // per-branch checkpoint, see §6
    struct HistorySpec;          // one feature descriptor

    // BPredUnit interface (gem5)
    bool lookup(ThreadID, Addr, void *&) override;
    void update(ThreadID, Addr, bool taken, void *bp_history,
                bool squashed, const StaticInstPtr &, Addr) override;
    void squash(ThreadID, void *bp_history) override;
    void uncondBranch(ThreadID, Addr, void *&) override;
    void btbUpdate(ThreadID, Addr, void *&) override;

  protected:
    // History structures — see §3
    GHist     ghist_;
    PathHist  path_;
    LocalHist local_;
    ModHist   modhist_;
    ModPath   modpath_;
    Imli      imli_;
    RecencyStack recency_;
    BlurryPathHist blurry_;
    AcyclicHist acyclic_outcomes_;
    AcyclicHist acyclic_addrs_;
    BackPathHist backpath_;

    // Weight tables: 33 tables of 6-bit saturating counters
    // Table 0: 2048 entries; tables 1..32: 4096 entries each
    std::vector<std::vector<int8_t>> weights_;     // size = 33

    // Adaptive threshold (Seznec O-GEHL style)
    int     theta_;
    int     tc_;     // threshold-control counter

    // Bloom filters for trivial-branch filtering (24 KB total)
    BloomFilter taken_bf_;
    BloomFilter ntaken_bf_;

    // Transfer-function lookup table (read-only)
    std::array<int, 128> xfer_;   // index by signed 6-bit weight

    // Feature descriptors, loaded once from constructor
    std::vector<HistorySpec> specs_;
};

// ---------- Combiner: TAGE-SC-L + MPP ----------
class MultiperspectivePerceptronTAGECBP2025 : public LTAGE
{
    // Owns (composition, not inheritance):
    MultiperspectivePerceptronCBP2025 mpp_;

    // Combiner state
    std::array<Slope, 24>    slopes_;
    std::array<Bias,  24>    biases_;
    std::array<std::array<uint8_t, 64>, 24> miss_counters_; // 3-bit, packed
    std::array<int8_t, 24>   selected_bias_;                // index into 64

    // gem5-side: index of the SC sub-predictor inside LTAGE for confidence read
    StatisticalCorrector *sc_;
    TAGE_SC_L_TAGE       *tage_;

    // Per-branch checkpoint augments LTAGE's; see §6
    struct CombinerBranchInfo;
};

} // namespace
```

---

## 3. History-tracking structures (every one of these is checkpointed)

Each structure exposes the same five-method contract:

```cpp
struct HistoryStructure {
    void update(Addr pc, bool predicted_taken, BranchClass cls);
    Snapshot snapshot() const;             // O(1) or O(structure-size)
    void restore(const Snapshot &);        // exact undo
    bool equals(const HistoryStructure &) const; // for asserts
    /* indexing helpers used by features */
};
```

The **snapshot/restore** pair is the heart of correct speculative update. There are two acceptable implementations:

- **A. Full copy.** Simple, robust, cheap when the structure is small (≤ a few hundred bytes). Use this for: IMLI counter, recency stack, blurry-path, acyclic, backpath, modhist/modpath registers.
- **B. Undo-log.** For larger structures (GHIST, full PATH), store only the bits/words that were modified by the update so they can be reverted. Use this only if `BranchInfo` size becomes a problem.

For v1, **use full copy everywhere except GHIST and PATH**. GHIST/PATH are large (multi-kilobit) but their per-branch update is a single shift+XOR; record the **single bit shifted out** and the **single bit shifted in** plus the previous head pointer, and revert by shifting the other way. See §6.2.

### 3.1 GHist (Global history)
- Storage: large bit register; the paper's item-(5) miscellaneous totals 3,323 bits across all global/path/modulo histories, exact length `TBD-from-source`.
- Update on `lookup(pc, predicted)`: shift `predicted` into LSB.
- Snapshot: `{head_idx, evicted_bit}` — 2 bytes.
- Restore: shift right by 1 and re-insert `evicted_bit` at MSB.
- **Also folded in:** unconditional branches, calls, and returns contribute address bits to the history (paper §4 "Extra information"). Each such instruction must call `update()` with the synthesized "direction" (which is just whatever bits the source folds in — `TBD-from-source`).

### 3.2 PathHist
- Storage: circular array of 16-bit truncated PCs.
- Update: write `pc & 0xFFFF` at `head`, `head = (head+1) % depth`.
- Snapshot: `{head_idx, prev_value_at_head}` — 4 bytes.
- Restore: write `prev_value_at_head` back, decrement head.

### 3.3 LocalHist
- Storage: 1,280 entries × 34-bit shift registers (per Cost item 4).
- Indexed by hash of branch PC.
- Update: shift predicted into LSB of `local_[h(pc)]`.
- Snapshot: `{index, evicted_bit}`. Restore: shift right + re-insert.
- **Multiple in-flight branches may map to the same local entry.** Snapshots are stacked; restore in LIFO order on squash. The `BranchInfo` records the *index* it modified, not a pointer.

### 3.4 ModHist (modulo history)
- Per (modulus, length) instance. Update writes only when `pc % modulus == 0`.
- Snapshot must record whether the update fired, and if so the evicted bit.
- Restore is conditional on the recorded fire-flag.

### 3.5 ModPath (modulo path)
- As ModHist, but stores 16-bit PC slices. Same fire-flag pattern.

### 3.6 IMLI (forward formulation)
- Storage: a single integer counter (per the source — multiple parameter-sets may exist; `TBD-from-source`).
- Update on a forward branch: increment if not-taken, reset to 0 if taken.
- **Critical for rollback:** snapshot is the *previous counter value* (single int). Cheap.
- Note: paper says only the *forward* IMLI is used in MPP; the *backward* IMLI is provided by TAGE-SC-L's existing IMLI feature.

### 3.7 RecencyStack
- Storage: fixed-depth stack of recent branch addresses, LRU-managed.
- Update on `lookup(pc, _)`: search for `pc`; if found, move to top; if not, push, evicting bottom.
- Snapshot: `{operation, evicted_addr_if_any, prev_position_if_found}`. ~12 bytes.
- Restore: invert the operation. (Move-to-top with prior position; or pop top and restore evicted at bottom.)

### 3.8 BlurryPathHist
- Storage: shift register of region numbers.
- Update fires only when `(pc >> region_shift) != current_region`. Conditional fire-flag pattern.

### 3.9 AcyclicHist (two instances: outcomes and hashed addresses)
- Storage: `H[n]`. Update: `H[pc % n] = predicted` (or hashed pc).
- Snapshot: `{slot_idx, prev_value}`. Restore: write back.

### 3.10 BackPathHist
- Like PathHist but only updated on backward branches. Conditional fire-flag.

### 3.11 BIAS feature
- No state — feature value is constant 0 XORed with PC. Nothing to snapshot.

### 3.12 GHISTPATH, GHISTMODPATH
- Derived features — they read from the *already-updated* GHIST+PATH (or MODHIST+MODPATH) at lookup time. **No independent state**, so nothing to snapshot beyond what their constituents already snapshot.

---

## 4. Feature descriptor — `HistorySpec`

Lifted directly from the CBP 2025 source. The spec array is read-only and known at construction time.

```cpp
enum class HistoryType : uint8_t {
    GHIST, PATH, LOCAL, GHISTPATH, BIAS,
    IMLI, MODHIST, MODPATH, GHISTMODPATH,
    RECENCY, RECENCYPOS, BLURRYPATH,
    ACYCLIC, ACYCLIC_HASH, BACKPATH, TAGE
};

struct HistorySpec {
    HistoryType type;
    int p1, p2, p3, p4;     // per-type parameters (depth, shift, modulus, etc.)
    int table_index;        // which of the 33 weight tables this feature reads
    uint32_t update_mask;   // bit-mask: which structures this feature updates
                            //          even on trivially-filtered branches
    // Hash function pointer, selected by `type`
};
```

**`update_mask`** corresponds to the paper's "Most of the MPP histories are updated even for trivial branches based on a mask tuned per feature" (§6 of paper). Every feature lookup that is trivially-filtered by Bloom still calls `update()` on history structures whose mask bit is set. Snapshots are still taken so squash can roll back.

The full list of `HistorySpec` entries — including all numeric parameters and the per-feature `update_mask` values — is `TBD-from-source` and **must be ported verbatim** from the released CBP 2025 source's `history_spec[]` array.

---

## 5. Prediction & update flow in gem5

### 5.1 `lookup(tid, pc, &bp_history)` — every fetch

```
1.  Allocate BranchInfo *bi = new BranchInfo(pc);
    bp_history = bi;            // gem5 will hand this back at squash/update.

2.  bi->trivial = bloom_check(pc);
    if (bi->trivial != NONTRIVIAL):
        // Trivial: predicted direction is determined.
        bi->predicted = (bi->trivial == TAKEN);
        // Still update every history structure whose update_mask bit is set,
        // taking snapshots (see §3). NEVER skip the snapshot just because
        // the prediction is "trivial".
        snapshot_and_update_all(bi, bi->predicted, /*trivial=*/true);
        return bi->predicted;

3.  // Nontrivial: compute yout.
    int yout = 0;
    for (HistorySpec &s : specs_):
        uint32_t idx = hash_for(s, pc);     // depends on type
        bi->indices[s.table_index] = idx;   // record so update() can find them
        int8_t   w   = weights_[s.table_index][idx];
        yout += xfer_[w + 64];              // transfer-function LUT

4.  bi->yout = yout;
    bi->predicted = (yout >= 0);

5.  // Speculatively update every history structure with bi->predicted,
    //   recording snapshots in bi.
    snapshot_and_update_all(bi, bi->predicted, /*trivial=*/false);

6.  return bi->predicted;
```

### 5.2 `squash(tid, bp_history)` — fetch-stage squash *only*

When gem5 squashes a not-yet-resolved branch (control mispredict, exception, etc.) before it commits, the history structures must be rewound *exactly*.

```
void squash(tid, void *bp_history):
    BranchInfo *bi = static_cast<BranchInfo *>(bp_history);
    // LIFO restore: structures are restored in reverse of the order they
    // were snapshotted in lookup(). This is the safe order even though
    // the structures are independent, because some structures (LOCAL) can
    // be the same one for different branches.
    restore_all_in_reverse(bi);
    delete bi;
```

**Critical invariants enforced by `restore_all_in_reverse`:**
1. After restore, every history structure has *byte-for-byte* the value it had immediately before `lookup()` for this branch.
2. No predictor table (weights_, θ, combiner counters, Bloom filters, miss_counters) is touched. Tables only mutate in `update()`.
3. Restores are LIFO across in-flight branches. gem5 calls `squash()` for the youngest in-flight branch first; per-branch restore is independent of order *between* branches as long as each branch's snapshot describes only the change it made.
4. `bi` is freed.

### 5.3 `update(tid, pc, taken, bp_history, squashed, ...)` — at commit

There are two paths gem5 may invoke:

**Path A: `squashed == true`** (gem5 wants you to also undo). gem5 already called `squash()` first in some configurations; in others `update(squashed=true)` *replaces* squash. Mirror what `multiperspective_perceptron_tage.cc` does today: treat `squashed=true` exactly like `squash()` — restore histories, delete `bi`, return.

**Path B: `squashed == false`** (committed branch).

```
1.  // History was already speculatively updated correctly:
    //   - if predicted == taken, no fix needed.
    //   - if predicted != taken, gem5 ALSO called squash() for the wrong-path
    //     fetches behind this branch, AND after squash gem5 re-fetches with
    //     correct direction, which causes a fresh lookup() to push the
    //     correct bit. Either way, history is now correct.
    //   So we do NOT touch history here.

2.  // Update Bloom filters with the resolved direction.
    if (taken)  taken_bf_.insert(pc);
    else        ntaken_bf_.insert(pc);

3.  // Trivial-filtered branches do not train weights.
    if (bi->trivial != NONTRIVIAL) goto cleanup;

4.  // Train weights only on misprediction OR low-confidence correct prediction.
    bool correct = (bi->predicted == taken);
    if (!correct || std::abs(bi->yout) < theta_):
        for (HistorySpec &s : specs_):
            int8_t &w = weights_[s.table_index][bi->indices[s.table_index]];
            if (taken) w = sat_inc(w, +31);   // 6-bit signed: range [-32, +31]
            else       w = sat_dec(w, -32);

    // Adaptive threshold (O-GEHL style; ports from existing
    // multiperspective_perceptron.cc::adjustThreshold).
    if (!correct) { tc_++; if (tc_ >= TC_MAX) { theta_++; tc_ = 0; } }
    else if (std::abs(bi->yout) < theta_) {
        tc_--; if (tc_ <= -TC_MAX) { theta_--; tc_ = 0; }
    }

cleanup:
    delete bi;
```

### 5.4 `uncondBranch(tid, pc, &bp_history)` — calls, jumps, returns
- Allocate a `BranchInfo`, **always predicted taken**.
- Take history snapshots for every structure that includes "extra information" from non-conditional control flow (paper §4).
- Speculatively update those structures with the synthesized bits used by the source's hash routines (`TBD-from-source`).
- Record `bi->is_uncond = true`.
- `update()` for unconditional branches must skip weight training.
- `squash()` for unconditional branches must still restore.

### 5.5 `btbUpdate(tid, pc, &bp_history)` — BTB miss, treat as unknown direction
- Existing gem5 convention: do *nothing* destructive; just leave `bp_history` consistent. The MPP standalone has no BTB-coupled state.

---

## 6. `BranchInfo` checkpoint layout

The struct that gem5 stores in `bp_history`. It must contain everything needed for **bit-perfect rollback** of every history structure plus everything the eventual `update()` needs.

```cpp
struct MultiperspectivePerceptronCBP2025::BranchInfo
{
    // -------- Identity --------
    Addr pc;
    bool predicted;
    bool is_uncond;
    enum Trivial { NONTRIVIAL, TRIV_TAKEN, TRIV_NTAKEN } trivial;

    // -------- Per-feature data needed by update() --------
    int yout;                          // sum after transfer function
    std::array<uint32_t, 33> indices;  // one index per weight table

    // -------- Snapshots (one per history structure) --------
    // Sized for full-copy or undo-log per §3.
    GHist::Snapshot           ghist_snap;       //  ~4 bytes (head + evicted bit)
    PathHist::Snapshot        path_snap;        //  ~4 bytes
    LocalHist::Snapshot       local_snap;       //  ~6 bytes (idx + evicted bit + tid)
    std::vector<ModHist::Snapshot>  modhist_snap;   // one per (modulus,len) instance
    std::vector<ModPath::Snapshot>  modpath_snap;
    Imli::Snapshot            imli_snap;        //  ~8 bytes (full int copy)
    RecencyStack::Snapshot    recency_snap;     //  ~12 bytes
    BlurryPathHist::Snapshot  blurry_snap;
    AcyclicHist::Snapshot     acy_out_snap;
    AcyclicHist::Snapshot     acy_addr_snap;
    BackPathHist::Snapshot    backpath_snap;

    // -------- Bloom-filter "previous bits" --------
    // Bloom filters are NOT speculatively updated in v1 — they're updated only
    // at commit (update()). So no snapshot needed for them.
};
```

The combiner adds one outer `CombinerBranchInfo` that wraps both LTAGE's `BranchInfo*` and the MPP `BranchInfo*`, plus the combiner-specific intermediate (TAGE prediction, TAGE confidence, TAGE-only prediction, MPP prediction, computed combiner sum, selected-bias index used). On squash, the wrapper forwards squash to each sub-predictor.

### 6.1 Memory accounting

For a typical out-of-order pipeline with ~200 in-flight branches, full-copy snapshots cost roughly:
```
   200 branches × ~150 bytes/BranchInfo ≈ 30 KB
```
This is *gem5 simulator memory*, not the hardware budget. The 192 KB hardware budget is unaffected (in real hardware these would be undo-log entries or a checkpointed register file; the paper's item-8 850-bit single speculative entry is the hardware-budget representative, and gem5 already does not count simulator-side bookkeeping against the budget).

### 6.2 Why snapshots use undo-info, not full copies, for GHIST/PATH

GHIST is potentially thousands of bits. Per-branch full-copy is wasteful. The shift operation is invertible:
```
shift_in(b):       evicted = bits[MSB]; bits = (bits << 1) | b
shift_out(saved):  bits = (bits >> 1) | (saved << MSB)
```
So `Snapshot` is `{evicted_bit}` (1 bit, padded to 1 byte). Restore reverses the shift. Same logic for PathHist, LocalHist, BlurryPath, BackPath.

For ModHist/ModPath, snapshots also carry a 1-bit `fired` flag because the update is conditional.

---

## 7. The combiner (`MultiperspectivePerceptronTAGECBP2025`)

### 7.1 Reusing existing gem5 TAGE-SC-L

gem5 ships `LTAGE` and `TAGE_SC_L_*`. The combiner inherits from `LTAGE` (or composes `TAGE_SC_L_64KB` — composition is preferred to keep MPP-specific changes out of the TAGE class hierarchy).

What the combiner needs from TAGE-SC-L at `lookup()` time:
- TAGE-only prediction bit
- TAGE-SC-L prediction bit
- TAGE confidence ∈ {0, 1, 2}
- SC perceptron sum (for the linear combination of confidences)

If `TAGE_SC_L_TAGE` does not already expose these, add public accessors that return them given the TAGE `BranchInfo*`. **Do not** widen the `BranchInfo` of the existing TAGE-SC-L classes; instead make accessors that compute or read existing fields. (This is also where the user's prior in-progress work intersects: the `speculativeHistUpdate=True` flag must be honored by the SC for combiner-correctness; see §10.)

### 7.2 Combiner `lookup()`

```
1.  TAGE-SC-L lookup → (tagescl_pred, tage_pred, tage_conf, sc_sum, tage_bi)
2.  MPP lookup       → (mpp_pred, yout, mpp_bi)
3.  cat = encode(tagescl_pred, mpp_pred, tage_pred, tage_conf)   // 0..23
4.  combined = slopes_[cat] * sc_sum + (1 - slopes_[cat]) * yout + biases_[cat]
              + selected_bias_value_for(cat)
5.  prediction = (combined >= 0)
6.  bi = new CombinerBranchInfo{ tage_bi, mpp_bi, cat, combined,
                                  current_selected_bias_index_[cat] }
7.  return prediction
```

### 7.3 Combiner `update()`
- Forward to MPP and TAGE-SC-L update with their respective sub-`BranchInfo*`.
- Update miss_counters_ as the paper describes:
  - For each of the 64 candidate bias values, simulate "what would the prediction have been with this bias?" by adding `(bias_value - 32)` to `combined - selected_bias`.
  - For each candidate that would have *missed* on this branch, increment `miss_counters_[cat][bias_value]` (3-bit saturating at 7).
  - When any counter saturates at 7, **decay all 64 in that category** by 1.
  - Periodically (or on each update) recompute `selected_bias_[cat] = argmin(miss_counters_[cat])`. The paper notes this is a linear scan; gem5 should mirror that for fidelity to MPKI numbers.

### 7.4 Combiner `squash()`
- Forward to both sub-predictors' `squash()`. **The combiner itself has no speculative state**: slopes/biases/miss_counters are not updated speculatively, so there is nothing combiner-side to roll back.

---

## 8. Bloom filters

- 2 filters (`taken_bf_`, `ntaken_bf_`), each with 3 hash functions and 32,768-bit tables → 24 KB total.
- Hash family: per-feature `TBD-from-source`.
- Update only at `update()` (commit), with the resolved direction.
- **Not speculatively updated** in v1 — this is conservative and avoids needing to checkpoint Bloom-filter bits per branch. The paper does not specify whether the reference implementation speculates on these; the safe gem5 default is non-speculative.
- First-time-encountered branch (neither bit set in either filter): direction is `(last 5 ghist bits all true) ? taken : not-taken`.
- "Trivial" classification used at lookup:
  - Set in `taken_bf_` only → `TRIV_TAKEN`
  - Set in `ntaken_bf_` only → `TRIV_NTAKEN`
  - Set in both → `NONTRIVIAL` (run full MPP)
  - Set in neither → first-time rule above; treat as nontrivial for training purposes? `TBD-from-source` — match the reference.

---

## 9. Sizing and parameters (`BranchPredictor.py`)

For the 192 KB variant:

| Parameter | Default | Source |
|---|---|---|
| `numWeightTables` | 33 | paper §10 |
| `firstTableEntries` | 2048 | paper §10 |
| `restTableEntries` | 4096 | paper §10 |
| `weightBits` | 6 | paper §3 |
| `numLocalHistories` | 1280 | paper §10 |
| `localHistoryLength` | 34 | paper §10 |
| `bloomFilterBits` | 32768 | paper §10 |
| `bloomFilterHashes` | 3 | paper §10 |
| `combinerCategories` | 24 | paper §7 |
| `biasValuesPerCategory` | 64 | paper §7 |
| `missCounterBits` | 3 | paper §7 |
| `transferFunctionTable` | (file) | from CBP 2025 source |
| `historySpecs` | (file) | from CBP 2025 source |
| `speculativeHistUpdate` | `True` | always-on for MPP; see §10 |

The `historySpecs` and `transferFunctionTable` are loaded from a header file generated from the released source — do **not** retype from the paper.

---

## 10. Interaction with the user's current in-progress gem5 work

> Memory context: the user is mid-debugging a segfault in `makeBranchInfo` during `operator new`, with the leading hypothesis being virtual dispatch ordering or a missing `speculativeHistUpdate = True` flag in `BranchPredictor.py`, and the goal is extending speculative history update to the SC component of `MultiperspectivePerceptronTAGE64KB`.

The CBP 2025 combiner described here **requires** the SC to support speculative history update for correctness. Specifically:

1. The combiner reads `sc_sum` at `lookup()` time. If the SC's history is non-speculative, every in-flight branch reads `sc_sum` from a stale history, which destroys the combiner's MPKI.
2. The combiner forwards `squash()` to `LTAGE`, which must in turn rewind SC history.
3. Therefore: the SC speculative-update extension is a **prerequisite** for v1 of this spec, not a follow-up.

Concrete items the SC extension must satisfy to be combinable with this MPP:

- `StatisticalCorrector::lookup()` snapshots its histories into the TAGE `BranchInfo` (or a `BranchInfo` it owns).
- `StatisticalCorrector::squash(BranchInfo*)` exists and restores those histories.
- `BranchPredictor.py` for the combiner sets `speculativeHistUpdate = True`, which propagates into the SC sub-component.
- The `makeBranchInfo` segfault: most likely cause given the symptom is that the SC's `BranchInfo`-derived class is missing a virtual destructor (or `makeBranchInfo` is being called on a partially-constructed base before the derived vtable is in place — i.e., from a base-class constructor). Verify:
  ```cpp
  class StatisticalCorrector::BranchInfo {
      virtual ~BranchInfo() = default;     // MUST exist
  };
  ```
  And verify `makeBranchInfo` is called from `lookup()`, not from any constructor.

This spec assumes those fixes land first. The combiner code in §7 *will not work correctly* against an SC whose histories are non-speculative.

---

## 11. Optional v2: speculative table/θ update

The paper's "speculative table/θ update" optimization (Figure 4: ≈ 0.033 MPKI when removed) speculatively trains weights and θ using the *combined prediction* itself, monitors low-confidence in-flight branches, and falls back to non-speculative mode when too many are outstanding.

Porting this to gem5 requires:
1. A second snapshot path in `BranchInfo` for the *weight deltas* applied at lookup time (typically: a list of `(table, index, prev_value)` triples, one per feature).
2. `squash()` must restore those weight values.
3. The "low-confidence in-flight" counter, decremented on commit and incremented on lookup of a low-confidence branch.
4. A mode bit gating whether `lookup()` applies speculative training.

This is **deferred to v2** because (a) it adds a non-trivial second rollback path, (b) the MPKI cost is small relative to v1's correctness/complexity tradeoff, and (c) gem5's existing TAGE family does not speculatively train, so the codebase has no precedent.

If pursued, the snapshot grows by approximately `(num_features × sizeof(triple)) ≈ 33 × 8 = 264 bytes` per `BranchInfo`.

---

## 12. Validation plan

1. **Snapshot/restore round-trip test.** For every history structure, generate 10⁶ random `(pc, predicted)` pairs, call `update()` then `restore()` from a fresh snapshot, and assert byte-equality with the pre-update state. Run as a unit test under `tests/` or as a gem5 standalone harness.
2. **Squash sequencing test.** Drive a synthetic stream of `lookup → lookup → squash(2nd) → update(1st)` and assert that, after the squash and 1st-branch commit, every history structure equals the value produced by `lookup → update` of just the first branch with no second branch ever seen.
3. **MPKI parity sweep.** Build the 192 KB variant; run gem5 against the SPEC traces (or whatever traces the gem5 BP regression suite uses); expect MPKI within a small δ of the CBP 2025 reported 3.39 (gem5 traces are not the CBP 2025 traces, so absolute parity is not the goal — relative MPKI vs. the existing `MultiperspectivePerceptronTAGE64KB` is).
4. **Optimization-removal sweep.** Match Figure 4 of the paper: remove each optimization in turn (always/never-taken filtering, transfer function, smart combining, tuned combiner constants — speculative table/θ update is not present in v1) and confirm MPKI deltas have the correct *sign* and rough magnitude. Absolute deltas will differ because traces differ.
5. **`speculativeHistUpdate=False` mode.** Add a build-time flag (or runtime param) that disables snapshotting and runs all history updates only at `update()`. Should compile and run; MPKI should *increase* (paper does not isolate this in Figure 4 for histories specifically, but it should be a clearly worse number). This mode is for debugging only.
6. **Stress test under high in-flight count.** Run with 256+ in-flight branches and walking-1s of squash patterns; assert no `BranchInfo` leaks (gem5's stat counters can be checked, or instrument with a counter).

---

## 13. Items the source must supply (do not invent)

This list is the *exhaustive* set of values this spec deliberately does not pin down. Each is `TBD-from-source`:

1. The full `history_spec[]` array: types, p1–p4 parameters, table indices, per-feature `update_mask`.
2. The transfer-function lookup table contents (the `3.2x/(1−x²/1600)` is approximate per paper).
3. The SC transfer-function lookup table contents (range −63..63).
4. Bloom-filter hash functions (3 per filter).
5. Local-history indexing hash function.
6. Number of (modulus, length) ModHist instances and their parameters.
7. Number of (modulus, length) ModPath instances and their parameters.
8. Recency-stack depth.
9. Blurry-path region shift, depth, and per-region shift.
10. Acyclic history sizes (both variants) and per-element shifts.
11. The 24 combiner slope/bias pairs.
12. Which of the 24 combiner categories were tuned individually vs. grouped.
13. The exact widths totaling the 3,323 bits of item-(5) histories in the cost table.
14. Exact GHIST register length and which sub-ranges each GHISTPATH/GHISTMODPATH instance hashes.
15. Initial conditions for θ, tc_, miss_counters, weights (typically zero, but `TBD`).
16. Whether the IMLI-forward feature has multiple instances.

Pin these by porting from the released CBP 2025 source. Encode them as `constexpr` arrays in a generated header (`mpp_cbp2025_params.hh`) so they're version-controlled, reviewable, and not retyped.
