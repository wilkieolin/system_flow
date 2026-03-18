# Migration Plan: HEP Examples from v1.0 to v2.0

## Context

The HEP directory contains 9 notebooks that model the CMS detector data acquisition system. They all use the **v1.0 API** (`systemflow/graph.py`) which represents the system as a flat `networkx.DiGraph` with node/edge attributes stored as dictionaries. The goal is to convert these to use the **v2.0 API** (`systemflow/node.py`) which uses structured `Component`, `Mutate`, `ExecutionGraph`, `Metric`, and `Link` classes.

The XRS notebooks (BCDI, ptychography) already use the v2.0 API and serve as reference implementations.

---

## Architectural Differences: v1.0 vs v2.0

| Aspect | v1.0 (`graph.py`) | v2.0 (`node.py`) |
|--------|-------------------|-------------------|
| Graph structure | Flat `nx.DiGraph` with dict attributes on nodes/edges | `ExecutionGraph` with typed `Component` and `Link` objects |
| Node definition | Dictionary of properties passed to `add_nodes_from` | `Component` class with explicit `Mutate` list and `parameters` dict |
| Edge definition | Dictionary of properties on edges | `Link` class with `Transport` operation |
| Computation | Procedural functions (`message_size`, `propagate_statistics`, `update_throughput`) called in sequence on the graph | Declarative: mutations define transforms; graph execution handles propagation automatically |
| Classification | `Classifier` objects stored as node attributes, called by `propagate_statistics` | `GaussianClassify` mutation wrapping a `Classifier`, embedded in a component's mutation chain |
| Parameter variation | Manual: copy graph, modify attributes, re-run `update_throughput` | `with_updated_parameters()` creates a new graph cleanly |
| Metrics | Computed ad-hoc in notebooks (power, F1, productivity) | `Metric` subclasses attached to graph, auto-computed on execution |
| State | Mutable graph modified in place by functions | Functional: each execution returns a new graph instance |

---

## Phase 1: Build HEP-specific Mutations

The v1.0 code computes several quantities procedurally. These need to become `Mutate` subclasses. Based on analysis of what the HEP notebooks compute, we need:

### 1.1 `DetectorSample` mutation
**Replaces:** `detectors()` function in `graph.py`
**Purpose:** Originator node for detector data — sets the initial message with sample data, sample rate, compression, and routing latency.

```
Inputs:
  host_parameters: sample_data, sample_rate, compression, routing_latency, op_efficiency
Outputs:
  msg_fields: "sample data (B)", "routing latency (s)"
  msg_properties: "sample rate (Hz)", "data reduction (%)"
  host_properties: "energy (J)"
```

### 1.2 `ProcessorTransform` mutation
**Replaces:** Core logic in `message_size()` + complexity function application
**Purpose:** Applies a complexity function to incoming message size to compute operation count, and applies data compression.

```
Inputs:
  msg_fields: Regex(r"\(B\)") — all data fields
  host_parameters: complexity_fn, compression, op_efficiency, op_latency
Outputs:
  msg_fields: "processed data (B)"
  host_properties: "ops (n)", "energy (J)"
```

### 1.3 `TriggerClassify` mutation
**Replaces:** `propagate_statistics()` + classifier invocation
**Purpose:** Wraps L1T/HLT/Gaussian classifiers to produce contingency tables. Generalizes the existing `GaussianClassify` to support all classifier types.

```
Inputs:
  msg_properties: "sample rate (Hz)"
  host_parameters: classifier (Classifier object), reduction_ratio
Outputs:
  msg_fields: "contingency (2x2)"
  msg_properties: "error matrix (2p, 2p)", "output rate (Hz)"
  host_properties: "input rate (Hz)", "output rate (Hz)", "discards (n)"
```

This is a critical generalization: the existing `GaussianClassify` mutation in `mutations.py` only works with `GaussianClassifier`. The new mutation should accept any `Classifier` subclass (including `L1TClassifier` and `HLTClassifier`) as a component parameter.

### 1.4 `LinkPower` transport
**Replaces:** `link_power()` and `link_throughput()` in `graph.py`
**Purpose:** Computes data throughput and power consumption for each link based on message size, output rate, and link efficiency (J/bit).

```
Inputs:
  msg_fields: message size data
  link_parameters: link_efficiency (J/bit)
Outputs:
  link_properties: "throughput (B/s)", "power (W)", "energy (J)"
```

### 1.5 `LatencyAccumulate` mutation (or transport)
**Replaces:** `propagate_latency()` in `graph.py`
**Purpose:** Tracks processing and routing latency through the pipeline, including parallelism effects.

---

## Phase 2: Build HEP-specific Metrics

The HEP notebooks extract several derived quantities. These should become `Metric` subclasses:

### 2.1 `PipelineContingency` metric
**Replaces:** `pipeline_contingency()` in `graph.py`
**Purpose:** Aggregates the contingency tables from all classifier nodes in the pipeline into a single system-level confusion matrix.

### 2.2 `SystemPower` metric
**Replaces:** `op_power()` + `link_power()` in `graph.py`
**Purpose:** Sums operational and communication power across all components and links.

### 2.3 `Productivity` metric
**Replaces:** ad-hoc calculation in notebooks
**Purpose:** Computes `(F1 × output_rate) / total_power` — the key figure of merit used across HEP analyses.

### 2.4 `ScaledPower` metric
**Replaces:** ad-hoc `density_scale_model()` application
**Purpose:** Applies technology scaling model to project power to a target year.

---

## Phase 3: Build System Configuration from Spreadsheets

### 3.1 `hep_graph_from_spreadsheet()` utility
**Replaces:** `graph_from_spreadsheet()`, `construct_graph()`, `detectors()`, `processors()` in `graph.py`
**Purpose:** Reads the existing Excel spreadsheet format (Detectors/Processors/Global sheets) and constructs v2.0 `Component`, `Link`, and `ExecutionGraph` objects.

This function should:
1. Read detector rows → create `Component` instances with `DetectorSample` mutation
2. Read processor rows → create `Component` instances with `ProcessorTransform` + `TriggerClassify` mutations
3. Read edges from the "Output" column → create `Link` instances with `LinkPower` transport
4. Read global parameters → attach as graph-level metadata
5. Return an `ExecutionGraph` with appropriate metrics attached

This preserves backward compatibility with the existing `.xlsx` configuration files in `HEP/configurations/`.

---

## Phase 4: Convert Notebooks

With the infrastructure above in place, convert each notebook. The notebooks fall into three categories:

### Category A: Core System Construction (convert first)
These establish the baseline system and are prerequisites for the others.

**A1. `cms.ipynb`**
- Currently: Loads 3 spreadsheet configs, calls `graph_from_spreadsheet()`, manually extracts power/metrics
- Migration: Use `hep_graph_from_spreadsheet()` to build v2.0 graphs, attach `SystemPower` and `PipelineContingency` metrics, use `metric_values` for results
- Complexity: Medium — straightforward graph construction + manual power breakdown calculations

**A2. `results_table.ipynb`**
- Currently: Builds 9 configurations with manual parameter modifications, extracts metrics
- Migration: Build base graph, use `with_updated_parameters()` for each variant, read from `metric_values`
- Complexity: Medium — many configurations but repetitive pattern

### Category B: Parameter Sweeps (convert second)
These sweep parameters and collect metrics — the v2.0 `with_updated_parameters()` pattern maps directly.

**B1. `pileup_vs_l1t.ipynb`**
- Currently: 2D sweep over L1T reduction and pileup (101×101), modifies graph attributes in-place
- Migration: Use `with_updated_parameters()` in a nested loop; pileup variation requires interpolating detector data rates (needs a helper function or custom parameter update)
- Complexity: High — pileup interpolation between Run-3/Run-5 detector configs is non-trivial; need to ensure `vary_pileup()` logic translates to parameter maps
- Key challenge: The v1.0 code interpolates *between two spreadsheets* to create intermediate pileup levels; in v2.0, this could be a utility that generates interpolated parameter maps

**B2. `vary_hlt.ipynb`**
- Currently: 2D sweep over L1T reduction and L1T skill boost (101×101) with multiprocessing
- Migration: Use `with_updated_parameters()` + parallel map over parameter grid
- Complexity: Medium — similar to pileup_vs_l1t but simpler parameter variation
- Note: The `L1TClassifier(skill_boost=x)` parameter needs to be passable as a component parameter

**B3. `vary_system.ipynb`**
- Currently: Sweeps L1T accept rate while adjusting HLT inversely, across 4 hardware configs
- Migration: Use `with_updated_parameters()`, implement constraint (L1T × HLT = constant rejection)
- Complexity: Medium-High — coupled parameter variation

**B4. `reduction_vs_productivity.ipynb`**
- Currently: Sweeps Inner Tracker data reduction (101 values) across 3 configs
- Migration: Use `with_updated_parameters()` to modify detector compression
- Complexity: Low-Medium — straightforward 1D sweep

### Category C: Classifier Analysis (convert third or keep as-is)
These are primarily about characterizing classifiers, not the system graph.

**C1. `l1t_trigger_skill.ipynb`**
- Currently: Fits L1T trigger efficiency curves, generates sample distributions, runs order tests
- Migration: Mostly classifier-internal logic; the `L1TClassifier` class itself doesn't change. Only the notebook scaffolding that uses `graph.py` utilities needs updating.
- Complexity: Low — classifier code is shared between v1.0 and v2.0

**C2. `hlt_trigger_skill.ipynb`**
- Currently: Fits HLT trigger efficiency curves and generates distributions
- Migration: Same as L1T — classifier internals stay, notebook wrappers update
- Complexity: Low

### Category D: Gradient Analysis (convert last)

**D1. `system_gradients.ipynb`**
- Currently: Computes numerical partial derivatives of productivity w.r.t. 7 parameters
- Migration: Use `with_updated_parameters()` for perturbations, `Productivity` metric for the objective
- Complexity: Medium — needs a clean numerical differentiation wrapper around v2.0 graphs

---

## Phase 5: Deprecate v1.0 Code

Once all notebooks are converted and produce equivalent results:

1. Move `graph.py` to `graph_legacy.py` or a `legacy/` subdirectory
2. Update `__init__.py` to export v2.0 classes
3. Remove v1.0 imports from all notebooks
4. Update `README.md` to reference the new framework and notebook structure

---

## Implementation Order

```
Phase 1 (Mutations)          Phase 2 (Metrics)
  │                            │
  ├─ DetectorSample            ├─ PipelineContingency
  ├─ ProcessorTransform        ├─ SystemPower
  ├─ TriggerClassify           ├─ Productivity
  ├─ LinkPower (Transport)     └─ ScaledPower
  └─ LatencyAccumulate
          │
          v
Phase 3 (Spreadsheet loader)
          │
          v
Phase 4 (Notebook conversion)
  │
  ├─ A1: cms.ipynb
  ├─ A2: results_table.ipynb
  ├─ B1: pileup_vs_l1t.ipynb
  ├─ B2: vary_hlt.ipynb
  ├─ B3: vary_system.ipynb
  ├─ B4: reduction_vs_productivity.ipynb
  ├─ C1: l1t_trigger_skill.ipynb
  ├─ C2: hlt_trigger_skill.ipynb
  └─ D1: system_gradients.ipynb
          │
          v
Phase 5 (Deprecate v1.0)
```

---

## Validation Strategy

For each converted notebook, validate by comparing:
1. **Contingency tables** — exact match (integer counts should be identical given same random seed)
2. **Power values** — exact match (deterministic computation)
3. **F1 / precision / recall** — exact match (derived from contingency)
4. **Productivity** — exact match
5. **Figures** — visual comparison of plots (same trends, same scale)

Set random seeds where classifiers use stochastic sampling (`L1TClassifier`, `HLTClassifier`) to ensure reproducibility during validation.

---

## Key Design Decisions

### Should `L1TClassifier` / `HLTClassifier` become mutations themselves?
**Recommendation: No.** Keep classifiers as standalone objects passed as component parameters. Create a single `TriggerClassify` mutation that accepts *any* `Classifier` subclass. This preserves the existing classifier code and allows the same classifier to be used in different mutation contexts.

### Should pileup interpolation be built into the framework?
**Recommendation: No.** Keep it as a notebook-level utility function that generates parameter maps. The framework's `with_updated_parameters()` is sufficient — the interpolation logic is HEP-domain-specific and doesn't belong in the general framework.

### Should the spreadsheet loader be in the framework or a separate HEP utility?
**Recommendation: Separate HEP utility module** (e.g., `HEP/hep_utils.py`). The spreadsheet format is specific to the CMS system configuration and shouldn't be in the general `systemflow` package. The v2.0 framework is domain-agnostic; domain-specific construction helpers live alongside the domain notebooks.

### Where should new HEP mutations live?
**Recommendation: `HEP/hep_mutations.py`** — alongside the notebooks. These mutations (DetectorSample, ProcessorTransform, TriggerClassify) are HEP-specific and model CMS-specific physics. General-purpose mutations remain in `systemflow/mutations.py`.
