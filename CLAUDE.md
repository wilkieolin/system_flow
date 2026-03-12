# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

SystemFlow is a Python framework for modeling scientific data processing systems as DAGs. It evaluates trade-offs in power, throughput, classification accuracy, and latency. Two active domain applications: HEP (particle physics triggers/readout) and XRS (x-ray scattering beamlines).

For detailed architecture and code examples, see [AGENTS.md](AGENTS.md).

## Setup & Environment

```bash
conda env create -f environment.yml
conda activate systemflow
pip install -e .
```

No test suite or linter is configured. Validation is done via notebooks.

## Architecture at a Glance

The core framework lives in `systemflow/node.py` with supporting types in `auxtypes.py` and `merges.py`:

```
System → ExecutionGraph → Component → Mutate → Message
                        → Link (with Transport)
                        → Metric
```

- **Message**: namedtuple with `fields` (sample-varying) and `properties` (metadata). Key naming convention: `"name (unit)"` (e.g., `"power (W)"`, `"sample data (B)"`).
- **Mutate**: Abstract transformation. Declares required inputs/outputs via `VarCollection`. Implement `transform()`.
- **Component**: DAG node hosting mutations. Has immutable `parameters` and mutable `properties` (set during execution).
- **Link/Transport**: DAG edges. Transport computes link-level effects (e.g., communication power).
- **ExecutionGraph**: DAG orchestrator. Calling `graph()` returns a **new** graph with results — the original is unchanged.
- **Metric**: Computes derived values from executed graph state.
- **System**: Orchestrates multiple ExecutionGraphs with feedback loops.

### HEP Domain (`systemflow/hep/`)

Builds on the core framework with CMS-specific mutations, metrics, and a spreadsheet-based graph loader:
- `hep_graph_from_spreadsheet()` builds a complete ExecutionGraph from an Excel config
- `hep_with_updated_parameters()` creates parameter variants with auto-recomputed detector ratios

### v1.0 Legacy (`systemflow/graph.py`)

Procedural API using flat networkx DiGraphs. Being replaced by the v2.0 framework above. Do not extend.

## Critical Pitfalls

1. **Link results go to `link.parameters`, not `link.properties`.** After execution, transport outputs are merged into `parameters`. Metrics reading link power must use `link.parameters["power (W)"]`.

2. **Use `hep_with_updated_parameters` for HEP graphs**, not `with_updated_parameters` directly. It auto-recomputes `"global ratio (1)"` on detectors when reduction ratios change.

3. **Classifier data paths are CWD-relative.** `L1TClassifier`/`HLTClassifier` load from `os.path.join(os.getcwd(), 'HEP', 'l1t_data')`. Notebooks must run from project root.

4. **Stochastic classifiers.** L1T/HLT use 50k samples internally — expect 1-5% variation between runs.

5. **Immutable execution model.** `graph()` and `with_updated_parameters()` return new objects. Never expect in-place mutation.

6. **Metric reconstruction.** `with_updated_parameters` recreates metrics via `metric.__class__()` (no-arg constructor). Metrics with constructor args (e.g., `ScaledPower(target_year=2032)`) will lose them.

7. **Single root node required.** ExecutionGraph assumes exactly one terminal node.

## Key Entry Points

- `examples/test_system.ipynb` — minimal tutorial for core framework
- `HEP/results_table_v2.ipynb` — comprehensive HEP analysis example
- `XRS/ptychography.ipynb` — System-level multi-graph example with feedback loops
- `docs/systemflow_v2_primitives.md` — detailed primitive documentation
