# AGENTS.md - Guide for AI Agents Working with SystemFlow

## What This Project Is

SystemFlow is a Python framework for modeling scientific data acquisition and processing systems as directed acyclic graphs (DAGs). It enables users to evaluate trade-offs in system design (power, throughput, classification accuracy, latency) by defining components, linking them, and sweeping over parameter spaces.

The framework has two active domain applications:
- **HEP (High-Energy Physics)**: Modeling CMS-like particle physics data acquisition systems with triggers, classifiers, and multi-stage data reduction
- **XRS (X-ray Science)**: Modeling ptychography and BCDI beamline systems with sample stages, detectors, and reconstruction algorithms

## Project Structure

```
systemflow/                    # Core framework
  node.py                      # Core primitives: Message, Mutate, Component, Link, Transport,
                               #   ExecutionGraph, Metric, System
  auxtypes.py                  # Message namedtuple, VarCollection, Regex, merge_message
  merges.py                    # Merge, OverwriteMerge
  mutations.py                 # Built-in mutations: CollectImage, Convolve, FourierTransform2D,
                               #   GaussianClassify, DataRate, StorageRate, StoreImage, etc.
  metrics.py                   # Built-in metrics: TotalOps, TotalLatency, precision, recall, f1_score
  classifier.py                # DummyClassifier, GaussianClassifier, L1TClassifier, HLTClassifier
  models.py                    # Technology scaling: density_scale_model, transistor_scale_model
  xrs.py                       # XRS-specific mutations: PositionSample, PhaseReconstruction2D/3D,
                               #   FlatFieldCorrection, MaskCorrection
  hep/                         # HEP domain package
    mutations.py               # HEPDetector, HEPProcessor, HEPMerge, HEPLinkTransport
    metrics.py                 # PipelineContingency, SystemPower, Productivity, ScaledPower
    utils.py                   # hep_graph_from_spreadsheet, hep_with_updated_parameters,
                               #   hep_construct_graph, dataframes_from_spreadsheet

HEP/                           # HEP analysis notebooks and data
  configurations/              # CMS system spreadsheets (cms_system_60/140/200.xlsx, smartpx variants)
  l1t_data/                    # L1T classifier empirical data (CSV files + trigger_rates.xlsx)
  hlt_data/                    # HLT classifier empirical data (CSV files)
  wall time scaling.xlsx       # CPU/GPU wall time scaling data

XRS/                            # X-ray science notebooks
examples/                       # Tutorial notebook (test_system.ipynb)
docs/                           # Architecture documentation
```

## Architecture: The Primitive Hierarchy

```
System
  +-- ExecutionGraph (one or more)
        +-- Component (nodes in the DAG)
        |     +-- Mutate (one or more, applied sequentially)
        +-- Link (edges connecting components)
        |     +-- Transport (applied during data transfer)
        +-- Metric (computed after execution)
```

### Message (auxtypes.py)

A `namedtuple` with two dicts: `fields` (sample-varying data) and `properties` (sample-independent metadata). Messages flow through the graph from leaf components to the root.

```python
from systemflow.auxtypes import Message
msg = Message(fields={"image data (B)": 8e6}, properties={"sample rate (Hz)": 1000})
```

Key naming convention: `"name (unit)"` - e.g. `"power (W)"`, `"resolution (n,n)"`, `"sample data (B)"`.

### Mutate (node.py)

Abstract class defining a single computation. Subclass it and implement `transform()`:

```python
from systemflow.node import Mutate, MutationInputs, MutationOutputs
from systemflow.auxtypes import VarCollection, Message

class MyMutation(Mutate):
    def __init__(self):
        msg_fields = VarCollection()           # required incoming message fields
        msg_properties = VarCollection()       # required incoming message properties
        host_parameters = VarCollection(       # required host component parameters
            sample_rate="sample rate (Hz)",
        )
        inputs = MutationInputs(msg_fields, msg_properties, host_parameters)

        out_fields = VarCollection(output="output data (B)")
        out_props = VarCollection()
        out_host = VarCollection(power="power (W)")
        outputs = MutationOutputs(out_fields, out_props, out_host)
        super().__init__("MyMutation", inputs, outputs)

    def transform(self, message: Message, component: 'Component'):
        rate = component.parameters[self.inputs.host_parameters.sample_rate]
        # ... compute ...
        return (
            {self.outputs.msg_fields.output: computed_data},    # new message fields
            {},                                                  # new message properties
            {self.outputs.host_properties.power: computed_power} # new host properties
        )
```

### Component (node.py)

A node in the graph hosting one or more mutations. Parameters are set at construction and stay constant during execution. Properties are set by mutations during execution.

```python
from systemflow.node import Component, collect_parameters

mutations = [MyMutation()]
vc = collect_parameters(mutations)  # VarCollection of all required parameters

node = Component(
    name="Sensor",
    mutations=mutations,
    parameters={vc.sample_rate: 40e6},  # must include all params required by mutations
    merge=OverwriteMerge(),             # optional, default is OverwriteMerge
)
```

### Link and Transport (node.py)

Links connect components (tx -> rx). A Transport computes link-level properties (throughput, power).

```python
from systemflow.node import Link, Transport, DefaultLink

# Simple passthrough link:
link = DefaultLink("Sensor -> CPU", "Sensor", "CPU")

# Link with transport computation:
link = Link("Sensor -> CPU", "Sensor", "CPU", MyTransport(), {"link efficiency (J/bit)": 1e-12})
```

**Important**: After execution, transport results are stored in `link.parameters` (not `link.properties`). This is because `Link.__call__` merges transport output into `self.properties | new_properties` and stores it as the new link's `parameters`.

### ExecutionGraph (node.py)

Orchestrates execution of the DAG. Calling it returns a new graph with computed results.

```python
from systemflow.node import ExecutionGraph

graph_def = ExecutionGraph("My System", nodes=[sensor, cpu], links=[link], metrics=[MyMetric()])

# Execute (returns NEW graph with results - original is unchanged)
result = graph_def()

# Access results
result.metric_values                          # dict of all metric outputs
result.root_node.output_msg                   # final message at the root
result.get_node("Sensor").properties          # properties set by mutations
result.get_all_node_properties()              # {name: properties_dict} for all nodes
```

### Parameter Sweeps with `with_updated_parameters`

Creates a new graph definition with modified parameters. The original is unchanged.

```python
# Single parameter change
variant = graph_def.with_updated_parameters({"CPU": {"filters (n)": 16}})
result = variant()

# Multiple parameters across multiple components
variant = graph_def.with_updated_parameters({
    "Sensor": {"sample rate (Hz)": 80e6},
    "CPU": {"filters (n)": 16, "skill (1)": 4.0},
})
```

### Metric (node.py)

Computes derived values from an executed graph. Override `__call__` or `metric()` to return a dict.

```python
from systemflow.node import Metric, Regex

class TotalPower(Metric):
    def __init__(self):
        super().__init__("Total Power", [], [Regex(r"power \(W\)")])

    def metric(self, message, properties):
        matches = self.graph_matches(properties)
        return {"total power (W)": sum(matches)}
```

### System (node.py)

Orchestrates multiple ExecutionGraphs with feedback loops. Override `step()` and `flow_control()`.

```python
from systemflow.node import System

class MySystem(System):
    def __init__(self, graphs):
        super().__init__("My System", graphs)

    def step(self):
        # Execute graphs in desired order, potentially feeding outputs between them
        pass

    def flow_control(self):
        # Iterate steps, update parameters between iterations
        return self.step()
```

### Merge (merges.py)

Defines how multiple input messages are combined at a component with multiple predecessors.

```python
from systemflow.merges import Merge

class MyMerge(Merge):
    def __init__(self):
        super().__init__(
            field_merges={"output data (B)": sum},
            property_merges={"sample rate (Hz)": min},
        )
```

## HEP Domain Package

The `systemflow.hep` package provides CMS-specific implementations built on the core primitives.

### Loading a System from Spreadsheet

```python
from systemflow.hep.utils import hep_graph_from_spreadsheet, hep_with_updated_parameters

# Build a graph definition from a CMS configuration spreadsheet
graph_def = hep_graph_from_spreadsheet("HEP/configurations/cms_system_200.xlsx", functions=funcs)

# Execute
result = graph_def()

# Access results
result.metric_values["total power (W)"]
result.metric_values["pipeline contingency (2x2)"]
result.metric_values["f1 score (%)"]
result.metric_values["productivity (ev/J)"]
```

### Creating Variants

Use `hep_with_updated_parameters` (not `with_updated_parameters` directly) when changing reduction ratios, because it auto-recomputes detector `global_ratio` values.

```python
from systemflow.hep.utils import hep_with_updated_parameters
from systemflow.classifier import L1TClassifier

# Change L1T classifier
variant = hep_with_updated_parameters(graph_def, {
    "Intermediate": {"classifier (obj)": L1TClassifier(skill_boost=0.40)}
})

# Change detector data (e.g., smart pixels reducing Inner Tracker data)
it_data = graph_def.get_node("Inner Tracker").parameters["sample data (B)"]
variant = hep_with_updated_parameters(graph_def, {
    "Inner Tracker": {"sample data (B)": it_data * 0.46}
})

# Change reduction ratios (triggers auto-recomputation of global_ratio)
variant = hep_with_updated_parameters(graph_def, {
    "Intermediate": {"reduction ratio (1)": 100},
    "Global": {"reduction ratio (1)": 53.3}
})
```

### HEP Parameter Keys

Detectors use: `"sample data (B)"`, `"sample rate (Hz)"`, `"data reduction (%)"`, `"op efficiency (J/op)"`, `"routing latency (s)"`, `"global ratio (1)"`, `"complexity (fn)"`

Processors use: `"sample data (B)"`, `"data reduction (%)"`, `"op efficiency (J/op)"`, `"routing latency (s)"`, `"reduction ratio (1)"`, `"classifier (obj)"`, `"complexity (fn)"`

Links use: `"link efficiency (J/bit)"`

### HEP Metrics Available After Execution

- `"pipeline contingency (2x2)"` - aggregated confusion matrix
- `"op power (W)"` - total component operational power
- `"link power (W)"` - total communication power
- `"total power (W)"` - op + link power
- `"precision (%)"`, `"recall (%)"`, `"f1 score (%)"`, `"accuracy (%)"`
- `"output rate (Hz)"` - events stored per second
- `"productivity (ev/J)"` - F1 * output_rate / total_power

### Complexity Functions

Custom complexity functions map message size to operation count. Pass them via the `functions` dict:

```python
from scipy.optimize import curve_fit

# Fit polynomial to empirical wall-time data
fit_poly = lambda x, k3, k2, k1: k3 * x**3 + k2 * x**2 + k1 * x
k, _ = curve_fit(fit_poly, scaling["Size"], scaling["Wall Time"])

funcs = {
    "Global": lambda x: fit_poly(x, *k),         # processor name -> complexity function
    "Intermediate": lambda x: x / 2.0e6,
}
graph_def = hep_graph_from_spreadsheet("HEP/configurations/cms_system_200.xlsx", functions=funcs)
```

## Common Patterns

### Parameter Sweep

```python
import numpy as np

reductions = np.linspace(10, 200, 50)
results = []
for r in reductions:
    variant = hep_with_updated_parameters(graph_def, {
        "Intermediate": {"reduction ratio (1)": r}
    })
    result = variant()
    results.append(result.metric_values)
```

### Extracting Classification Metrics

```python
from systemflow.classifier import get_passed, get_rejected
from systemflow.metrics import precision, recall, f1_score

contingency = result.metric_values["pipeline contingency (2x2)"]
p = precision(contingency)
r = recall(contingency)
f1 = f1_score(contingency)
output_events = get_passed(contingency)  # [false_positives, true_positives]
rejected = get_rejected(contingency)      # [true_negatives, false_negatives]
```

### Technology Scaling

```python
from systemflow.models import density_scale_model

# Project power to 2032 technology node
scaled_power = total_power / density_scale_model(2032)
```

### Multi-Step Systems (XRS Pattern)

For systems requiring iteration (e.g., scanning a sample across positions):

```python
# Execute, extract output, feed back as parameters for next iteration
result = graph_def()
new_params = match_message_to_parameters(result, vc_storage, "CPU host")
next_graph = result.with_updated_parameters(new_params | position_update)
result2 = next_graph()
```

### Using Classifiers for Sweeps with Multiprocessing

When sweeping over classifier `skill_boost` with multiprocessing, create independent classifier instances per worker:

```python
from copy import deepcopy

def vary_skill(base_def, skill):
    classifier = deepcopy(base_def.get_node("Intermediate").parameters["classifier (obj)"])
    classifier.skill_boost = skill
    variant = hep_with_updated_parameters(base_def, {
        "Intermediate": {"classifier (obj)": classifier}
    })
    return variant()
```

## Critical Pitfalls

1. **Link transport results are in `link.parameters`, not `link.properties`.**
   After `Link.__call__`, the new link stores transport output merged into its `parameters` dict. When writing metrics that read link power, use `link.parameters.get("power (W)", 0.0)`.

2. **`hep_with_updated_parameters` vs `with_updated_parameters`.**
   Use `hep_with_updated_parameters` for HEP graphs when changing `"reduction ratio (1)"` on any processor. It auto-recomputes `"global ratio (1)"` on all detectors. Using `with_updated_parameters` directly will leave global ratios stale.

3. **Classifier data path dependency.**
   `L1TClassifier` and `HLTClassifier` load data from `os.path.join(os.getcwd(), 'HEP', 'l1t_data')` and `'hlt_data'` respectively. The working directory must be the project root when instantiating these classifiers.

4. **Stochastic classifiers.**
   L1T and HLT classifiers use 50,000 stochastic samples internally. Results vary 1-5% between runs. For reproducibility, set `np.random.seed()` before instantiation, or use `n_samples` parameter.

5. **Functional/immutable execution.**
   `graph_def()` returns a NEW `ExecutionGraph` - the original `graph_def` is unchanged and can be reused. Similarly, `with_updated_parameters()` returns a new graph definition. Never expect in-place mutation.

6. **Metric `__class__()` reconstruction.**
   `with_updated_parameters` recreates metrics via `metric.__class__()`. If your metric has constructor arguments (like `ScaledPower(target_year=2032)`), they will be lost. Use metrics with no-arg constructors, or override `with_updated_parameters`.

7. **Single root node required.**
   `ExecutionGraph` assumes exactly one root node (node with out-degree 0). Multiple terminal nodes will raise an assertion error.

8. **Component parameter validation.**
   `Component.__init__` checks that all parameters required by its mutations are present. Missing parameters raise an `AssertionError` at construction time, not at execution time.

## Running the Project

```bash
# Install dependencies (conda recommended)
conda env create -f environment.yml
conda activate systemflow

# Run notebooks from the project root (required for classifier data paths)
cd /path/to/system_flow
jupyter notebook
```

## Key Files for Understanding the Framework

- Start with `examples/test_system.ipynb` for a tutorial walkthrough
- Read `systemflow/node.py` for all core primitives
- Read `systemflow/hep/mutations.py` for a complete domain-specific implementation
- Read `HEP/results_table_v2.ipynb` for a comprehensive HEP analysis example
- Read `XRS/ptychography.ipynb` for a System-level multi-graph example with feedback
