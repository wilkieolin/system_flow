# SystemFlow v2.0: Primitives and Architecture

This document describes the core primitives in the SystemFlow v2.0 framework (defined in `systemflow/node.py` and supporting modules) and how they compose to build models of scientific computing systems.

## Overview

SystemFlow models scientific data acquisition and processing systems as directed acyclic graphs (DAGs) where data originates at sensors, flows through processing stages, and terminates at storage or analysis endpoints. The framework provides a hierarchical set of abstractions:

```
System
  └── ExecutionGraph (one or more)
        ├── Component (nodes)
        │     └── Mutate (one or more, applied sequentially)
        ├── Link (edges)
        │     └── Transport (applied during data transfer)
        └── Metric (computed after execution)
```

Each level in this hierarchy is designed to be subclassed by the user to model domain-specific behaviors.

---

## Core Primitives

### 1. Message

**File:** `systemflow/auxtypes.py`

A `Message` is a named tuple that represents data flowing between components. It has two dictionaries:

- **fields**: Sample-varying data (e.g., image bytes, classification results, latencies). These change per sample processed.
- **properties**: Sample-independent metadata (e.g., resolution, sample rate, encoding format). These describe the format or context of the data.

```python
Message = namedtuple("Message", ["fields", "properties"])
```

Messages are immutable in spirit — mutations produce new messages via `merge_message()` rather than modifying in place. String keys are used as identifiers, with a convention of `"name (unit)"` (e.g., `"image data (B)"`, `"sample rate (Hz)"`).

### 2. VarCollection

**File:** `systemflow/auxtypes.py`

A convenience class for managing named references to message fields, properties, and component parameters. Instead of repeating string literals, you define a `VarCollection` once and reference its attributes:

```python
vc = VarCollection(resolution="resolution (n,n)", bitdepth="bit depth (n)")
# Then use vc.resolution instead of "resolution (n,n)" everywhere
```

This reduces typos and makes dependencies between mutations and components explicit. The `collect_parameters()` function aggregates all required host parameters from a list of mutations into a single `VarCollection`.

### 3. Regex

**File:** `systemflow/auxtypes.py`

A wrapper around a regex string used for pattern-matching in requirement checks and metrics. Allows mutations and metrics to match fields/properties by pattern rather than exact name — e.g., `Regex(r"\(B\)")` matches any field with bytes as its unit.

### 4. Mutate (Abstract)

**File:** `systemflow/node.py`

The fundamental unit of computation. A `Mutate` defines a transformation applied to a message within a component. Each mutation declares:

- **Inputs** (`MutationInputs` named tuple):
  - `msg_fields`: Required fields from the incoming message
  - `msg_properties`: Required properties from the incoming message
  - `host_parameters`: Required parameters from the host component

- **Outputs** (`MutationOutputs` named tuple):
  - `msg_fields`: New fields added to the outgoing message
  - `msg_properties`: New properties added to the outgoing message
  - `host_properties`: New properties imparted on the host component

The user implements the `transform()` method which reads from inputs and returns three dictionaries (new message fields, new message properties, new host properties). The base class handles requirement checking, deep copying, and merging automatically.

**Key built-in mutations** (`systemflow/mutations.py`):
- `CollectImage` — models 2D pixel sensor data acquisition
- `CollectTemperature` — models point sensor data acquisition
- `Convolve` — models convolution operation cost
- `FourierTransform2D` — models FFT operation cost
- `GaussianClassify` — models Gaussian-based binary classification
- `DataRate` — aggregates all data fields by regex matching on `(B)` units
- `StorageRate` — computes data throughput from total data and sample rate
- `StoreImage` — models accumulating images in a buffer

**Domain-specific mutations** (`systemflow/xrs.py`):
- `PositionSample` — models 2D sample stage positioning with movement latency
- `FlatFieldCorrection` — models flat-field image correction cost
- `MaskCorrection` — models masked pixel interpolation cost
- `PhaseReconstruction2D` — models iterative phase retrieval for single diffraction patterns
- `PhaseReconstruction3D` — models ptychographic reconstruction from overlapping patterns

### 5. Component (Abstract)

**File:** `systemflow/node.py`

A `Component` represents a piece of hardware (sensor, CPU, GPU, FPGA, storage device) that hosts one or more mutations applied sequentially to an incoming message. It has:

- **name**: Unique identifier within the graph
- **mutations**: Ordered list of `Mutate` instances applied sequentially
- **parameters**: Dictionary of control values (set at construction, constant during execution)
- **properties**: Dictionary of state values (updated by mutations during execution)
- **merge**: A `Merge` strategy for combining multiple input messages (default: `OverwriteMerge`)

When called, a component:
1. Finds its predecessor components in the graph
2. Recursively calls them to produce their output messages
3. Merges multiple input messages using its merge strategy
4. Sequentially applies its mutations via `itertools.accumulate`
5. Returns a new component instance with the output message and updated properties

### 6. Merge (Abstract)

**File:** `systemflow/merges.py`

Defines how multiple input messages are combined when a component has more than one predecessor. Each merge strategy specifies per-key reduction functions for fields and properties:

- **OverwriteMerge** (default): Takes the first value for any conflicting key
- Custom merges can specify functions per key, e.g., `np.min` for sample rates when a slower sensor rate should dominate

```python
class RateMerge(Merge):
    def __init__(self):
        super().__init__({}, {"sample rate (Hz)": np.min})
```

### 7. Link (Abstract) and Transport (Abstract)

**File:** `systemflow/node.py`

A `Link` represents a directed connection (edge) between two components. Each link can host a `Transport` operation that models effects of data transfer (e.g., communication power, latency, error rates).

- **DefaultLink**: A simple link with `DummyTransport` (pass-through, no modification)
- Custom links can model bandwidth limitations, power consumption per bit transferred, etc.

Links are evaluated after the graph executes, using the updated component states.

### 8. ExecutionGraph (Abstract)

**File:** `systemflow/node.py`

The central orchestrator. An `ExecutionGraph` manages a DAG of components and links, providing:

- **Construction**: Builds a `networkx.DiGraph` from components and links, validates acyclicity, identifies the single root (terminal) node
- **Execution** (`__call__`): Triggers recursive evaluation from root to leaves, producing a new graph instance with updated states
- **Parameter variation** (`with_updated_parameters`): Creates a new graph with modified component parameters without changing the original — essential for parameter sweeps
- **Metrics**: After execution, evaluates all attached `Metric` instances and stores results in `metric_values`
- **Introspection**: `get_node()`, `get_edge()`, `get_all_node_properties()`, `get_all_node_parameters()`, `get_output_msg()`, `list_nodes()`

Each call to an ExecutionGraph increments its `iteration` counter and returns a completely new graph instance (functional/immutable style).

### 9. Metric (Abstract)

**File:** `systemflow/node.py`, implementations in `systemflow/metrics.py`

A `Metric` computes derived values from an executed graph's state. It can query:

- **Message fields/properties** at the root node (via `msg_queries`, supporting `Regex`)
- **Component properties** across all nodes (via `host_queries`, supporting `Regex`)

Built-in metrics:
- `TotalOps` — sums all operation counts (matching `ops (n,n)`) across components
- `TotalLatency` — sums all latency fields (matching `latency (s)`) in the output message

### 10. System (Abstract)

**File:** `systemflow/node.py`

The highest-level abstraction, orchestrating multiple `ExecutionGraph` instances. A `System` provides:

- **step()**: Defines the order in which graphs execute within a single iteration
- **flow_control()**: Defines how the system iterates — can implement feedback loops, adaptive control, parameter updates between steps

This enables modeling of closed-loop systems where the output of one execution graph influences the input of another (e.g., adaptive scanning strategies in ptychography where reconstruction quality guides the next measurement position).

---

## How Primitives Interplay

### Data Flow
```
[Leaf Component] → Message → [Link/Transport] → [Component w/ Merge] → Mutate₁ → Mutate₂ → ... → Message → ... → [Root Component]
```

1. Leaf components (no predecessors) start with blank messages
2. Each mutation reads from the message and host parameters, then produces new fields/properties
3. Mutations are chained: output of Mutateₙ becomes input to Mutateₙ₊₁
4. When a component has multiple predecessors, their output messages are combined via the component's Merge strategy
5. The root node's output message is the final result of the graph

### Parameter Separation
The framework enforces a clear separation between:
- **Parameters** (component-level, set at construction, constant during execution) — represent hardware configuration
- **Fields** (message-level, varying per sample) — represent data that changes with each measurement
- **Properties** (message-level, sample-independent) — represent format/context metadata
- **Host properties** (component-level, set by mutations) — represent derived component state like power consumption

### Functional Style
All execution produces new instances rather than modifying in place:
- `Mutate.__call__` returns a new `Message` and new host properties
- `Component.__call__` returns a new `Component` with output message
- `ExecutionGraph.__call__` returns a new `ExecutionGraph` with all updated nodes
- `with_updated_parameters` returns a new graph with modified parameters

This makes parameter sweeps natural: create variations of a graph, execute each independently, collect results.

---

## Classifier Integration

**File:** `systemflow/classifier.py`

Classifiers model the statistical performance of data reduction stages (triggers). They estimate error matrices (2×2 confusion matrices) given:
- Input sample rates (positive vs. negative)
- A target reduction ratio

Available classifiers:
- **DummyClassifier** — passes all data (no rejection)
- **GaussianClassifier** — models classification with Gaussian-distributed scores parameterized by skill (distribution separation) and variance
- **L1TClassifier** — models CMS Level-1 Trigger using empirical efficiency curves fit to real CMS data across 4 trigger types (jet, muon, egamma, tau)
- **HLTClassifier** — models CMS High-Level Trigger using empirical efficiency curves across 6 reconstruction paths (B2G, Higgs, muon, SUSY, tracking, tau)

In the v2.0 framework, classifiers are integrated via the `GaussianClassify` mutation, which wraps a `GaussianClassifier` and produces contingency tables and error matrices as message fields/properties.
