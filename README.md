# SystemFlow

**Modeling dependencies and effects in scientific data processing systems.**

SystemFlow is a Python framework for constructing, simulating, and analyzing scientific data processing pipelines as directed acyclic graphs (DAGs). It enables modeling of hardware components, data transformations, classification stages, and inter-component data transport to evaluate system-level metrics such as power consumption, throughput, latency, and classification performance.

This repository also contains the code and data necessary to reproduce the results for the paper "[Modeling Performance of Data Collection Systems for High-Energy Physics](https://pubs.aip.org/aip/aml/article/2/4/046113/3326300)."

## Installation

The required Python environment can be constructed in Conda using the included environment file:

```bash
conda env create -f environment.yml
conda activate systemflow
pip install -e .

If you need the historical processor data for the CPU scaling figure, clone the submodule:

```bash
git clone --recursive git@github.com:wilkieolin/system_flow.git
```

## Package Overview

### Core Framework (`systemflow/node.py`)

The framework provides a hierarchy of abstractions for modeling data processing systems:

- **Message** -- Named tuple representing data flowing through the system, carrying both fields (transformed by mutations) and properties (metadata passed through).
- **Mutate** -- Abstract transformation applied to a message within a component (e.g., data acquisition, convolution, classification).
- **Component** -- A hardware node that hosts one or more mutations and processes incoming messages sequentially.
- **Link / Transport** -- Directed edges connecting components, with optional transport effects (e.g., power cost of data transfer).
- **Merge** -- Strategy for combining multiple input messages when a component has several predecessors.
- **ExecutionGraph** -- DAG orchestrator that manages components, links, execution order, and parameter variation.
- **Metric** -- Derived quantities computed from the execution state (e.g., total power, productivity).
- **System** -- Top-level container orchestrating multiple execution graphs.

### Domain-Specific Modules

#### High-Energy Physics (`systemflow/hep/`)

CMS-specific components for modeling the detector, trigger, and readout pipeline:

- **Mutations** (`systemflow/hep/mutations.py`) -- `HEPDetector`, `HEPProcessor`, `HEPMerge`, `HEPLinkTransport`
- **Metrics** (`systemflow/hep/metrics.py`) -- `PipelineContingency`, `SystemPower`, `Productivity`, `ScaledPower`
- **Utilities** (`systemflow/hep/utils.py`) -- `hep_graph_from_spreadsheet` to build a complete execution graph from a CMS system spreadsheet, `hep_with_updated_parameters` for parameter sweeps

#### X-Ray Scattering (`systemflow/xrs.py`)

Mutations for coherent diffraction imaging workflows: `PositionSample`, `FlatFieldCorrection`, `MaskCorrection`, `PhaseReconstruction2D`, `PhaseReconstruction3D`.

#### Classifiers (`systemflow/classifier.py`)

Stochastic and deterministic classifiers used to model trigger and selection stages:

- `L1TClassifier` / `HLTClassifier` -- CMS Level-1 and High-Level Trigger models built from measured efficiency curves
- `GaussianClassifier` -- Parameterized by skill and variance for general-purpose modeling
- `DummyClassifier` -- Pass-through (no rejection)

#### General-Purpose Mutations (`systemflow/mutations.py`)

Reusable building blocks: `CollectImage`, `CollectTemperature`, `Convolve`, `FourierTransform2D`, `DataRate`, `StorageRate`, and others.

#### Technology Models (`systemflow/models.py`)

Transistor density and performance scaling models for projecting system performance across technology generations.

## Getting Started

### Building a system from a spreadsheet (HEP)

```python
from systemflow.hep.utils import hep_graph_from_spreadsheet

graph = hep_graph_from_spreadsheet("HEP/configurations/cms_system_200.xlsx")
graph.execute()

# Evaluate metrics
from systemflow.hep.metrics import SystemPower, Productivity
power = SystemPower()
prod = Productivity()
print("Power:", power(graph), "W")
print("Productivity:", prod(graph))
```

### Parameter sweeps

```python
from systemflow.hep.utils import hep_with_updated_parameters

# Create a variant with updated parameters
updated_graph = hep_with_updated_parameters(graph, {"L1T": {"reduction": 0.01}})
updated_graph.execute()
```

### Example notebooks

- [HEP/results_table_v2.ipynb](HEP/results_table_v2.ipynb) -- Reproduces paper tables and demonstrates basic graph construction and metric evaluation
- [HEP/vary_system_v2.ipynb](HEP/vary_system_v2.ipynb) -- Parameter sweeps across system configurations
- [XRS/ptychography.ipynb](XRS/ptychography.ipynb) -- Ptychographic imaging pipeline with online search strategy
- [examples/test_system.ipynb](examples/test_system.ipynb) -- Minimal example of the core framework

## Reproducing Publication Figures

The figures and tables from the paper can be reproduced using the notebooks below.

The illustrative figures 2, 3, and 4 were created in Adobe Illustrator and are not included as part of this repository.

### Figure 1 -- CPU Scaling

Notebook: [figures/cpu_scaling.ipynb](figures/cpu_scaling.ipynb)

Note: ensure that the submodule containing historical processor data is downloaded. If necessary, run `git clone --recursive git@github.com:wilkieolin/system_flow.git` to clone the submodule.

### Figure 5 -- Pileup vs. L1T

Notebook: [HEP/pileup_vs_l1t_v2.ipynb](HEP/pileup_vs_l1t_v2.ipynb)

### Figure 6 -- System Variation

Notebook: [HEP/vary_system_v2.ipynb](HEP/vary_system_v2.ipynb)

### Tables 1 & 2

Notebook: [HEP/results_table_v2.ipynb](HEP/results_table_v2.ipynb)

### Additional Analysis Notebooks

These notebooks provide further analyses beyond the publication figures:

- [HEP/vary_hlt_v2.ipynb](HEP/vary_hlt_v2.ipynb) -- HLT parameter variation studies
- [HEP/reduction_vs_productivity_v2.ipynb](HEP/reduction_vs_productivity_v2.ipynb) -- Reduction ratio vs. productivity trade-offs
- [HEP/system_gradients_v2.ipynb](HEP/system_gradients_v2.ipynb) -- System gradient analysis

## CMS System Configurations

Predefined system configurations are provided as Excel spreadsheets in [HEP/configurations/](HEP/configurations/):

| File | Description |
|------|-------------|
| `cms_system_60.xlsx` | CMS at 60 pileup |
| `cms_system_140.xlsx` | CMS at 140 pileup |
| `cms_system_200.xlsx` | CMS at 200 pileup |
| `cms_system_200_smartpx.xlsx` | CMS at 200 pileup with smart pixel detectors |
| `cms_system_200_smartpx_uled.xlsx` | CMS at 200 pileup with smart pixel + ULED links |
| `fcchh_system_950.xlsx` | FCC-hh at 950 pileup |

## License

Copyright 2025, UChicago Argonne LLC. See [LICENSE](LICENSE) for details.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/wilkieolin/system_flow/HEAD?labpath=HEP/vary_hlt_v2.ipynb)
