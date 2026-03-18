"""
HEP-specific metrics for CMS-like data acquisition system models.

These metrics replicate the post-processing computations from graph.py
(pipeline_contingency, op_power, link_power) as v2.0 Metric subclasses.
"""

import numpy as np
from functools import reduce

from systemflow.node import Metric, ExecutionGraph, Message
from systemflow.classifier import get_passed, get_rejected
from systemflow.metrics import precision, recall, f1_score
from systemflow.models import density_scale_model


class PipelineContingency(Metric):
    """
    Aggregates contingency tables across all active classifiers in the pipeline.

    Replicates pipeline_contingency() from graph.py:
    - For classifiers with a downstream classifier, only rejections (row 0) are added
    - For the final classifier, the full contingency table is added

    Produces: "pipeline contingency (2x2)" in metric_values
    """
    def __init__(self):
        super().__init__("Pipeline Contingency", [], [])

    def _has_downstream_classifier(self, graph, node_name):
        """Check if any node downstream of node_name has an active classifier."""
        successors = list(graph.graph.successors(node_name))
        for s in successors:
            s_component = graph.get_node(s)
            if s_component.properties.get("classifier active", False):
                return True
            if self._has_downstream_classifier(graph, s):
                return True
        return False

    def __call__(self, graph: ExecutionGraph):
        # Find all nodes with active classifiers
        active_nodes = []
        for node in graph.nodes:
            if node.properties.get("classifier active", False):
                active_nodes.append(node)

        if len(active_nodes) == 0:
            return {"pipeline contingency (2x2)": np.zeros((2, 2))}

        contingency = np.zeros_like(active_nodes[0].properties["contingency (2x2)"])
        for node in active_nodes:
            if self._has_downstream_classifier(graph, node.name):
                # Only add rejections from intermediate classifiers
                contingency[0, :] += node.properties["contingency (2x2)"][0, :]
            else:
                # Add full contingency from the final classifier
                contingency += node.properties["contingency (2x2)"]

        return {"pipeline contingency (2x2)": contingency}


class SystemPower(Metric):
    """
    Computes total operational and link power across the system.

    Replicates op_power() + link_power() from graph.py.

    Produces:
    - "op power (W)": total operational power across all components
    - "link power (W)": total communication power across all links
    - "total power (W)": sum of op + link power
    """
    def __init__(self):
        super().__init__("System Power", [], [])

    def __call__(self, graph: ExecutionGraph):
        # Sum operational power from all components
        op_power = sum(
            node.properties.get("power (W)", 0.0)
            for node in graph.nodes
        )

        # Sum link power from all links
        # Note: Link.__call__ stores transport results in the new link's
        # `parameters` dict (not `properties`), due to how Link.__call__ works.
        link_power = sum(
            link.parameters.get("power (W)", 0.0)
            for link in graph.links
        )

        return {
            "op power (W)": op_power,
            "link power (W)": link_power,
            "total power (W)": op_power + link_power,
        }


class Productivity(Metric):
    """
    Computes system productivity: (F1 × output_rate) / total_power.

    This is the key figure of merit used across HEP analyses to evaluate
    the efficiency of different system configurations.

    Depends on PipelineContingency and SystemPower being computed first
    (metrics are evaluated in order).

    Produces:
    - "precision (%)": pipeline precision
    - "recall (%)": pipeline recall
    - "f1 score (%)": pipeline F1 score
    - "accuracy (%)": pipeline accuracy
    - "output rate (Hz)": events stored per second
    - "productivity (ev/J)": F1 × output_rate / total_power
    """
    def __init__(self):
        super().__init__("Productivity", [], [])

    def __call__(self, graph: ExecutionGraph):
        # Get pipeline contingency from the already-computed metrics
        # Since metrics are reduced with |, prior metrics are available
        # We need to compute it ourselves since we can't rely on ordering
        pc_metric = PipelineContingency()
        pc_result = pc_metric(graph)
        contingency = pc_result["pipeline contingency (2x2)"]

        # Get power
        sp_metric = SystemPower()
        sp_result = sp_metric(graph)
        total_power = sp_result["total power (W)"]

        # Classification metrics
        p = precision(contingency)
        r = recall(contingency)
        f1 = f1_score(contingency)
        tp = contingency[1, 1]
        tn = contingency[0, 0]
        acc = (tp + tn) / np.sum(contingency) if np.sum(contingency) > 0 else 0.0

        # Output rate from root node
        output_rate = graph.root_node.properties.get("output rate (Hz)", 0.0)

        # Productivity
        productivity = (f1 * output_rate) / total_power if total_power > 0 else 0.0

        return {
            "pipeline contingency (2x2)": contingency,
            "op power (W)": sp_result["op power (W)"],
            "link power (W)": sp_result["link power (W)"],
            "total power (W)": total_power,
            "precision (%)": p,
            "recall (%)": r,
            "f1 score (%)": f1,
            "accuracy (%)": acc,
            "output rate (Hz)": output_rate,
            "productivity (ev/J)": productivity,
        }


class ScaledPower(Metric):
    """
    Applies technology scaling to project power consumption to a target year.

    Uses density_scale_model() from systemflow.models to estimate how
    transistor density improvements reduce power over time.

    Produces:
    - "scaled power (W)": total power normalized by density scaling
    - "scale factor": the density scaling factor applied
    """
    def __init__(self, target_year: float = 2032):
        self.target_year = target_year
        super().__init__("Scaled Power", [], [])

    def __call__(self, graph: ExecutionGraph):
        sp_metric = SystemPower()
        sp_result = sp_metric(graph)
        total_power = sp_result["total power (W)"]

        scale = density_scale_model(self.target_year)
        scaled_power = total_power / scale

        return {
            "scaled power (W)": scaled_power,
            "scale factor": scale,
        }
