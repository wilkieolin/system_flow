"""
Utilities for constructing HEP system models from CMS spreadsheet configurations.

Replaces graph_from_spreadsheet(), construct_graph(), detectors(), and
processors() from systemflow.graph (v1.0) with v2.0 Component/ExecutionGraph
construction.
"""

import numpy as np
import pandas as pd
import functools
import networkx as nx

from systemflow.node import Component, Link, ExecutionGraph
from systemflow.classifier import (
    DummyClassifier, GaussianClassifier, L1TClassifier, HLTClassifier
)

from systemflow.hep.mutations import (
    HEPDetector, HEPProcessor, HEPMerge, HEPLinkTransport
)
from systemflow.hep.metrics import Productivity


def _compute_global_ratios(detector_data, processor_data):
    """
    Pre-compute the global reduction ratio for each detector.

    The global ratio is the product of all reduction ratios downstream of a node.
    For detectors, this equals the product of all processor reduction ratios
    in the path from the detector to the storage node.

    This replicates calc_rejection() from graph.py but computed from the
    spreadsheet data before graph construction.
    """
    # Build a temporary networkx graph to compute topology
    g = nx.DiGraph()

    # Add detector edges: detector -> category processor
    for i in range(len(detector_data)):
        det = detector_data.iloc[i]
        g.add_edge(det["Detector"], det["Category"])

    # Add processor edges and store reduction ratios
    reduction_ratios = {}
    for i in range(len(processor_data)):
        proc = processor_data.iloc[i]
        name = proc["Name"]
        reduction_ratios[name] = proc["Reduction Ratio"]
        output = proc["Output"]
        if not pd.isna(output):
            g.add_edge(name, output)

    # For detectors, reduction_ratio = 1.0
    for i in range(len(detector_data)):
        det = detector_data.iloc[i]
        reduction_ratios[det["Detector"]] = 1.0

    # Compute global ratio for each node
    global_ratios = {}
    for node in g.nodes:
        downstream = list(nx.dfs_postorder_nodes(g, node))
        ratios = [reduction_ratios.get(n, 1.0) for n in downstream]
        global_ratios[node] = functools.reduce(lambda x, y: x * y, ratios)

    return global_ratios


def _build_detector_components(detector_data, global_ratios):
    """Build Component instances for each detector in the spreadsheet."""
    components = []
    for i in range(len(detector_data)):
        det = detector_data.iloc[i]
        name = det["Detector"]
        routing_latency = det.get("Routing Latency", 900e-9)

        component = Component(
            name=name,
            mutations=[HEPDetector()],
            parameters={
                "sample data (B)": det["Data (bytes)"],
                "sample rate (Hz)": det["Sample Rate"],
                "data reduction (%)": 1.0 - det["Compression"],
                "op efficiency (J/op)": det["Op Efficiency (J/op)"],
                "routing latency (s)": routing_latency,
                "global ratio (1)": global_ratios[name],
                "complexity (fn)": lambda x: x,
            },
        )
        components.append(component)
    return components


def _build_processor_components(processor_data):
    """Build Component instances for each processor in the spreadsheet."""
    components = []
    for i in range(len(processor_data)):
        proc = processor_data.iloc[i]
        name = proc["Name"]
        classifier_type = proc["Classifier"]
        routing_latency = proc.get("Routing Latency", 900e-9)

        if classifier_type == "Gaussian":
            classifier = GaussianClassifier(proc["Skill mean"], proc["Skill variance"])
        elif classifier_type == "L1T":
            classifier = L1TClassifier()
        elif classifier_type == "HLT":
            classifier = HLTClassifier()
        else:
            classifier = DummyClassifier()

        # Processors that receive from multiple predecessors need HEPMerge
        merge = HEPMerge()

        component = Component(
            name=name,
            mutations=[HEPProcessor()],
            parameters={
                "sample data (B)": proc["Data (bytes)"],
                "data reduction (%)": 1.0 - proc["Compression"],
                "op efficiency (J/op)": proc["Op Efficiency (J/op)"],
                "routing latency (s)": routing_latency,
                "reduction ratio (1)": proc["Reduction Ratio"],
                "classifier (obj)": classifier,
                "complexity (fn)": lambda x: x,
            },
            merge=merge,
        )
        components.append(component)
    return components


def _build_links(detector_data, processor_data):
    """Build Link instances from the spreadsheet topology."""
    links = []

    # Detector -> Category processor links
    for i in range(len(detector_data)):
        det = detector_data.iloc[i]
        tx_name = det["Detector"]
        rx_name = det["Category"]
        link_name = f"{tx_name} -> {rx_name}"
        link_eff = det["Link Efficiency (J/bit)"]

        link = Link(
            name=link_name,
            tx=tx_name,
            rx=rx_name,
            transport_op=HEPLinkTransport(),
            parameters={"link efficiency (J/bit)": link_eff},
        )
        links.append(link)

    # Processor -> Processor links
    for i in range(len(processor_data)):
        proc = processor_data.iloc[i]
        output = proc["Output"]
        if not pd.isna(output):
            tx_name = proc["Name"]
            rx_name = output
            link_name = f"{tx_name} -> {rx_name}"
            link_eff = proc["Link Efficiency (J/bit)"]

            link = Link(
                name=link_name,
                tx=tx_name,
                rx=rx_name,
                transport_op=HEPLinkTransport(),
                parameters={"link efficiency (J/bit)": link_eff},
            )
            links.append(link)

    return links


def dataframes_from_spreadsheet(filename: str):
    """
    Read a CMS system configuration spreadsheet and return DataFrames.

    Returns:
        tuple: (detectors_df, processors_df, globals_df)
    """
    detectors = pd.read_excel(filename, sheet_name="Detectors")
    processors = pd.read_excel(filename, sheet_name="Processors")
    globals_df = pd.read_excel(filename, sheet_name="Global")

    return detectors, processors, globals_df


def hep_graph_from_spreadsheet(filename: str, functions: dict = {},
                                metrics: list = None):
    """
    Read a CMS system configuration spreadsheet and construct a v2.0
    ExecutionGraph.

    Replaces graph_from_spreadsheet() from graph.py.

    Args:
        filename: Path to the Excel spreadsheet with Detectors, Processors,
                  and Global sheets.
        functions: Dict mapping component names to complexity functions.
                   Overrides the default identity function (lambda x: x).
        metrics: List of Metric instances. Defaults to [Productivity()].

    Returns:
        ExecutionGraph: A v2.0 execution graph ready to be called.
    """
    if metrics is None:
        metrics = [Productivity()]

    detector_data, processor_data, globals_df = dataframes_from_spreadsheet(filename)

    # Compute global ratios for all nodes
    global_ratios = _compute_global_ratios(detector_data, processor_data)

    # Build components
    detector_components = _build_detector_components(detector_data, global_ratios)
    processor_components = _build_processor_components(processor_data)

    # Apply custom complexity functions
    all_components = detector_components + processor_components
    for comp in all_components:
        if comp.name in functions:
            comp.parameters["complexity (fn)"] = functions[comp.name]

    # Build links
    links = _build_links(detector_data, processor_data)

    # Build execution graph
    graph = ExecutionGraph(
        name=filename,
        nodes=all_components,
        links=links,
        metrics=metrics,
    )

    return graph


def _recompute_global_ratios(graph):
    """
    Recompute global_ratio for each detector node based on current
    processor reduction ratios in the graph.

    Returns a parameter update map that updates all detector components'
    global_ratio parameters.
    """
    g = graph.graph  # networkx DiGraph

    # Collect reduction ratios from all nodes
    reduction_ratios = {}
    for node in graph.nodes:
        rr = node.parameters.get("reduction ratio (1)", 1.0)
        reduction_ratios[node.name] = rr

    # For detector nodes, compute product of downstream reduction ratios
    updates = {}
    for node in graph.nodes:
        if "global ratio (1)" in node.parameters:
            downstream = list(nx.dfs_postorder_nodes(g, node.name))
            ratios = [reduction_ratios.get(n, 1.0) for n in downstream]
            new_ratio = functools.reduce(lambda x, y: x * y, ratios)
            if new_ratio != node.parameters["global ratio (1)"]:
                updates[node.name] = {"global ratio (1)": new_ratio}

    return updates


def hep_with_updated_parameters(graph, new_parameters_map):
    """
    Create a new ExecutionGraph with updated parameters, automatically
    recomputing detector global_ratios when reduction ratios change.

    This wraps ExecutionGraph.with_updated_parameters() to handle the
    dependency between processor reduction ratios and detector global ratios.

    Args:
        graph: The original ExecutionGraph
        new_parameters_map: Dict mapping component names to parameter dicts

    Returns:
        A new ExecutionGraph with all parameters updated consistently.
    """
    # First apply the user's parameter changes
    updated_graph = graph.with_updated_parameters(new_parameters_map)

    # Then recompute global ratios if any reduction ratios changed
    ratio_changed = any(
        "reduction ratio (1)" in params
        for params in new_parameters_map.values()
    )

    if ratio_changed:
        ratio_updates = _recompute_global_ratios(updated_graph)
        if ratio_updates:
            updated_graph = updated_graph.with_updated_parameters(ratio_updates)

    return updated_graph


def hep_construct_graph(detector_data, processor_data, globals_df,
                         functions: dict = {}, metrics: list = None):
    """
    Construct a v2.0 ExecutionGraph from DataFrames.

    Replaces construct_graph() from graph.py.

    Args:
        detector_data: DataFrame with detector specifications
        processor_data: DataFrame with processor specifications
        globals_df: DataFrame with global parameters
        functions: Dict mapping component names to complexity functions
        metrics: List of Metric instances

    Returns:
        ExecutionGraph: A v2.0 execution graph ready to be called.
    """
    if metrics is None:
        metrics = [Productivity()]

    global_ratios = _compute_global_ratios(detector_data, processor_data)
    detector_components = _build_detector_components(detector_data, global_ratios)
    processor_components = _build_processor_components(processor_data)

    all_components = detector_components + processor_components
    for comp in all_components:
        if comp.name in functions:
            comp.parameters["complexity (fn)"] = functions[comp.name]

    links = _build_links(detector_data, processor_data)

    graph = ExecutionGraph(
        name="CMS System",
        nodes=all_components,
        links=links,
        metrics=metrics,
    )

    return graph
