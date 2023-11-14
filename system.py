import networkx as nx
import numpy as np
import pandas as pd

def entry_to_confusion(entry: pd.core.series.Series):
    tp = entry["True Pass"]
    td = entry["True Discard"]
    fp = entry["False Pass (alpha)"]
    fd = entry["False Discard (beta)"]

    confusion = np.array([[td, fd], [fp, tp]])
    return confusion

def detectors(detector_data: pd.DataFrame):
    n = len(detector_data)
    nodes = []
    edges = []

    for i in range(n):
        detector = detector_data.iloc[i]
        name = detector["Detector"]
        system = detector["Category"]
        properties = {
                      "sample data": detector["Data (bytes)"],
                      "error matrix": entry_to_confusion(detector),
                      "reduction": 1.0 - detector["Compression"],
                      "complexity": lambda x: x,
                      }
        nodes.append((name, properties))
        edges.append((name, system))

    return nodes, edges

def triggers(trigger_data: pd.DataFrame):
    n = len(trigger_data)
    edges = []
    triggers = []

    for i in range(n):
        trigger = trigger_data.iloc[i]
        name = trigger["Name"]
        edge = (trigger["Input"], trigger["Output"])
        properties = {
            "error matrix": entry_to_confusion(trigger),
            "reduction": 1.0 - trigger["Compression"],
            "sample data": trigger["Data (bytes)"],
        }
        triggers.append((name, properties))
        edges.append(edge)

    return triggers, edges

def construct_graph(detector_data: pd.DataFrame, trigger_data: pd.DataFrame):
    g = nx.DiGraph()

    detector_nodes, detector_edges = detectors(detector_data)
    g.add_nodes_from(detector_nodes)

    trigger_nodes, trigger_edges = triggers(trigger_data)
    g.add_nodes_from(trigger_nodes)
    g.add_edges_from(detector_edges)
    g.add_edges_from(trigger_edges)
    return g

    globals = measure(g)
    for k, v in globals:
        g[k] = v

    return g, globals

