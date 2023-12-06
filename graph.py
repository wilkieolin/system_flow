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
                      "sample rate": detector["Sample Rate"],
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
        properties = {
            "error matrix": entry_to_confusion(trigger),
            "reduction": 1.0 - trigger["Compression"],
            "sample data": trigger["Data (bytes)"],
            "complexity": lambda x: x,
        }
        triggers.append((name, properties))

        output = trigger["Output"]
        if not pd.isna(output):
            edge = (trigger["Name"], trigger["Output"])
            edges.append(edge)

    return triggers, edges

def identify_root(graph: nx.classes.digraph):
    od = list(graph.out_degree)
    roots = list(filter(lambda x: x[1] == 0, od))
    assert len(roots) == 1, "More than 1 root identified"
    return roots[0]

def construct_graph(detector_data: pd.DataFrame, trigger_data: pd.DataFrame, functions: dict):
    g = nx.DiGraph()
    #add the nodes for detectors
    detector_nodes, detector_edges = detectors(detector_data)
    g.add_nodes_from(detector_nodes)
    #add the nodes for trigger systems
    trigger_nodes, trigger_edges = triggers(trigger_data)
    g.add_nodes_from(trigger_nodes)
    #connect the systems
    g.add_edges_from(detector_edges)
    g.add_edges_from(trigger_edges)

    #check that it's acyclic
    if not nx.is_directed_acyclic_graph(g):
        print("Graph must be a tree (acyclic), check definition")
        return None

    #update the complexity functions with those passed from the dictionary
    for n in g.nodes:
        if n in functions.keys():
            fn = functions[n]
            g.nodes[n]["complexity"] = fn

    #identify the final (root) node
    root = identify_root(g)
    g.graph["Root Node"] = root[0]
    g = update_throughput(g)
    return g


def lean_copy(graph: nx.classes.digraph):
    g = nx.DiGraph()
    for n in list(graph.nodes):
        g.add_node(n)

    for e in list(graph.edges):
        g.add_edge(*e)

    return g

def classifier_rate(error_matrix: np.ndarray):
        """
        Return the negative / positive classification rate from an error matrix
        """
        rates = np.einsum("ab,a -> a", error_matrix, np.array([1, 1]))
        return rates / rates.sum()

def message_size(graph: nx.classes.digraph, node: str):
    """
    Return the total size of a message processed by a node
    """
    inputs = list(graph.predecessors(node))
    input_data = sum([message_size(graph, n) for n in inputs])
    this_node = graph.nodes[node]
    if "sample data" in this_node:
        total = input_data + this_node["sample data"]
    else:
        total = input_data

    total = total * this_node["reduction"]
    this_node["message size"] = total
    this_node["ops"] = this_node["complexity"](total)
    return total

def message_rate(graph: nx.classes.digraph, node: str):
    """
    Return the rate of messages produced by a node
    """
    this_node = graph.nodes[node]
    if "sample rate" in this_node:
        #get the sampling rate of a detector
        input_rate = this_node["sample rate"]
    else:
        #or take the maximum rate of the inputs
        inputs = list(graph.predecessors(node))
        assert len(inputs) > 0, "Missing sample rate from detector or isolated node"
        input_rate = max([message_rate(graph, n) for n in inputs])
    
    output_rate = classifier_rate(this_node["error matrix"])[1] * input_rate
    this_node["message rate"] = output_rate
    return output_rate 

def link_throughput(graph: nx.classes.digraph):
    def calc_throughput(edge):
        input_node = graph.nodes[edge[0]]
        throughput = input_node["message size"] * input_node["message rate"]
        graph.edges[edge]["throughput"] = throughput

    list(map(calc_throughput, graph.edges))

def update_throughput(graph: nx.classes.digraph):
    graph = graph.copy()
    #call the recursive update function on the root node
    root = graph.graph["Root Node"]
    message_size(graph, root)
    message_rate(graph, root)
    link_throughput(graph)
    
    return graph
    