import networkx as nx
import numpy as np
import pandas as pd
import functools
import metrics
from classifier import *

"""
Determine if a node has an active classifier
"""
def has_classifier(node):
    #determine if it's a dummy classifier
    if "classifier" in node.keys():
        return type(node["classifier"]) is Classifier
    #determine if the classifier is active
    else:
        return node["classifier"].active
    
"""
Given a node, determine if there's a processing node after it which contains an active classifier
"""
def downstream_classifier(graph, node):
    def traverse(start):
        down = list(graph.successors(start))
        
        if len(down) == 0:
            return has_classifier(graph.nodes[start])
        else:
            return has_classifier(graph.nodes[start]) + functools.reduce(lambda x, y: x + y, map(traverse, down))
        
    return traverse(node) > 1

"""
Find the nodes in a graph with active classifiers
"""
def active_classifiers(graph):
    nodes = [(n, has_classifier(graph.nodes[n])) for n in graph.nodes()]
    nodes = list(filter(lambda x: x[1], nodes))
    nodes = [n[0] for n in nodes]
    return nodes

"""
Determine the classification contingency table for the entire pipeline
"""
def pipeline_contingency(graph):
    #find the nodes which classify
    classifiers = active_classifiers(graph)
    assert len(classifiers) > 0, "No classifiers in processing graph"

    #initialize the
    contingency = np.zeros_like(graph.nodes[classifiers[0]]["contingency"])
    for c in classifiers:
        #if there's another classifier after this, add only the rejection statistics
        if downstream_classifier(graph, c):
            contingency[0,:] += graph.nodes[c]["contingency"][0,:]
        else:
            contingency += graph.nodes[c]["contingency"]

    return contingency
    

"""
Return nodes and edges from a dataframe representing the system's data sources (detectors)
"""
def detectors(detector_data: pd.DataFrame):
    n = len(detector_data)
    nodes = []
    edges = []

    for i in range(n):
        detector = detector_data.iloc[i]
        name = detector["Detector"]
        classifier = DummyClassifier()
        
        node_properties = {
                      "sample data": detector["Data (bytes)"],
                      "sample rate": detector["Sample Rate"],
                      "type": "detector",
                      "op efficiency": detector["Op Efficiency (J/op)"],
                      "classifier": classifier,
                      "error matrix": classifier.error_matrix,
                      "reduction ratio": 1.0,
                      "reduction": 0.0, #by definition, a detector produces data and does not reject any
                      "data reduction": 1.0 - detector["Compression"], #it can compress it, though
                      "complexity": lambda x: x,
                      }
        nodes.append((name, node_properties))

        edge_properties = {"link efficiency": detector["Link Efficiency (J/bit)"],}
        edges.append((name, detector["Category"], edge_properties))

    return nodes, edges

"""
Return nodes and edges from a dataframe representing the system's data processing nodes (triggers)
"""
def triggers(trigger_data: pd.DataFrame):
    n = len(trigger_data)
    edges = []
    triggers = []

    for i in range(n):
        trigger = trigger_data.iloc[i]
        name = trigger["Name"]
        rr = trigger["Reduction Ratio"]
        reduction = ratio_to_reduction(rr)
        classifier_properties = [reduction, trigger["Skill mean"], trigger["Skill variance"]]

        node_properties = {
            "classifier properties": classifier_properties,
            "type": "processor",
            "reduction ratio": rr,
            "reduction": reduction,
            "data reduction": 1.0 - trigger["Compression"],
            "op efficiency": trigger["Op Efficiency (J/op)"],
            "sample data": trigger["Data (bytes)"],
            "complexity": lambda x: x,
        }
        triggers.append((name, node_properties))

        output = trigger["Output"]
        if not pd.isna(output):
            edge_properties = {"link efficiency": trigger["Link Efficiency (J/bit)"],}
            edge = (trigger["Name"], trigger["Output"], edge_properties)
            edges.append(edge)

    return triggers, edges

"""
From a graph, identify the root of the tree (final processing node / storage)
"""
def identify_root(graph: nx.classes.digraph):
    od = list(graph.out_degree)
    roots = list(filter(lambda x: x[1] == 0, od))
    assert len(roots) == 1, "More than 1 root identified"
    node = roots[0][0]
    graph.nodes[node]["type"] = "storage"
    return node


"""
Given dataframes representing the system's detectors, processing nodes, and processor scaling estimates,
construct the graph representing it
"""
def construct_graph(detector_data: pd.DataFrame, trigger_data: pd.DataFrame, globals: pd.DataFrame, functions: dict):
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
    #add other information
    g.graph["globals"] = globals

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
    g.graph["Root Node"] = root
    g = update_throughput(g)
    return g


"""
Make a copy of a graph without attributes (names only) - for plotting
"""
def lean_copy(graph: nx.classes.digraph):
    g = nx.DiGraph()
    for n in list(graph.nodes):
        g.add_node(n)

    for e in list(graph.edges):
        g.add_edge(*e)

    return g

"""
Recursively process message passing over the graph to determine the data through each node
"""
def message_size(graph: nx.classes.digraph, node: str):
    #find the inputs to this node
    inputs = list(graph.predecessors(node))
    #list how much data each produces
    input_data = sum([message_size(graph, n) for n in inputs])
    this_node = graph.nodes[node]

    #calculate the total number of bits accepted as input
    if "sample data" in this_node:
        total = input_data + this_node["sample data"]
    else:
        total = input_data

    this_node["message size"] = total

    #predict the processing cost this will incur
    this_node["ops"] = this_node["complexity"](total)
    #apply any data reduction which might take place
    total = total * this_node["data reduction"]
    return total

"""
For each node, calculate the ratio of messages it produces to the number stored
at the final node
"""
def calc_rejection(graph: nx.classes.digraph):
    def inner(node):
        downstream = list(nx.dfs_postorder_nodes(graph, node))
        ratios = [graph.nodes[n]["reduction ratio"] for n in downstream]
        total_reduction = functools.reduce(lambda x, y: x * y, ratios)
        graph.nodes[node]["global ratio"] = total_reduction

    list(map(inner, graph.nodes))


"""
Propagate the classification of relevant/irrelevant messages kept and discarded
through the pipeline
"""
def propagate_statistics(graph: nx.classes.digraph, node_name: str):
    node = graph.nodes[node_name]

    #if the node is a detector, begin the error propagation
    if node["type"] == "detector":
        #determine the number of true and false samples which will propagate out
        positives = node["sample rate"] / node["global ratio"]
        negatives = node["sample rate"] - positives
        node["contingency"] = np.array([[0, 0], [negatives, positives]])
        node["input rate"] = node["sample rate"]
        node["output rate"] = node["sample rate"]
        output = get_passed(node["contingency"])
        node["discards"] = get_rejected(node["contingency"])

        return output

    #if it's a processor, recurse through the inputs
    else:
        previous = list(graph.predecessors(node_name))
        n_previous = len(previous)

        #collect statistics over inputs
        inputs = [propagate_statistics(graph, n) for n in previous]
        #store inputs into the incoming edges
        edges = [(n, node_name) for n in previous]
        for (i,e) in enumerate(edges):
            graph.edges[e]["statistics"] = inputs[i]
        
        #take the average over the input nodes to collate inputs into a single file
        inputs = functools.reduce(lambda x, y: x + y, inputs) / n_previous
        #construct the classifier model for this node
        node["input rate"] = np.sum(inputs)
        classifier = Classifier(*node["classifier properties"], inputs=inputs)
        node["classifier"] = classifier
        node["error matrix"] = classifier.error_matrix

        #obtain results produced through this classifier
        statistics = classifier(inputs)
        node["contingency"] = statistics
        #separate messages discarded & accepted by classifier
        node["discards"] = get_rejected(statistics)
        output = get_passed(statistics)
        node["output rate"] = np.sum(output)

        return output


"""
For each edge in a graph, calculate data throughput via each link
"""
def link_throughput(graph: nx.classes.digraph):
    def calc_throughput(edge):
        input_node = graph.nodes[edge[0]]
        throughput = input_node["message size"] * input_node["output rate"]
        graph.edges[edge]["message size"] = input_node["message size"]
        graph.edges[edge]["throughput"] = throughput

    list(map(calc_throughput, graph.edges))

def link_power(graph: nx.classes.digraph):
    def calc_power(edge):
        #include bytes to bits conversion
        e = edge["link efficiency"] * edge["message size"] * 8
        edge["energy"] = e

        p = edge["link efficiency"] * edge["throughput"] * 8
        edge["power"] = p
        
        return p
    
    power = [calc_power(graph.edges[e]) for e in graph.edges]
    total = functools.reduce(lambda x, y: x + y, power)
    return total

def op_power(graph: nx.classes.digraph):
    def calc_power(node):
        e = node["op efficiency"] * node["ops"]
        node["energy"] = e

        p = e * node["input rate"]
        node["power"] = p

        return p
    
    power = [calc_power(graph.nodes[n]) for n in graph.nodes]
    total = functools.reduce(lambda x, y: x + y, power)
    return total
    

"""
Given a graph, propagate message sizes, estimate processing requirements, 
calculate data rates, and produce classification statistics
"""
def update_throughput(graph: nx.classes.digraph):
    graph = graph.copy()
    #call the recursive update function on the root node
    root = graph.graph["Root Node"]
    #calculate the global reduction ratio
    calc_rejection(graph)
    #propagate from root to leaves
    
    message_size(graph, root)
    propagate_statistics(graph, root)
    #update graph statistics (postprocess)
    link_throughput(graph)
    
    #calculate overall classifier performance
    #propagate_statistics

    #calculate power resources & metrics
    graph.graph["link power"] = link_power(graph)
    graph.graph["op power"] = op_power(graph)
    graph.graph["performance"] = pipeline_contingency(graph)
    
    return graph
    
def dataframes_from_spreadsheet(filename: str):
    detectors = pd.read_excel(filename, sheet_name="Detectors")
    triggers = pd.read_excel(filename, sheet_name="Triggers")
    globals = pd.read_excel(filename, sheet_name="Global")

    return detectors, triggers, globals

def graph_from_spreadsheet(filename: str, functions: dict):
    detectors, triggers, globals = dataframes_from_spreadsheet(filename)
    graph = construct_graph(detectors, triggers, globals, functions)

    return graph