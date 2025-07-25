"""
Copyright 2025, UChicago Argonne LLC. 
Please refer to 'license' in the root directory for details and disclosures.
"""

import networkx as nx
import numpy as np
import pandas as pd
import functools
from systemflow.metrics import *
from systemflow.classifier import *
from collections import namedtuple

"""
Determine if a node has an active classifier
"""
def has_classifier(node):
    #determine if it's a dummy classifier
    if "classifier" in node.keys():
        return not issubclass(type(node["classifier"]), DummyClassifier)
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
Return the amount of energy expended by the system to reach the current node
"""
def upstream_energy(graph, node):
    def get_energy(node):
        if "energy" in node.keys():
            return node["energy"]
        else:
            return 0.0
    
    def traverse(start):
        up = list(graph.predecessors(start))

        if len(up) == 0:
            return get_energy(graph.nodes[start])
        else:
            return get_energy(graph.nodes[start]) + functools.reduce(lambda x, y: x + y, map(traverse, up))
    
    return traverse(node)

"""
Return the latency required to route to the current node
"""
def propagate_latency(graph, node):
    def arrival_latency(predecessors):
        latencies = [graph.nodes[n]["routing latency"] for n in predecessors]
        if len(latencies) > 0:
            latency = np.max(latencies)
        else:
            latency = 0.0
        
        return latency
    
    def traverse(start):
        up = list(graph.predecessors(start))
        this_node = graph.nodes[start]
        processing_latency = this_node["op latency"] * this_node["parallelism"](this_node["ops"])

        if len(up) == 0:
            message_time = processing_latency
        else:
            message_time = processing_latency + arrival_latency(up) + np.max(list(map(traverse, up)))
        this_node["message time"] = message_time
        return message_time

    traverse(node)
    return

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

    #initialize the contingency matrix
    contingency = np.zeros_like(graph.nodes[classifiers[0]]["contingency"])
    for c in classifiers:
        #if there's another classifier after this, add only the rejection statistics
        if downstream_classifier(graph, c):
            contingency[0,:] += graph.nodes[c]["contingency"][0,:]
        else:
            contingency += graph.nodes[c]["contingency"]

    return contingency
    
def quantify_error_cost(graph):
    #the cost of a true positive is the cost to get to the final node
    positive = upstream_energy(graph, graph.graph["Root Node"])
    #the cost of a true negative is the average energy for a discarded message
    classifiers = active_classifiers(graph)
    energy = [upstream_energy(graph, c) for c in classifiers]
    negatives = [np.sum(graph.nodes[c]["discards"]) for c in classifiers]
    negative = np.average(energy, weights=negatives)

    return (negative, positive)

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
        routing_latency = detector.get("Routing Latency", 900e-9) #default of 900 ns
        
        node_properties = {
                      "sample data": detector["Data (bytes)"],
                      "sample rate": detector["Sample Rate"],
                      "type": "detector",
                      "op efficiency": detector["Op Efficiency (J/op)"],
                      "op latency": 1.0e-9,
                      "classifier": classifier,
                      "error matrix": classifier.error_matrix,
                      "routing latency": routing_latency,
                      "reduction ratio": 1.0,
                      "reduction": 0.0, #by definition, a detector produces data and does not reject any
                      "data reduction": 1.0 - detector["Compression"], #it can compress it, though
                      "complexity": lambda x: x,
                      "parallelism": lambda x: 1,
                      }
        nodes.append((name, node_properties))

        edge_properties = {"link efficiency": detector["Link Efficiency (J/bit)"],}
        edges.append((name, detector["Category"], edge_properties))

    return nodes, edges

"""
Return nodes and edges from a dataframe representing the system's data processing nodes (triggers)
"""
def processors(processor_data: pd.DataFrame):
    n = len(processor_data)
    edges = []
    processors = []

    for i in range(n):
        processor = processor_data.iloc[i]
        name = processor["Name"]
        rr = processor["Reduction Ratio"]
        classifier_type = processor["Classifier"]
        routing_latency = processor.get("Routing Latency", 900e-9) #default of 900 ns

        if classifier_type == "Gaussian":
            classifier = GaussianClassifier(processor["Skill mean"], processor["Skill variance"])
        elif classifier_type == "L1T":
            classifier = L1TClassifier()
        elif classifier_type == "HLT":
            classifier = HLTClassifier()
        else:
            classifier = DummyClassifier()

        node_properties = {
            "type": "processor",
            "reduction ratio": rr,
            "classifier": classifier,
            "data reduction": 1.0 - processor["Compression"],
            "op efficiency": processor["Op Efficiency (J/op)"],
            "op latency": 1.0e-9,
            "routing latency": routing_latency,
            "sample data": processor["Data (bytes)"],
            "complexity": lambda x: x,
            "parallelism": lambda x: 1,
        }
        
        
        processors.append((name, node_properties))

        output = processor["Output"]
        if not pd.isna(output):
            edge_properties = {"link efficiency": processor["Link Efficiency (J/bit)"],}
            edge = (processor["Name"], processor["Output"], edge_properties)
            edges.append(edge)

    return processors, edges

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
def construct_graph(detector_data: pd.DataFrame, processor_data: pd.DataFrame, globals: pd.DataFrame, functions: dict):
    g = nx.DiGraph()
    #add the nodes for detectors
    detector_nodes, detector_edges = detectors(detector_data)
    g.add_nodes_from(detector_nodes)
    #add the nodes for processor systems
    processor_nodes, processor_edges = processors(processor_data)
    g.add_nodes_from(processor_nodes)
    #connect the systems
    g.add_edges_from(detector_edges)
    g.add_edges_from(processor_edges)
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
        #graph.nodes[node]["classifier"].reduction = 1 / graph.nodes[node]["reduction ratio"]
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
        node["contingency"] = np.array([[0, 0], [negatives, positives]]).astype("int")
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
        inputs = inputs.astype("int")
        
        #construct the classifier model for this node
        node["input rate"] = np.sum(inputs)
        
        #obtain results produced through this classifier
        reduction = ratio_to_reduction(node["reduction ratio"])
        statistics = node["classifier"](inputs, reduction)
        node["error matrix"] = node["classifier"].error_matrix
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
    propagate_latency(graph, root)
    
    #calculate overall classifier performance

    #calculate power resources & metrics
    graph.graph["link power"] = link_power(graph)
    graph.graph["op power"] = op_power(graph)
    graph.graph["performance"] = pipeline_contingency(graph)
    
    return graph
    
System = namedtuple("System", ["detectors", "processors", "globals"]) 

def dataframes_from_spreadsheet(filename: str):
    detectors = pd.read_excel(filename, sheet_name="Detectors")
    processors = pd.read_excel(filename, sheet_name="Processors")
    globals = pd.read_excel(filename, sheet_name="Global")

    sys = System(detectors, processors, globals)
    return sys

def graph_from_spreadsheet(filename: str, functions: dict):
    detectors, processors, globals = dataframes_from_spreadsheet(filename)
    graph = construct_graph(detectors, processors, globals, functions)

    return graph