import networkx as nx
import numpy as np
import pandas as pd
import functools
import metrics
from scipy.stats import norm
from scipy.optimize import minimize_scalar

"""
Placeholder object for a node which doesn't classify (reject) data - only passes it
"""
class DummyClassifier:
    def __init__(self):
        self.ratio  = 1
        self.error_matrix = np.array([[0.0, 0.0], [0.0, 1.0]])

"""
Object to estimate the classification statistics of a processing node based on normal processes
"""
class Classifier:
    def __init__(self, ratio, skill, varscale = 1.0):
        self.skill = skill
        self.ratio  = ratio
        self.selectivity = 1 / ratio
        self.n = ratio + 1
        self.varscale = varscale

        #distribution of Y = 0 (reject) given X (data)
        self.false = lambda x: norm.cdf(x, loc=0.0, scale=varscale)
        #distribution of Y = 1 (accept) given X (data)
        self.true = lambda x: norm.cdf(x, loc=skill, scale=varscale)
        #data rejected given a threshold
        #assume the selectivity we're giving reflects the ratio of the true scores generated
        self.reject = lambda x: ((self.ratio) * self.false(x) + self.true(x))/ self.n
        #data accepted given a threshold
        self.accept = lambda x: 1.0 - self.reject(x)
        self.ratio_fn = lambda x: self.accept(x) / self.reject(x)
        self.threshold = self.solve_ratio()

        self.tn = self.false(self.threshold) * (ratio / self.n)
        self.fn = self.true(self.threshold) * (1 / self.n)
        self.tp = (1.0 - self.true(self.threshold)) * (1 / self.n)
        self.fp = (1.0 - self.false(self.threshold)) * (ratio / self.n)

        self.error_matrix = np.array([[self.tn, self.fn], [self.fp, self.tp]])

    def solve_ratio(self):
        opt_fn = lambda x: np.abs(self.selectivity - self.ratio_fn(x))
        soln = minimize_scalar(opt_fn, bounds=(0.0, 20.0))
        if soln.success:
            return soln.x
        else:
            print("Solving for classification threshold failed")
            
"""
Determine if a node has an active classifier
"""
def has_classifier(node):
    if "classifier" in node.keys():
        return type(node["classifier"]) is Classifier
    else:
        return False
    
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

    contingency = np.zeros_like(graph.nodes[classifiers[0]]["contingency"])
    for c in classifiers:
        #if there's another classifier after this, add only the rejection statistics
        if downstream_classifier(graph, c):
            contingency[0,:] += graph.nodes[c]["contingency"][0,:]
        else:
            contingency += graph.nodes[c]["contingency"]

    return contingency
    
"""
Given a pandas series that defines a classifier by data reduction and skill,
convert it to a classifier object or dummy classifier (no classification in node)
"""
def entry_to_classifier(entry: pd.core.series.Series):
    skill_u = entry["Skill mean"]
    skill_v = entry["Skill variance"]
    reduction = entry["Reduction"]
    
    if reduction == 1.0:
        classifier = DummyClassifier()
    else:
        classifier = Classifier(reduction, skill_u, varscale = skill_v)

    return classifier


"""
Error matrix representing a dummy classifier (no classification)
"""
def passing_node():
    error_matrix = np.array([[0.0, 0.0], [0.0, 1.0]])
    return error_matrix

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
        
        node_properties = {
                      "sample data": detector["Data (bytes)"],
                      "sample rate": detector["Sample Rate"],
                      "op efficiency": detector["Op Efficiency (J/op)"],
                      "error matrix": passing_node(),
                      "reduction": 1.0 - detector["Compression"],
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
        classifier = entry_to_classifier(trigger)

        node_properties = {
            "classifier": classifier,
            "error matrix": classifier.error_matrix,
            "reduction": 1.0 - trigger["Compression"],
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
    return roots[0]


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
    g.graph["Root Node"] = root[0]
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
Return the negative / positive classification rate from an error matrix
"""
def classifier_rate(error_matrix: np.ndarray):
        
        rates = np.einsum("ab,a -> a", error_matrix, np.array([1, 1]))
        return rates / rates.sum()

"""
Starting from the root node, determine the performance at each classifier node
"""
def contingency(graph: nx.classes.digraph):
    def calc_contingency(node):
        this_node = graph.nodes[node]
        this_node["contingency"] = np.round(this_node["error matrix"] * this_node["input rate"])

    list(map(calc_contingency, graph.nodes))

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
    total = total * this_node["reduction"]
    return total

"""
Return the rate of messages produced by a node
"""
def message_rate(graph: nx.classes.digraph, node: str):
    this_node = graph.nodes[node]
    if "sample rate" in this_node:
        #get the sampling rate of a detector
        input_rate = this_node["sample rate"]
    else:
        #or take the maximum rate of the inputs
        inputs = list(graph.predecessors(node))
        assert len(inputs) > 0, "Missing sample rate from detector or isolated node"
        input_rate = max([message_rate(graph, n) for n in inputs])

    this_node["input rate"] = input_rate
    #find how many outputs will be produced per second by this node
    output_rate = classifier_rate(this_node["error matrix"])[1] * input_rate
    this_node["message rate"] = output_rate
    return output_rate

"""
For each edge in a graph, calculate data throughput via each link
"""
def link_throughput(graph: nx.classes.digraph):
    def calc_throughput(edge):
        input_node = graph.nodes[edge[0]]
        throughput = input_node["message size"] * input_node["message rate"]
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
    #propagate from root to leaves
    message_size(graph, root)
    message_rate(graph, root)
    #update graph statistics (postprocess)
    link_throughput(graph)
    #calculate overall classifier performance
    contingency(graph)

    #calculate power resources & metrics
    graph.graph["link power"] = link_power(graph)
    graph.graph["op power"] = op_power(graph)
    graph.graph["performance"] = pipeline_contingency(graph)
    
    return graph
    

def graph_from_spreadsheet(filename: str, functions: dict):
    detectors = pd.read_excel(filename, sheet_name="Detectors")
    triggers = pd.read_excel(filename, sheet_name="Triggers")
    globals = pd.read_excel(filename, sheet_name="Global")

    graph = construct_graph(detectors, triggers, globals, functions)
    return graph