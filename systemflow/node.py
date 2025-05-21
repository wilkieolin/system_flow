import networkx as nx
import sys
from ruamel.yaml import YAML
import numpy as np
from abc import ABC, abstractmethod
from collections import namedtuple
from copy import deepcopy
from functools import reduce
from typing import Callable, Any
from itertools import accumulate



# Message handling
Message = namedtuple("Message", ["fields", "properties"])

class Merge(ABC):
    def __init__(self, field_merges: dict[str, Callable], property_merges: dict[str, Callable]):
        super().__init__()
        self.field_merges = field_merges
        self.property_merges = property_merges

    """
    If only one dictionary has a value defined, take that value. If there are more,
    reduce by the function in "merges" if it exists, otherwise take the first value.
    """
    def merge_dictionaries(self, dicts: list[dict], merges: dict[str, Callable]) -> dict:
        # Get all unique keys across dictionaries
        all_keys = reduce(lambda x, y: x.union(y), (set(d.keys()) for d in dicts))
        keys_list = list(all_keys)
        if len(keys_list) == 0:
            return {}
        
        # Create boolean matrix showing key presence in each dictionary
        matches = np.array([[k in d for d in dicts] for k in keys_list])
        key_counts = np.sum(matches, axis=1)  # Count per key
        
        merged_dict = {}
        for idx, count in enumerate(key_counts):
            current_key = keys_list[idx]
            
            if count == 1:  # Single dictionary has the key
                dict_idx = np.where(matches[idx])[0][0]
                merged_dict[current_key] = dicts[dict_idx][current_key]
            else:  # Multiple dictionaries have the key
                # Get values from all dictionaries containing the key
                values = [d[current_key] for d in dicts if current_key in d]
                
                if current_key in merges:  # Use custom merge function
                    merged_dict[current_key] = merges[current_key](values)
                else:  # Take first occurrence
                    print("No merge provided for " + current_key + ", taking first value")
                    merged_dict[current_key] = values[0]
                    
        return merged_dict


    def __call__(self, messages: list[Message]) -> Message:
        fields = [message.fields for message in messages]
        properties = [message.properties for message in messages]

        fields = self.merge_dictionaries(fields, self.field_merges)
        properties = self.merge_dictionaries(properties, self.property_merges)

        merged_message = Message(fields, properties)
        return merged_message

class OverwriteMerge(Merge):
    def __init__(self):
        super().__init__({}, {})
    
# Mutations

"""
Abstract class which defines the template for component augmentation operatiuons.
Mutate transforms take a set input fields from an incoming message and create or change a field.
These transforms can impart a set of properties (power consumption, etc) on the host component.
"""
class Mutate(ABC):
    def __init__(self, msg_fields: list[str], msg_properties: list[str], host_parameters: list[str]):
        #fields contain the data necessary for the transform
        self.msg_fields = msg_fields 
        #parameters control the behavior of the transform
        self.msg_properties = msg_properties 
        self.host_parameters = host_parameters

    def _missing_keys(self, matches: list, values: list) -> str:
        missing = []
        for (i,m) in enumerate(matches):
            if not m:
                missing.append(values[i])

        missing_fields = reduce(lambda x, y: x + ", " + y, missing)
        return missing_fields

    def _field_check(self, message: Message) -> None:
        matches = [f in message.fields.keys() for f in self.msg_fields]
        field_chk = np.all(matches)
        if not field_chk:
            missing = self._missing_keys(matches, self.msg_fields)
            assert field_chk, "Input field for transform not found in incoming message: " + missing

    def _property_check(self, message: Message) -> None:
        matches = [f in message.properties.keys() for f in self.msg_properties]
        props_chk = np.all(matches)
        if not props_chk:
            missing = self._missing_keys(matches, self.msg_properties)
            assert props_chk, "Transform's properties not found in incoming message: " + missing
    
    def _param_check(self, component: 'Component') -> None:
        matches = [f in component.parameters.keys() for f in self.host_parameters]
        params_chk = np.all(matches)
        if not params_chk:
            missing = self._missing_keys(matches, self.host_parameters)
            assert params_chk, "Transform's control parameters not found in host component: " + missing

    def transform(self, component: 'Component', message: Message) -> tuple[dict, dict, dict]:
        new_msg_fields = {}
        new_msg_properties = {}
        new_host_properties = {}
        
        # USER - calculate new fields/properties for message/component here
        return new_msg_fields, new_msg_properties, new_host_properties
    
    def __call__(self, component: 'Component', message: Message) -> tuple[Message, dict]:
        #check that all incoming messages have the field(s)/parameters necessary for the transform
        self._field_check(message)
        #check that the incoming message has the parameters necessary for the transform
        self._property_check(message)
        #check that the host component has the parameters necessary for the transform
        self._param_check(component)

        #create independent copies of the message and component
        component = deepcopy(component)
        message = deepcopy(message)
        #determine the new fields for the message and properties for message and host
        new_msg_fields, new_msg_props, new_host_props = self.transform(component, message)
        #merge information into a new outgoing message
        new_message = Message(message.fields | new_msg_fields, message.properties | new_msg_props)

        return new_message, new_host_props
    
# Transformation nodes

class Component(ABC):
    def __init__(self, name: str, mutations: list[Mutate], parameters: dict, properties: dict, merge: Merge = OverwriteMerge()) -> None:
        super().__init__()
        self.name = name
        self.merge = merge
        assert len(mutations) > 0, "Should have at least one mutation in a component"
        self.mutations = mutations
        self.parameters = parameters
        self.properties = properties

    def blank_message(self) -> Message:
        message = Message({}, {})
        return message

    def __call__(self, exg: 'ExecutionGraph') -> 'Component':
        #find which components send messages here
        predecessors = exg.get_predecessors(self)
        if len(predecessors) > 0:
            #gather their input messages
            input_components = [node(exg) for node in predecessors]
            input_messages = [component.output_msg for component in input_components]
            #merge them
            input_msg = self.merge(input_messages)
        else:
            #otherwise, create a blank message
            input_msg = self.blank_message()
        
        if len(self.mutations) > 0:
            #go through the mutations on this component
            mutations = list(accumulate(self.mutations, lambda x, f: f(x[0], x[1]), initial=(self, input_msg)))
            #separate the message and property outputs
            properties = [output[1] for output in mutations[1:]]
            merged_properties = reduce(lambda x, y: x | y, properties) | self.properties
            output_msg = mutations[-1][0]
        else:
            #pass through the existing properties & merged message if no mutations present
            merged_properties = self.properties
            output_msg = input_msg

        #store the new output message in the new component
        new_component = Component(self.name, self.mutations, self.parameters, merged_properties)
        new_component.output_msg = output_msg
        
        return new_component
    
# Transport Processes

class Transport(ABC):
    def __init__(self, msg_fields: list[str], msg_properties: list[str], parameters: list[str]) -> None:
        self.msg_fields = msg_fields
        self.msg_properties = msg_properties
        self.parameters = parameters
        self.properties = {}

    def _missing_keys(self, matches: list, values: list) -> str:
        missing = []
        for (i,m) in enumerate(matches):
            if not m:
                missing.append(values[i])

        missing_fields = reduce(lambda x, y: x + ", " + y, missing)
        return missing_fields

    def _field_check(self, message: Message) -> None:
        matches = [f in message.fields.keys() for f in self.msg_fields]
        field_chk = np.all(matches)
        if not field_chk:
            missing = self._missing_keys(matches, self.msg_fields)
            assert field_chk, "Input field for transform not found in transmitted message: " + missing

    def _property_check(self, message: Message) -> None:
        matches = [f in message.properties.keys() for f in self.msg_properties]
        props_chk = np.all(matches)
        if not props_chk:
            missing = self._missing_keys(matches, self.msg_properties)
            assert props_chk, "Transform's properties not found in transmitted message: " + missing
    
    def _param_check(self, link: 'Link') -> None:
        matches = [f in link.parameters.keys() for f in self.host_parameters]
        params_chk = np.all(matches)
        if not params_chk:
            missing = self._missing_keys(matches, self.host_parameters)
            assert params_chk, "Transform's control parameters not found in host link: " + missing

    def transport(self, link: 'Link', message: Message) -> dict:
        parameters = link.parameters
        new_properties = {}
        # USER - calculate new properties for the link here
        return new_properties

    def __call__(self, link: 'Link') -> dict:
        tx_message = link.tx.output_msg
        #check that all incoming messages have the field(s)/parameters necessary for the transform
        self._field_check(tx_message)
        #check that the incoming message has the parameters necessary for the transform
        self._property_check(tx_message)
        #check that the host component has the parameters necessary for the transform
        self._param_check(link)
        
        new_properties = self.transport(link, tx_message)
        return new_properties
    
class DummyTransport(Transport):
    def __init__(self):
        super().__init__([], [], [])

#Links

class Link(ABC):
    def __init__(self, name: str, tx: Component, rx: Component, transport: Transport, parameters: dict):
        self.name = name
        self.tx = tx
        self.rx = rx
        self.transport = transport
        self.parameters = parameters
        self.properties = {}

    def __call__(self) -> 'Link':
        new_properties = self.transport(self)
        new_link = Link(self.name, self.tx, self.rx, self.transport, self.properties | new_properties)
        return new_link

class DefaultLink(Link):
    def __init__(self, name, tx, rx):
        super().__init__(name, tx, rx, [], {})

# Execution graphs
def flatten_tuple(t):
    for item in t:
        if isinstance(item, tuple):
            yield from flatten_tuple(item)
        else:
            yield item
"""
Contains the graph of execution flow for a task
"""    
class ExecutionGraph(ABC):
    """
    Construct a new execution graph given a name, list of nodes (producers and components), and list of edges
    (links between nodes).
    """
    def __init__(self, name: str, nodes: list[Component], links: list[Link], iteration: int = 0):
        self.name = name
        self.nodes = nodes
        self.links = links
        self.iteration = iteration

        nodes = [(n.name, {"ref": n,}) for n in nodes]
        edges = [(l.tx, l.rx, {"ref": l}) for l in links]
        self.graph = self.construct_graph(nodes, edges)
        self.root = self.identify_root()

    def construct_graph(self, nodes: list[Component], edges: list[Link]) -> nx.classes.digraph:
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.graph["name"] = self.name
        #check that it's acyclic
        assert nx.is_directed_acyclic_graph(g), "Graph must be a tree (acyclic), check definition"

        return g
    
    def get_node(self, name: str) -> Component:
        g = self.graph
        component = g.nodes[name]["ref"]
        return component
    
    def get_edge(self, tx: str, rx: str) -> Link:
        g = self.graph
        link = g.edges[(tx, rx)]["ref"]
        return link

    """
    Identify and return the root node of the execution graph
    """
    def identify_root(self,):
        od = list(self.graph.out_degree)
        roots = list(filter(lambda x: x[1] == 0, od))
        assert len(roots) == 1, "More than 1 root identified"
        node = roots[0][0]
        return node
    
    """
    Retrieve the nodes which are the parents of a given component
    """
    def get_predecessors(self, node: Component) -> list[Component]:
        up = list(self.graph.predecessors(node.name))
        components = [self.get_node(name) for name in up]
        return components
    
    """
    Make a copy of a graph without attributes (names only) - for plotting
    """
    def lean_copy(self):
        graph = self.graph
        lean_copy = nx.DiGraph()
        for n in list(graph.nodes):
            lean_copy.add_node(n)

        for e in list(graph.edges):
            lean_copy.add_edge(*e)

        return lean_copy
    
    def __call__(self) -> 'ExecutionGraph':
        #Get the root processing node and propagate calls recursively up from it
        root_node = self.get_node(self.root)
        new_nodes = flatten_tuple(root_node(self))
        #update the link information given the new nodes
        new_links = [link(new_nodes) for link in self.links]
        #create the new execution graph
        exg = ExecutionGraph(self.name, new_nodes, new_links, self.iteration + 1)
        return exg

"""
Contains & orchestrates a series of execution graphs implemented by a system
"""
class System(ABC):
    def __init__(self, name: str, exec_graphs: list[ExecutionGraph], iter: int = 0):
        self.name = name
        self.exec_graphs = exec_graphs

    def __call__(self) -> 'System':
        new_graphs = [graph() for graph in self.exec_graphs]
        new_system = System(self.name, new_graphs, self.iter + 1)
        return new_system
    
# Basic & example implementations

class Convolve(Mutate):
    def __init__(self,):
        fields = ["image data"]
        properties = ["sensor"]
        parameters = ["kernel x", "kernel y", "filters"]
        super().__init__(fields, properties, parameters)

    def transform(self, component: Component, message: Message):
        #access the required fields/properties/parameters
        res = message.properties["sensor"]
        kernel_x = component.parameters["kernel x"]
        kernel_y = component.parameters["kernel y"]
        filters = component.parameters["filters"]

        #calculate the number of ops required for the kernel
        kernel_ops = kernel_x * kernel_y * filters
        steps_x = (res[0] - kernel_x) // kernel_x
        steps_y = (res[1] - kernel_y) // kernel_y
        kernel_repeats = steps_x * steps_y

        #calculate the number of ops required for the kernel
        transform_operations = kernel_ops * kernel_repeats

        properties = {}
        properties["transform operations"] = transform_operations
        message.fields["features"] = np.prod((steps_x, steps_y, filters)) * ureg.byte

        return message, properties
       
    