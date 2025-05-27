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
from collections.abc import Iterable


# Message handling
#: A named tuple representing a message passed between components.
#:
#: Attributes:
#:  fields (dict): A dictionary containing the primary data of the message.
#:  properties (dict): A dictionary containing metadata or secondary attributes of the message.
Message = namedtuple("Message", ["fields", "properties"])

class Merge(ABC):
    """
    Abstract Base Class for defining strategies to merge multiple Message objects.

    When multiple messages are input to a Component, a Merge strategy dictates
    how their fields and properties are combined into a single message.
    """
    def __init__(self, field_merges: dict[str, Callable], property_merges: dict[str, Callable]):
        super().__init__()
        self.field_merges = field_merges
        self.property_merges = property_merges

    """
    If only one dictionary has a value defined, take that value. If there are more,
    reduce by the function in "merges" if it exists, otherwise take the first value.
    """
    def merge_dictionaries(self, dicts: list[dict], merges: dict[str, Callable]) -> dict:
        """
        Merges a list of dictionaries into a single dictionary.

        For each key present across the input dictionaries:
        - If the key exists in only one dictionary, its value is taken.
        - If the key exists in multiple dictionaries:
            - If a specific merge function is provided for that key in `merges`,
              it's used to combine the values.
            - Otherwise, a warning is printed, and the value from the first
              dictionary (in the order they appear in `values`) is taken.

        Args:
            dicts: A list of dictionaries to merge.
            merges: A dictionary mapping keys to callable functions. Each function
                    should accept a list of values and return a single merged value.

        Returns:
            A new dictionary containing the merged key-value pairs.
        """
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
        """
        Merges a list of Message objects into a single Message.

        Args:
            messages: A list of Message objects to be merged.

        Returns:
            A new Message object containing the merged fields and properties.
        """
        fields = [message.fields for message in messages]
        properties = [message.properties for message in messages]

        fields = self.merge_dictionaries(fields, self.field_merges)
        properties = self.merge_dictionaries(properties, self.property_merges)

        merged_message = Message(fields, properties)
        return merged_message

class OverwriteMerge(Merge):
    """
    A simple Merge strategy where conflicting keys are resolved by taking the
    value from the first message encountered. No specific merge functions are defined,
    so it relies on the default behavior of `Merge.merge_dictionaries`.
    """
    def __init__(self):
        super().__init__({}, {})
    
# Mutations

"""
Abstract class which defines the template for component augmentation operatiuons.
Mutate transforms take a set of input fields/properties from an incoming message
and/or parameters from its host component to create or change fields in the message
and properties in the host component.
These transforms can impart a set of properties (power consumption, etc) on the host component.
"""
class Mutate(ABC):
    """
    Abstract Base Class for operations that transform a Message and/or its host Component.

    A Mutate object defines a specific transformation, specifying the required
    input message fields, message properties, and host component parameters.
    When called, it applies this transformation and can update the message
    and/or add new properties to the host component.
    """
    def __init__(self, name: str, msg_fields: list[str], msg_properties: list[str], host_parameters: list[str]):
        self.name = name
        #fields contain the data necessary for the transform
        self.msg_fields = msg_fields 
        #parameters control the behavior of the transform
        self.msg_properties = msg_properties 
        self.host_parameters = host_parameters

    def _missing_keys(self, matches: list, values: list) -> str:
        """
        Helper function to generate a comma-separated string of missing items.

        Args:
            matches: A list of booleans indicating presence (True) or absence (False).
            values: A list of corresponding item names.

        Returns:
            A string listing the names of items where `matches` was False.
        """
        missing = []
        for (i,m) in enumerate(matches):
            if not m:
                missing.append(values[i])

        missing_fields = reduce(lambda x, y: x + ", " + y, missing)
        return missing_fields

    def _field_check(self, message: Message) -> None:
        """
        Checks if all required message fields are present in the incoming message.

        Args:
            message: The input Message object.

        Raises:
            AssertionError: If any of the fields specified in `self.msg_fields` are missing.
        """
        matches = [f in message.fields.keys() for f in self.msg_fields]
        field_chk = np.all(matches)
        if not field_chk:
            missing = self._missing_keys(matches, self.msg_fields)
            assert field_chk, "Input field for transform " + self.name + " not found in incoming message: " + missing

    def _property_check(self, message: Message) -> None:
        """
        Checks if all required message properties are present in the incoming message.

        Args:
            message: The input Message object.

        Raises:
            AssertionError: If any of the properties specified in `self.msg_properties` are missing.
        """
        matches = [f in message.properties.keys() for f in self.msg_properties]
        props_chk = np.all(matches)
        if not props_chk:
            missing = self._missing_keys(matches, self.msg_properties)
            assert props_chk, self.name + " transform's properties not found in incoming message: " + missing
    
    def _param_check(self, component: 'Component') -> None:
        """
        Checks if all required host parameters are present in the host Component.

        Args:
            component: The host Component object.

        Raises:
            AssertionError: If any of the parameters specified in `self.host_parameters` are missing.
        """
        matches = [f in component.parameters.keys() for f in self.host_parameters]
        params_chk = np.all(matches)
        if not params_chk:
            missing = self._missing_keys(matches, self.host_parameters)
            assert params_chk, self.name + " transform's control parameters not found in host component: " + missing

    def transform(self, message: Message, component: 'Component') -> tuple[dict, dict, dict]:
        """
        The core transformation logic to be implemented by subclasses.

        This method should access data from the `message` (fields and properties)
        and `component` (parameters), perform calculations, and return new
        data to be incorporated into the output message and host component.

        Args:
            message: The input Message object (a deep copy).
            component: The host Component object (a deep copy).

        Returns:
            A tuple containing three dictionaries:
            - new_msg_fields: New fields to be added to or overwrite in the message.
            - new_msg_properties: New properties to be added to or overwrite in the message.
            - new_host_properties: New properties to be added to or overwrite in the host component.
        """
        new_msg_fields = {}
        new_msg_properties = {}
        new_host_properties = {}
        
        # USER - calculate new fields/properties for message/component here
        return new_msg_fields, new_msg_properties, new_host_properties
    
    def __call__(self, message: Message, component: 'Component') -> tuple[Message, dict]:
        """
        Applies the mutation to a message and its host component.

        This method first performs checks for required fields, properties, and parameters.
        Then, it calls the `transform` method to get the new data and merges this
        data into a new Message object and updates the host component's properties.

        Args:
            message: The input Message object.
            component: The host Component that this mutation is part of.

        Returns:
            A tuple containing:
            - new_message: The transformed Message object.
            - new_host_props: A dictionary of new properties for the host component.
        """
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
        new_msg_fields, new_msg_props, new_host_props = self.transform(message, component)
        #merge information into a new outgoing message
        new_message = Message(message.fields | new_msg_fields, message.properties | new_msg_props)

        return new_message, new_host_props
    
# Transformation nodes

class Component(ABC):
    """
    Represents a node in the execution graph that processes messages.

    A Component hosts a list of `Mutate` operations that are applied sequentially
    to an incoming message (or a merged message if there are multiple inputs).
    It also has its own set of parameters that can control the behavior of its
    mutations, and properties that can be updated by these mutations.

    Attributes:
        name (str): The unique name of the component.
        mutations (list[Mutate]): A list of mutation operations to apply.
        parameters (dict): Configuration parameters for this component and its mutations.
        properties (dict): State properties of this component, potentially modified by mutations.
        merge (Merge): A Merge strategy object for combining multiple input messages.
    """
    def __init__(self, name: str, mutations: list[Mutate], parameters: dict, properties: dict, merge: Merge = OverwriteMerge()) -> None:
        super().__init__()
        self.name = name
        self.merge = merge
        assert len(mutations) > 0, "Should have at least one mutation in a component"
        self.mutations = mutations
        self.parameters = parameters
        self.properties = properties

    def blank_message(self) -> Message:
        """
        Creates an empty Message object.

        Used as a starting point if a component has no predecessors.

        Returns:
            An empty Message(fields={}, properties={}).
        """
        message = Message({}, {})
        return message

    def __call__(self, exg: 'ExecutionGraph', verbose: bool = False) -> list['Component', 'Component']:
        """
        Executes the component's logic within the context of an ExecutionGraph.

        This involves:
        1. Identifying predecessor components in the graph.
        2. Recursively calling them to get their output messages.
        3. Merging input messages if there are multiple predecessors.
        4. Applying its sequence of `Mutate` operations to the (merged) input message.
        5. Storing the final output message and updated properties in a new instance of itself.

        Args:
            exg: The ExecutionGraph this component belongs to.
            verbose: If True, prints execution status.

        Returns:
            A tuple: (new_component_state, predecessor_results).
            - new_component_state: A new Component instance representing the state after execution,
                                   with `output_msg` and updated `properties`.
            - predecessor_results: A list containing the results (similar tuples) from calling
                                   each predecessor component. This forms a nested structure
                                   representing the execution trace of the upstream graph.
        """
        #find which components send messages here
        if verbose:
            print("Executing on node ", self.name)
        predecessors = exg.get_predecessors(self)
        if len(predecessors) > 0:
            #gather their input messages
            input_components = [node(exg, verbose) for node in predecessors]
            input_messages = [component[0].output_msg for component in input_components]
            #merge them
            input_msg = self.merge(input_messages)
        else:
            #otherwise, create a blank message
            input_components = []
            input_msg = self.blank_message()
        
        if len(self.mutations) > 0:
            #go through the mutations on this component
            mutations = list(accumulate(self.mutations, lambda x, f: f(x[0], self), initial=(input_msg, {})))
            #separate the message and property outputs
            properties = [output[1] for output in mutations]
            merged_properties = reduce(lambda x, y: x | y, properties) | self.properties
            output_msg = mutations[-1][0]
        else:
            #pass through the existing properties & merged message if no mutations present
            merged_properties = self.properties
            output_msg = input_msg

        #store the new output message in the new component
        new_component = Component(self.name, self.mutations, self.parameters, merged_properties)
        new_component.output_msg = output_msg
        
        return new_component, input_components
    
# Transport Processes

class Transport(ABC):
    """
    Abstract Base Class for defining operations that occur on a Link during message transfer.

    A Transport object can modify a message or the link itself as data
    is "transmitted" from one Component to another. It specifies required
    message fields/properties and link parameters.

    Subclasses should implement the `transport` method.
    """
    def __init__(self, msg_fields: list[str], msg_properties: list[str], parameters: list[str]) -> None:
        self.msg_fields = msg_fields
        self.msg_properties = msg_properties
        self.parameters = parameters
        self.properties = {}

    def _missing_keys(self, matches: list, values: list) -> str:
        """
        Helper function to generate a comma-separated string of missing items.

        Args:
            matches: A list of booleans indicating presence (True) or absence (False).
            values: A list of corresponding item names.

        Returns:
            A string listing the names of items where `matches` was False.
        """
        missing = []
        for (i,m) in enumerate(matches):
            if not m:
                missing.append(values[i])

        missing_fields = reduce(lambda x, y: x + ", " + y, missing)
        return missing_fields

    def _field_check(self, message: Message) -> None:
        """
        Checks if all required message fields are present in the transmitted message.

        Args:
            message: The Message object being transmitted.

        Raises:
            AssertionError: If any of the fields specified in `self.msg_fields` are missing.
        """
        matches = [f in message.fields.keys() for f in self.msg_fields]
        field_chk = np.all(matches)
        if not field_chk:
            missing = self._missing_keys(matches, self.msg_fields)
            assert field_chk, "Input field for transform not found in transmitted message: " + missing

    def _property_check(self, message: Message) -> None:
        """
        Checks if all required message properties are present in the transmitted message.

        Args:
            message: The Message object being transmitted.

        Raises:
            AssertionError: If any of the properties specified in `self.msg_properties` are missing.
        """
        matches = [f in message.properties.keys() for f in self.msg_properties]
        props_chk = np.all(matches)
        if not props_chk:
            missing = self._missing_keys(matches, self.msg_properties)
            assert props_chk, "Transform's properties not found in transmitted message: " + missing
    
    def _param_check(self, link: 'Link') -> None:
        """
        Checks if all required parameters are present in the host Link.

        Args:
            link: The Link object hosting this transport operation.

        Raises:
            AssertionError: If any of the parameters specified in `self.parameters` are missing.
        """
        matches = [f in link.parameters.keys() for f in self.parameters]
        params_chk = np.all(matches)
        if not params_chk:
            missing = self._missing_keys(matches, self.parameters)
            assert params_chk, "Transform's control parameters not found in host link: " + missing

    def transport(self, link: 'Link', message: Message) -> dict:
        """
        The core transport logic to be implemented by subclasses.

        This method defines how the link's properties are affected by the
        transmission of the message.

        Args:
            link: The Link object on which the transport occurs.
            message: The Message being transmitted.

        Returns:
            A dictionary of new properties to be updated on the link.
        """
        parameters = link.parameters
        new_properties = {}
        # USER - calculate new properties for the link here
        return new_properties

    def __call__(self, link: 'Link', tx_node: Component) -> dict:
        """
        Applies the transport operation to a link.

        Args:
            link: The Link object.
            tx_node: The transmitting Component (source of the message).

        Returns:
            A dictionary of new properties for the link, resulting from the transport operation.
        """
        tx_message = tx_node.output_msg
        #check that all incoming messages have the field(s)/parameters necessary for the transform
        self._field_check(tx_message)
        #check that the incoming message has the parameters necessary for the transform
        self._property_check(tx_message)
        #check that the host component has the parameters necessary for the transform
        self._param_check(link)
        
        new_properties = self.transport(link, tx_message)
        return new_properties
    
class DummyTransport(Transport):
    """
    A transport operation that does nothing.

    It requires no specific message fields, properties, or link parameters,
    and results in no changes to the link's properties.
    """
    def __init__(self):
        super().__init__([], [], [])

#Links

class Link(ABC):
    """
    Represents a directed connection (edge) between two Components in an ExecutionGraph.

    A Link can have an associated `Transport` operation that models effects
    of message transmission over this link (e.g., latency, data loss, power consumption).

    Attributes:
        name (str): The name of the link.
        tx (str): The name of the transmitting (source) Component.
        rx (str): The name of the receiving (destination) Component.
        transport_op (Transport): The transport operation associated with this link.
        parameters (dict): Parameters controlling the transport operation.
        properties (dict): Properties of the link, potentially modified by the transport operation.
    """
    def __init__(self, name: str, tx: Component, rx: Component, transport_op: Transport, parameters: dict):
        self.name = name
        self.tx = tx
        self.rx = rx
        self.transport_op = transport_op
        self.parameters = parameters
        self.properties = {}

    def get_node_from_list(self, name: str, nodes: list[Component]) -> Component:
        """
        Retrieves a Component from a list by its name.

        Args:
            name: The name of the Component to find.
            nodes: A list of Component objects.

        Returns:
            The Component object with the matching name.

        Raises:
            AssertionError: If no component with the given name is found in the list.
        """
        names = [node.name for node in nodes]
        assert name in names, "Node not found in list"
        for node in nodes:
            if name == node.name:
                return node

    def __call__(self, nodes: list['Component']) -> 'Link':
        """
        Applies the link's transport operation.

        This is typically called after the graph execution, using the updated states
        of the components. It finds the transmitting node from the provided list
        of (potentially updated) nodes and applies the `transport_op`.

        Args:
            nodes: A list of Component objects, representing their state after an execution iteration.

        Returns:
            A new Link instance with potentially updated properties due to the transport operation.
        """
        tx_node = self.get_node_from_list(self.tx, nodes)
        new_properties = self.transport_op(self, tx_node)
        new_link = Link(self.name, self.tx, self.rx, self.transport_op, self.properties | new_properties)
        return new_link

class DefaultLink(Link):
    """A simple Link with a `DummyTransport` and no parameters."""
    def __init__(self, name, tx, rx):
        super().__init__(name, tx, rx, DummyTransport(), {})

# Execution graphs
def flatten(lst):
    """Flattens a list of arbitrary depth."""
    for item in lst:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten(item)
        else:
            yield item
    
"""
Represents a directed acyclic graph (DAG) of Components and Links,
defining a flow of execution for a specific task or process.
"""    
class ExecutionGraph(ABC):
    """
    Manages a graph of Components and Links, orchestrating their execution.

    Attributes:
        name (str): The name of the execution graph.
        nodes (list[Component]): The initial list of components in the graph.
        links (list[Link]): The initial list of links connecting the components.
        iteration (int): A counter for the number of times this graph has been executed.
        graph (nx.DiGraph): The underlying `networkx` directed graph.
        root (str): The name of the root (terminal) node of the graph.
        root_node (Component): The Component instance corresponding to the root node.
    """
    def __init__(self, name: str, nodes: list[Component], links: list[Link], metrics: list['Metric'] = [], iteration: int = 0):
        self.name = name
        self.nodes = nodes
        self.links = links
        self.metrics = metrics
        self.iteration = iteration

        nodes = [(n.name, {"ref": n,}) for n in nodes]
        edges = [(l.tx, l.rx, {"ref": l}) for l in links]
        self.graph = self.construct_graph(nodes, edges)
        self.root, self.root_node = self.identify_root()

    def construct_graph(self, nodes: list[Component], edges: list[Link]) -> nx.classes.digraph:
        """
        Builds the `networkx.DiGraph` from the list of nodes and links.

        Args:
            nodes: A list of tuples, where each tuple is (node_name, { 'ref': Component_instance }).
            edges: A list of tuples, where each tuple is (tx_node_name, rx_node_name, { 'ref': Link_instance }).

        Returns:
            A `networkx.DiGraph` instance.

        Raises:
            AssertionError: If the constructed graph is not a Directed Acyclic Graph (DAG).
        """
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.graph["name"] = self.name
        #check that it's acyclic
        assert nx.is_directed_acyclic_graph(g), "Graph must be a tree (acyclic), check definition"

        return g
    
    def get_node(self, name: str) -> Component:
        """
        Retrieves a Component instance from the graph by its name.

        Args:
            name: The name of the component to retrieve.

        Returns:
            The Component instance.
        """
        g = self.graph
        component = g.nodes[name]["ref"]
        return component
    
    def get_edge(self, tx: str, rx: str) -> Link:
        """
        Retrieves a Link instance from the graph by the names of its source and target nodes.

        Args:
            tx: The name of the transmitting (source) Component.
            rx: The name of the receiving (destination) Component.

        Returns:
            The Link instance.
        """
        g = self.graph
        link = g.edges[(tx, rx)]["ref"]
        return link

        
    def identify_root(self,):
        """
        Identifies the root (terminal) node(s) of the execution graph.
        Assumes a single root node for the current implementation.

        Returns:
            A tuple (root_node_name, root_Component_instance).

        Raises:
            AssertionError: If zero or more than one root node is identified.
        """
        od = list(self.graph.out_degree)
        roots = list(filter(lambda x: x[1] == 0, od))
        assert len(roots) == 1, "More than 1 root identified"
        root = roots[0][0]
        root_node = self.get_node(root)
        return root, root_node
    
    def get_predecessors(self, node: Component) -> list[Component]:
        """
        Retrieves the list of immediate predecessor (parent) Components for a given Component.

        Args:
            node: The Component whose predecessors are to be found.

        Returns:
            A list of Component instances that are predecessors to the given node.
        """
        up = list(self.graph.predecessors(node.name))
        components = [self.get_node(name) for name in up]
        return components
    
        
    def lean_copy(self):
        """
        Creates a 'lean' copy of the graph, containing only nodes and edges (names only),
        without any associated Component or Link objects or attributes. Useful for plotting.

        Returns:
            A new `networkx.DiGraph` with only node names and edge connections.
        """
        graph = self.graph
        lean_copy = nx.DiGraph()
        for n in list(graph.nodes):
            lean_copy.add_node(n)

        for e in list(graph.edges):
            lean_copy.add_edge(*e)

        return lean_copy
    
    def get_all_node_properties(self) -> dict[str, dict]:
        """
        Collects all properties from the nodes (Components) contained in self.nodes,
        returning them in a dictionary keyed by the node name.
        """
        all_properties = {}
        for component in self.nodes:
            # Ensuring the component has the expected attributes
            if hasattr(component, 'name') and hasattr(component, 'properties'):
                all_properties[component.name] = component.properties
            # else:
                # Optionally, log a warning or handle components that don't fit the expected structure
                # print(f"Warning: Component {getattr(component, 'name', 'Unnamed')} lacks 'properties'.")
        return all_properties

    def get_all_node_parameters(self) -> dict[str, dict]:
        """
        Collects all parameters from the nodes (Components) contained in self.nodes,
        returning them in a dictionary keyed by the node name.
        """
        all_parameters = {}
        for component in self.nodes:
            if hasattr(component, 'name') and hasattr(component, 'parameters'):
                all_parameters[component.name] = component.parameters
        return all_parameters
    
    def with_updated_parameters(self, new_parameters_map: dict[str, dict]) -> 'ExecutionGraph':
        """
        Creates a new ExecutionGraph with updated parameters for its components.

        The original ExecutionGraph instance remains unchanged. This method is useful
        for creating variations of a graph with different parameter configurations
        without modifying the original.

        Args:
            new_parameters_map: A dictionary where keys are component names (str)
                                and values are dictionaries of parameters (dict)
                                to update for that specific component.
                                For each component name found in this map:
                                 - Its existing parameters are taken as a base.
                                 - The parameters from new_parameters_map[component_name]
                                   are then merged in, overriding any existing parameters
                                   with the same key and adding new ones.
                                Components not listed in new_parameters_map retain their
                                original parameters.

        Returns:
            A new ExecutionGraph instance with the specified parameter updates.
            The iteration count of the new graph will be the same as the original
            graph from which it was derived, as this is a configuration change,
            not an execution step.
        """
        updated_components = []
        # self.nodes stores the original list of Component instances
        for original_component in self.nodes:
            component_name = original_component.name
            # Start with a copy of the component's current parameters
            updated_params = original_component.parameters.copy()

            if component_name in new_parameters_map:
                # Merge/update with the new parameters provided for this component
                updated_params.update(new_parameters_map[component_name])

            new_comp = Component(
                name=original_component.name,
                mutations=original_component.mutations,
                parameters=updated_params,
                properties=original_component.properties,  # Use initial properties
                merge=original_component.merge
            )
            updated_components.append(new_comp)

        return ExecutionGraph(
            name=self.name,
            nodes=updated_components,
            links=self.links,
            iteration=self.iteration
        )
    
    def __call__(self, verbose: bool = False) -> 'ExecutionGraph':
        """
        Executes the entire graph.

        This triggers a recursive execution starting from the root node.
        Each component processes its inputs and produces an output.
        After all components have executed, the links are also updated.

        Args:
            verbose: If True, enables verbose output during component execution.

        Returns:
            A new ExecutionGraph instance representing the state of the system after this iteration.
        """
        #Get the root processing node and propagate calls recursively up from it
        new_nodes = list(flatten(self.root_node(self, verbose)))
        #update the link information given the new nodes
        new_links = [link(new_nodes) for link in self.links]
        #calculate metrics
        metrics = [metric(self) for metric in self.metrics]
        #create the new execution graph
        exg = ExecutionGraph(self.name, new_nodes, new_links, self.metrics, self.iteration + 1)
        #store the metrics
        exg.metric_values = metrics
        return exg
    

class Metric(ABC):
    def __init__(self, name: str, msg_fields: list[str], msg_properties: list[str], host_properties: dict[str]):
        self.name = name
        self.msg_fields = msg_fields
        self.msg_properties = msg_properties
        self.host_properties = host_properties

    def _missing_keys(self, matches: list, values: list) -> str:
        """
        Helper function to generate a comma-separated string of missing items.

        Args:
            matches: A list of booleans indicating presence (True) or absence (False).
            values: A list of corresponding item names.

        Returns:
            A string listing the names of items where `matches` was False.
        """
        missing = []
        for (i,m) in enumerate(matches):
            if not m:
                missing.append(values[i])

        missing_fields = reduce(lambda x, y: x + ", " + y, missing)
        return missing_fields
    
    def _field_check(self, message: Message) -> None:
        """
        Checks if all required message fields are present in the incoming message.

        Args:
            message: The input Message object.

        Raises:
            AssertionError: If any of the fields specified in `self.msg_fields` are missing.
        """
        matches = [f in message.fields.keys() for f in self.msg_fields]
        field_chk = np.all(matches)
        if not field_chk:
            missing = self._missing_keys(matches, self.msg_fields)
            assert field_chk, "Input field for metric " + self.name + " not found in incoming message: " + missing

    def _msg_property_check(self, message: Message) -> None:
        """
        Checks if all required message properties are present in the incoming message.

        Args:
            message: The input Message object.

        Raises:
            AssertionError: If any of the properties specified in `self.msg_properties` are missing.
        """
        matches = [f in message.properties.keys() for f in self.msg_properties]
        props_chk = np.all(matches)
        if not props_chk:
            missing = self._missing_keys(matches, self.msg_properties)
            assert props_chk, self.name + " metric's properties not found in incoming message: " + missing
    
    def _host_property_check(self, all_properties: dict[str]) -> None:
        """
        Checks if all required host properties are present in the host ExecutionGraph.

        Args:
            component: The host Component object.

        Raises:
            AssertionError: If any of the properties specified in `self.host_properties` are missing.
        """
        for (k,v) in self.host_properties.items():
            assert k in all_properties.keys(), "Missing component" + str(k) + " from properties"
            props = all_properties[k]
            assert v in props.keys(), "Missing property " + str(v) + " from component " + str(k)
        

    def metric(self, graph: ExecutionGraph) -> dict[str]:
        """
        The core metric logic to be implemented by subclasses.

        This method should access data from the ExecutionGraph 'graph',
        utilizing the fields and properties in its root output message
        and all component parameters to produce one or more desired metrics
        to be stored in the ExecutionGraph.

        Args:
            graph: The host ExecutionGraph

        Returns:
            metrics: A dictionary of metrics to be stored in the host graph
        """
        metrics = {}
        return metrics

    def __call__(self, graph: ExecutionGraph):
        """
        Produces a metric from an ExecutionGraph.

        This method first performs checks for required fields, properties, and parameters.
        Then, it calls the `metric` method to get the new data and returns this data
        to be merged into its host ExecutionGraph.

        Args:
            component: The host ExecutionGraph that this metric is part of.

        Returns:
            metrics: A dictionary of metrics to be stored in the host graph
        """
        message = graph.root_node.output_msg
        #check that the graph output message has the field(s)/parameters necessary for the transform
        self._field_check(message)
        #check that the graph output message has the parameters necessary for the transform
        self._msg_property_check(message)
        #check that the host component has the parameters necessary for the transform
        self._host_property_check(graph.get_all_node_properties())

        metrics = self.metric(graph)
        return metrics


"""
Orchestrates a collection of ExecutionGraph instances.
"""
class System(ABC):
    """
    Represents a higher-level system composed of one or more ExecutionGraphs.

    When a System is called, it executes all of its contained ExecutionGraphs.

    Attributes:
        name (str): The name of the system.
        exec_graphs (list[ExecutionGraph]): A list of ExecutionGraphs managed by this system.
        iter (int): A counter for system-level iterations (currently not incremented in __call__).
    """
    def __init__(self, name: str, exec_graphs: list[ExecutionGraph], iter: int = 0):
        self.name = name
        self.exec_graphs = exec_graphs
        self.iter = iter

    def __call__(self) -> 'System':
        """
        Executes all ExecutionGraphs contained within this System.

        Returns:
            A new System instance with the updated states of its ExecutionGraphs.
            The iteration count of the new System is incremented (Note: original code had self.iter + 1,
            but self.iter was not an attribute, assuming it meant to track system iterations).
        """
        new_graphs = [graph() for graph in self.exec_graphs]
        new_system = System(self.name, new_graphs, getattr(self, 'iter', 0) + 1)
        return new_system
    
# Basic & example implementations

class Convolve(Mutate):
    """
    An example Mutate operation that simulates a convolution process on image data.
    This version is specific to the node.py file and might differ from notebook examples.
    """
    def __init__(self,):
        fields = ["image data"]
        properties = ["sensor"]
        parameters = ["kernel x", "kernel y", "filters"]
        super().__init__(fields, properties, parameters)

    def transform(self, component: Component, message: Message):
        """
        Performs a mock convolution operation.

        It calculates the number of operations based on sensor resolution and kernel parameters,
        and adds "features" to the message fields and "transform operations" to component properties.

        Args:
            component: The host Component (provides kernel parameters).
            message: The input Message (provides sensor resolution from its properties).

        Returns:
            A tuple (updated_message, new_component_properties).
        """
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
        # Assuming ureg.byte was from a units library, which is not imported here.
        # For this standalone file, we'll store it as a raw number.
        message.fields["features"] = np.prod((steps_x, steps_y, filters))

        return message, properties
       
    