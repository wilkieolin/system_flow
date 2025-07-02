# Mutations
import numpy as np
from abc import ABC, abstractmethod
from collections import namedtuple
from copy import deepcopy
from functools import reduce
import re

from systemflow.auxtypes import *
from systemflow.merges import *

MutationInputs = namedtuple("MutationInputs", ["msg_fields", "msg_properties", "host_parameters"])
MutationOutputs = namedtuple("MutationOutputs", ["msg_fields", "msg_properties", "host_properties"])

def collect_parameters(mutations: list['Mutate'], to_dict: bool = False) -> dict:
    host_params_dict = {}
    for mutation in mutations:
        for (k, v) in mutation.inputs.host_parameters.__dict__.items():
            host_params_dict[k] = v
    if to_dict:
        return host_params_dict
    else:
        collection = VarCollection(**host_params_dict)
        return collection


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
    def __init__(self, name: str, inputs: MutationInputs, outputs: MutationOutputs):
        # #Example:
        # #Input message fields
        # msg_fields = VarCollection()
    
        # #Input message properties
        # msg_properties = VarCollection()

        # #Input host parameters
        # host_parameters = VarCollection()
        
        # inputs = MutationInputs(msg_fields, msg_properties, host_parameters)

        # #Output message fields
        # msg_fields = VarCollection()

        # #Output message properties
        # msg_properties = VarCollection()

        # #Output host properties
        # host_properties = VarCollection()
        # outputs = MutationOutputs(msg_fields, msg_properties, host_properties)

        # super().__init__(name, inputs, outputs)
        
        self.name = name
        self.inputs = inputs
        self.outputs = outputs

    def _missing_keys(self, matches: list, values: VarCollection) -> str:
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
                x = list(values.__dict__.values())[i]
                if type(x) == Regex:
                    missing.append("Unit regex: " + x.str)
                else:
                    missing.append(x)

        missing_fields = reduce(lambda x, y: x + ", " + y, missing)
        return missing_fields
    
    def _get_matches(self, d: dict, vars: VarCollection) -> list:
        requests = vars.__dict__.values()

        matches = []
        for r in requests:
            if type(r) == Regex:
                match = True in [bool(re.search(r.str, f)) for f in d.keys()]
                matches.append(match)
            else:
                matches.append(r in d.keys())

        return matches
        

    def _field_check(self, message: Message) -> None:
        """
        Checks if all required message fields are present in the incoming message.

        Args:
            message: The input Message object.

        Raises:
            AssertionError: If any of the fields specified in `self.msg_fields` are missing.
        """
        matches = self._get_matches(message.fields, self.inputs.msg_fields)
        field_chk = np.all(matches)
        if not field_chk:
            missing = self._missing_keys(matches, self.inputs.msg_fields)
            assert field_chk, "Input field for transform " + self.name + " not found in incoming message: " + missing

    def _property_check(self, message: Message) -> None:
        """
        Checks if all required message properties are present in the incoming message.

        Args:
            message: The input Message object.

        Raises:
            AssertionError: If any of the properties specified in `self.msg_properties` are missing.
        """
        matches = self._get_matches(message.properties, self.inputs.msg_properties)
        props_chk = np.all(matches)
        if not props_chk:
            missing = self._missing_keys(matches, self.inputs.msg_properties)
            assert props_chk, self.name + " transform's properties not found in incoming message: " + missing
    
    def _param_check(self, component: 'Component') -> None:
        """
        Checks if all required host parameters are present in the host Component.

        Args:
            component: The host Component object.

        Raises:
            AssertionError: If any of the parameters specified in `self.host_parameters` are missing.
        """
        matches = self._get_matches(component.parameters, self.inputs.host_parameters)
        params_chk = np.all(matches)
        if not params_chk:
            missing = self._missing_keys(matches, self.inputs.host_parameters)
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
    