# Mutations
import numpy as np
from abc import ABC, abstractmethod
from collections import namedtuple
from copy import deepcopy
from functools import reduce
import re

from systemflow.auxtypes import *
from systemflow.merges import *
from systemflow.classifier import *
from systemflow.metrics import serial_parallel_ops

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
    
# Basic & example implementations

class DummyMutate(Mutate):
    """
    A blank template which can be used to define mutations
    """
    def __init__(self, name: str = "DummyMutate"):
        #Input message fields
        msg_fields = VarCollection()
    
        #Input message properties
        msg_properties = VarCollection()

        #Input host parameters
        host_parameters = VarCollection()
        
        inputs = MutationInputs(msg_fields, msg_properties, host_parameters)

        #Output message fields
        msg_fields = VarCollection()

        #Output message properties
        msg_properties = VarCollection()

        #Output host properties
        host_properties = VarCollection()
        outputs = MutationOutputs(msg_fields, msg_properties, host_properties)

        super().__init__(name, inputs, outputs)

    def transform(self, message: Message, component: 'Component') -> tuple[dict, dict, dict]:
        #access the required fields/properties/parameters

        #calculate new fields and properties
       
        #create the new fields in the message
        msg_fields = {}
        msg_props = {}

        #create the new properties in the host
        host_props = {}

        return msg_fields, msg_props, host_props
        

class CollectImage(Mutate):
    """
    A mutation which models collecting an image from a 2-D Pixel sensor.
    """
    def __init__(self, name: str = "CollectImage"):
        #Input message fields
        #(None)
        msg_fields = VarCollection()
    
        #Input message properties
        #(None)
        msg_fields = VarCollection()

        #Input host parameters
        host_parameters = VarCollection(resolution = "resolution (n,n)",
                                    bitdepth = "bit depth (n)",
                                    readout = "readout latency (s)",
                                    pixelenergy = "pixel energy (J)",
                                    sample_rate = "sample rate (Hz)",)
        
        inputs = MutationInputs(msg_fields, msg_fields, host_parameters)

        #Output message fields
        msg_fields = VarCollection(image_data = "image data (B)",
                                   readout = "readout latency (s)",)

        #Output message properties
        msg_properties = VarCollection(resolution = "resolution (n,n,n)",
                                       sample_rate = "sample rate (Hz)",)

        #Output host properties
        host_properties = VarCollection(sensor_power = "sensor power (W)",)

        outputs = MutationOutputs(msg_fields, msg_properties, host_properties)
        super().__init__(name, inputs, outputs)

    def transform(self, message: Message, component: 'Component') -> tuple[dict, dict, dict]:
        #access the requiredg fields/properties/parameters
        (n_px_x, n_px_y) = component.parameters[self.inputs.host_parameters.resolution] 
        resolution = (n_px_x,
                      n_px_y,
                      component.parameters[self.inputs.host_parameters.bitdepth],)
        n_bytes = np.prod(resolution) / 8.0
        latency = component.parameters[self.inputs.host_parameters.readout]
        sensor_power = component.parameters[self.inputs.host_parameters.pixelenergy] * n_px_x * n_px_y
        sample_rate = component.parameters[self.inputs.host_parameters.sample_rate]

        #create the new fields in the message

        msg_fields = {self.outputs.msg_fields.image_data: n_bytes,
                    self.outputs.msg_fields.readout: latency}
        
        msg_props = {self.outputs.msg_properties.resolution: resolution,
                     self.outputs.msg_properties.sample_rate: sample_rate}

        #create the new properties in the host
        host_props = {self.outputs.host_properties.sensor_power: sensor_power,}

        return msg_fields, msg_props, host_props
       
class CollectTemperature(Mutate):
    """
    A mutation which models collecting a single value from a point (0-D) temperature sensor.
    """

    def __init__(self, name: str = "CollectTemperature"):
        #Example:
        #Input message fields
        msg_fields = VarCollection()
    
        #Input message properties
        msg_fields = VarCollection()

        #Input host parameters
        host_parameters = VarCollection(bitdepth = "temperature bitdepth (n)",
                                      samplerate = "sample rate (Hz)",
                                      sensor_power = "thermocouple power (W)",)
        
        inputs = MutationInputs(msg_fields, msg_fields, host_parameters)

        #Output message fields
        msg_fields = VarCollection(data = "temperature data (B)",
                                    time = "time (s)",)

        #Output message properties
        msg_properties = VarCollection(sample_rate = "sample rate (Hz)",)

        #Output host properties
        host_properties = VarCollection(sensor_power = "thermocouple power (W)")

        outputs = MutationOutputs(msg_fields, msg_properties, host_properties)

        super().__init__(name, inputs, outputs)

    def transform(self, message: Message, component: 'Component') -> tuple[dict, dict, dict]:
        #access the required fields/properties/parameters
        n_bytes = component.parameters[self.inputs.host_parameters.bitdepth] / 8.0
        sample_rate = component.parameters[self.inputs.host_parameters.samplerate]
        time = 0.0
        sensor_power = component.parameters[self.inputs.host_parameters.sensor_power]

        #create the new fields in the message
        msg_fields = {self.outputs.msg_fields.data: n_bytes,
                    self.outputs.msg_fields.time: time,}
        
        msg_props = {self.outputs.msg_properties.sample_rate: sample_rate,}

        #create the new properties in the host
        host_props = {self.outputs.host_properties.sensor_power: sensor_power,}

        return msg_fields, msg_props, host_props
       

class Convolve(Mutate):
    """
    Models the operations and resources used during a convolution operation.
    """
    def __init__(self, name: str = "Convolve"):
        #Input message fields
        msg_fields = VarCollection()
    
        #Input message properties
        msg_properties = VarCollection(resolution = "resolution (n,n,n)",)

        #Input host parameters
        host_parameters = VarCollection(kernel = "kernel (n,n)",
                                        filters = "filters (n)",)
        
        inputs = MutationInputs(msg_fields, msg_properties, host_parameters)

        #Output message fields
        msg_fields = VarCollection(features = "features (B)",)

        #Output message properties
        msg_properties = VarCollection()

        #Output host properties
        host_properties = VarCollection(ops = "conv ops (n)",)

        outputs = MutationOutputs(msg_fields, msg_fields, host_properties)

        super().__init__(name, inputs, outputs)


    def transform(self, message: Message, component: 'Component'):
        """Previously, we defined the inputs and outputs which this mutation has. Now, we create concrete
           transforms which take those inputs and set the outputs"""
        #access the required fields/properties/parameters
        res = message.properties[self.inputs.msg_properties.resolution]
        kernel_x = component.parameters[self.inputs.host_parameters.kernel][0]
        kernel_y = component.parameters[self.inputs.host_parameters.kernel][1]
        filters = component.parameters[self.inputs.host_parameters.filters]
        rate = message.properties[self.inputs.msg_properties.sample_rate]

        #calculate the number of ops required for the kernel
        kernel_ops = kernel_x * kernel_y * filters
        steps_x = (res[0] - kernel_x) // kernel_x
        steps_y = (res[1] - kernel_y) // kernel_y
        kernel_repeats = steps_x * steps_y

        #calculate the number of ops required for the kernel
        transform_operations = kernel_ops * kernel_repeats * rate

        """Above, we access the properties from the input message and host component required by the mutation,
           and calculate the outputs. These are stored in dictionaries and accessed by the host component to update
           its own properties and output message:"""
        msg_fields = {self.outputs.msg_fields.features: np.prod((steps_x, steps_y, filters)),}
        msg_properties = {}
        component_properties = {self.outputs.host_properties.ops: transform_operations,}

        return msg_fields, msg_properties, component_properties
    
class FourierTransform(Mutate):
    """
    Models the operations and resources used during a forward or inverse (Fast) Fourier operation.
    """
    def __init__(self, name: str = "FFT"):
        #Input message fields
        #transform on any field with data (bytes - B)
        msg_fields = VarCollection()
    
        #Input message properties
        msg_properties = VarCollection(resolution = "resolution (n,n,n)",)

        #Input host parameters
        host_parameters = VarCollection(parallelism = "parallelism (%)",
                                        op_latency = "op latency (s)",)
        
        inputs = MutationInputs(msg_fields, msg_properties, host_parameters)

        #Output message fields
        msg_fields = VarCollection(fft_data = "frequency data (B)",
                                   fft_latency = "fft latency (s)",)

        #Output message properties
        msg_properties = VarCollection(fft = "frequencies (n,n)",)

        #Output host properties
        host_properties = VarCollection(ops = "fft ops (n,n)",)

        outputs = MutationOutputs(msg_fields, msg_properties, host_properties)

        super().__init__(name, inputs, outputs)


    def transform(self, message: Message, component: 'Component'):
        #access the required fields/properties/parameters
        n, m, bitdepth = message.properties[self.inputs.msg_properties.resolution]
        parallelism = component.parameters[self.inputs.host_parameters.parallelism]
        op_latency = component.parameters[self.inputs.host_parameters.op_latency]

        fft_ops = n * m * np.log10(m) + m * n * np.log10(n)
        freq_data = n * m * bitdepth
        serial_ops, parallel_ops = serial_parallel_ops(fft_ops, parallelism)
        latency = op_latency * serial_ops

        msg_fields = {self.outputs.msg_fields.fft_data: freq_data,
                      self.outputs.msg_fields.fft_latency: latency,}
        msg_properties = {self.outputs.msg_properties.fft: (m,n)}
        msg_properties = {}
        component_properties = {self.outputs.host_properties.ops: (serial_ops, parallel_ops),}
        

        return msg_fields, msg_properties, component_properties
       

class GaussianClassify(Mutate):
    def __init__(self, name: str = "GaussianClassify"):
        #Input message fields
        msg_fields = VarCollection()
    
        #Input message properties
        msg_properties = VarCollection(sample_rate = "sample rate (Hz)",)

        #Input host parameters
        host_parameters = VarCollection(skill = "skill (1)",
                                        variance = "variance (1)",
                                        reduction = "reduction (%)",)
        
        inputs = MutationInputs(msg_fields, msg_properties, host_parameters)

        #Output message fields
        msg_fields = VarCollection(contingency = "contingency (2x2)",)

        #Output message properties
        msg_properties = VarCollection(error_matrix = "error matrix (2p, 2p)",)

        #Output host properties
        host_properties = VarCollection()
        outputs = MutationOutputs(msg_fields, msg_properties, host_properties)

        super().__init__(name, inputs, outputs)

    def transform(self, message: Message, component: 'Component'):
        #access the required fields/properties/parameters
        sample_rate = message.properties[self.inputs.msg_properties.sample_rate]
        skill = component.parameters[self.inputs.host_parameters.skill]
        variance = component.parameters[self.inputs.host_parameters.variance]
        reduction = component.parameters[self.inputs.host_parameters.reduction]

        #calculate the error statistics
        falses = sample_rate * reduction
        trues = sample_rate - falses
        inputs = np.array([falses, trues])
        
        gc = GaussianClassifier(skill, varscale=variance)
        output = gc(inputs, reduction)

        msg_fields = {self.outputs.msg_fields.contingency: output}
        msg_properties = {self.outputs.msg_properties.error_matrix: gc.error_matrix}
        component_properties = {}

        return msg_fields, msg_properties, component_properties
    
class ClassifiedStorageRate(Mutate):
    def __init__(self, name: str = "ClassifiedStorageRate"):
        #Input message fields
        msg_fields = VarCollection(total_data = "total data (B)",)
    
        #Input message properties
        msg_properties = VarCollection(error_matrix = "error matrix (2p, 2p)",)

        #Input host parameters
        host_parameters = VarCollection()
        
        inputs = MutationInputs(msg_fields, msg_properties, host_parameters)

        #Output message fields
        msg_fields = VarCollection()

        #Output message properties
        msg_properties = VarCollection()

        #Output host properties
        host_properties = VarCollection(storage_rate = "storage rate (B/s)",)
        outputs = MutationOutputs(msg_fields, msg_properties, host_properties)

        super().__init__(name, inputs, outputs)

    def transform(self, message: Message, component: 'Component'):
        storage_rate = message.fields[self.inputs.msg_fields.total_data] * get_passed(message.properties[self.inputs.msg_properties.error_matrix])
        new_host_properties = {self.outputs.host_properties.storage_rate: storage_rate,}
        return {}, {}, new_host_properties
    
class DataRate(Mutate):
    def __init__(self, name: str = "DataRate"):
        #Input message fields
        #transform on any field with data (bytes - B)
        msg_fields = VarCollection(bytes = Regex(r"\(B\)"),)
    
        #Input message properties
        msg_properties = VarCollection()

        #Input host parameters
        host_parameters = VarCollection()
        
        inputs = MutationInputs(msg_fields, msg_properties, host_parameters)

        #Output message fields
        msg_fields = VarCollection()

        #Output message properties
        msg_properties = VarCollection()

        #Output host properties
        host_properties = VarCollection(total_data = "total data (B)",)

        outputs = MutationOutputs(msg_fields, msg_properties, host_properties)

        super().__init__(name, inputs, outputs)

    def transform(self, message: Message, component: 'Component'):
        total_data = 0.0
        data_unit = self.inputs.msg_fields.bytes.str
        for (key, value) in message.fields.items():
            if bool(re.search(data_unit, key)) and key != "total data":
                total_data += value

        new_msg_fields = {self.outputs.host_properties.total_data: total_data}
        return new_msg_fields, {}, {}
    
class StorageRate(Mutate):
    def __init__(self, name: str = "StorageRate"):
        #Input message fields
        #transform on any field with data (bytes - B)
        msg_fields = VarCollection(total_data = "total data (B)",)
    
        #Input message properties
        msg_properties = VarCollection(sample_rate = "sample rate (Hz)",)

        #Input host parameters
        host_parameters = VarCollection()
        
        inputs = MutationInputs(msg_fields, msg_properties, host_parameters)

        #Output message fields
        msg_fields = VarCollection()

        #Output message properties
        msg_properties = VarCollection()

        #Output host properties
        host_properties = VarCollection(storage_rate = "data rate (B/s)",)

        outputs = MutationOutputs(msg_fields, msg_fields, host_properties)

        super().__init__(name, inputs, outputs)

    def transform(self, message: Message, component: 'Component'):
        storage_rate = message.fields[self.inputs.msg_fields.total_data] * message.properties[self.inputs.msg_properties.sample_rate]
        new_host_properties = {self.outputs.host_properties.storage_rate: storage_rate}
        return {}, {}, new_host_properties