"""
Copyright 2025, UChicago Argonne LLC. 
Please refer to 'license' in the root directory for details and disclosures.
"""

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
from systemflow.node import *
from systemflow.metrics import serial_parallel_ops

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
        
class InputMessage(Mutate):
    """
    A mutation which can accept a message as a parameter - this can be used to link
    the output of one ExecutionGraph into the input of another.
    """
    def __init__(self, name: str = "InputMessage"):
        #Input message fields
        msg_fields = VarCollection()
    
        #Input message properties
        msg_properties = VarCollection()

        #Input host parameters
        host_parameters = VarCollection(input_message = "input message (Message)",)
        
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

        #create the new fields in the message
        input_message = component.parameters[self.inputs.host_parameters.input_message]
        msg_fields = {**input_message.fields}
        msg_props = {**input_message.properties}

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
        msg_properties = VarCollection(resolution = "resolution (n,n)",
                                       bitdepth = "bitdepth (n)",
                                       sample_rate = "sample rate (Hz)",
                                       images = "images (n)",)

        #Output host properties
        host_properties = VarCollection(sensor_power = "sensor power (W)",
                                        )

        outputs = MutationOutputs(msg_fields, msg_properties, host_properties)
        super().__init__(name, inputs, outputs)

    def transform(self, message: Message, component: 'Component') -> tuple[dict, dict, dict]:
        #access the requiredg fields/properties/parameters
        resolution = component.parameters[self.inputs.host_parameters.resolution] 
        bitdepth = component.parameters[self.inputs.host_parameters.bitdepth]
        n_bytes = np.prod(resolution) * bitdepth / 8.0
        latency = component.parameters[self.inputs.host_parameters.readout]
        sensor_power = component.parameters[self.inputs.host_parameters.pixelenergy] * np.prod(resolution)
        sample_rate = component.parameters[self.inputs.host_parameters.sample_rate]

        #create the new fields in the message

        msg_fields = {self.outputs.msg_fields.image_data: n_bytes,
                    self.outputs.msg_fields.readout: latency}
        
        msg_props = {self.outputs.msg_properties.resolution: resolution,
                     self.outputs.msg_properties.bitdepth: bitdepth,
                     self.outputs.msg_properties.sample_rate: sample_rate,
                     self.outputs.msg_properties.images: 1,}

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
                                      sample_rate = "sample rate (Hz)",
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
        sample_rate = component.parameters[self.inputs.host_parameters.sample_rate]
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
        msg_properties = VarCollection(resolution = "resolution (n,n)",
                                       sample_rate = "sample rate (Hz)",)

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
    
class FourierTransform2D(Mutate):
    """
    Models the operations and resources used during a forward or inverse (Fast) Fourier operation.
    """
    def __init__(self, name: str = "FFT"):
        #Input message fields
        #transform on any field with data (bytes - B)
        msg_fields = VarCollection()
    
        #Input message properties
        msg_properties = VarCollection(resolution = "resolution (n,n)",
                                       bitdepth = "bitdepth (n)",)

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
        resolution = message.properties[self.inputs.msg_properties.resolution]
        m = resolution[0]
        n = resolution[1] 
        bitdepth = message.properties[self.inputs.msg_properties.bitdepth]
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
    
class StoreImage(Mutate):
    """
    Accumulate images in a buffer written to disk
    """
    def __init__(self, name: str = "StoreImage"):
        #Input message fields
        msg_fields = VarCollection(image_data = "image data (B)",)
    
        #Input message properties
        msg_properties = VarCollection(images = "images (n)",)

        #Input host parameters
        host_parameters = VarCollection(storage_rate = "disk storage rate (B/s)",
                                        stored_data = "stored data (B)",
                                        stored_images = "stored images (n)",)
        
        inputs = MutationInputs(msg_fields, msg_properties, host_parameters)

        #Output message fields
        msg_fields = VarCollection(stored_data = "stored data (B)",)

        #Output message properties
        msg_properties = VarCollection(stored_images = "stored images (n)",)

        #Output host properties
        host_properties = VarCollection(storage_latency = "storage latency (s)",)
        
        outputs = MutationOutputs(msg_fields, msg_properties, host_properties)

        super().__init__(name, inputs, outputs)

    def transform(self, message: Message, component: 'Component') -> tuple[dict, dict, dict]:
        #access the required fields/properties/parameters
        images = message.properties[self.inputs.msg_properties.images]
        n_bytes = message.fields[self.inputs.msg_fields.image_data]

        storage_rate = component.parameters[self.inputs.host_parameters.storage_rate]
        stored_data = component.parameters[self.inputs.host_parameters.stored_data]
        stored_images = component.parameters[self.inputs.host_parameters.stored_images]

        #create the new fields in the message
        msg_fields = {self.outputs.msg_fields.stored_data: stored_data + n_bytes,}
        msg_props = {self.outputs.msg_properties.stored_images: stored_images + images,}

        #create the new properties in the host
        host_props = {self.outputs.host_properties.storage_latency: n_bytes / storage_rate,}

        return msg_fields, msg_props, host_props
    