
import numpy as np
from systemflow.node import *
from systemflow.mutations import *
from systemflow.metrics import *
from systemflow.auxtypes import *


class PositionSample(Mutate):
    def __init__(self, name: str = "PositionSample", relevancy_f: Callable = lambda x: 1.0): 
        #"Secret" sample function which determines which locations are of interest
        self.relevancy_f = relevancy_f

        #Originator node, no input fields
        #Input message fields
        msg_fields = VarCollection()
    
        #Input message properties
        msg_properties = VarCollection()

        #Input host parameters
        host_parameters = VarCollection(position = "position (mm,mm)",
                                        last_position = "last position (mm,mm)",
                                        move_rate = "move rate (mm/s)",
                                        settle_time = "settle time (s)",)
        
        inputs = MutationInputs(msg_fields, msg_properties, host_parameters)

        #Output message fields
        msg_fields = VarCollection(relevancy = "relevancy (%)",
                                   position = "position (mm,mm)",
                                   move_latency = "movement latency (s)",)

        #Output message properties
        msg_properties = VarCollection()

        #Output host properties
        host_properties = VarCollection(last_position = "last_position (mm,mm)",)
        outputs = MutationOutputs(msg_fields, msg_properties, host_properties)

        super().__init__(name, inputs, outputs)

    def transform(self, message: Message, component: Component) -> tuple[dict, dict, dict]:
        #predict the relevancy of the collected data based on position
        position = component.parameters[self.inputs.host_parameters.position]
        last_position = component.parameters[self.inputs.host_parameters.last_position]
        relevancy = self.relevancy_f(position)

        x_vec = position[0] - last_position[0]
        y_vec = position[1] - last_position[1]
        distance = np.linalg.norm([x_vec, y_vec])
        movement_time = distance / component.parameters[self.inputs.host_parameters.move_rate]
        latency = movement_time + component.parameters[self.inputs.host_parameters.settle_time]

        msg_fields = {self.outputs.msg_fields.relevancy: relevancy,
                    self.outputs.msg_fields.position: position,
                    self.outputs.msg_fields.move_latency: latency,}
        
        msg_props = {}
        
        host_props = {self.inputs.host_parameters.last_position: position,}

        
        return msg_fields, msg_props, host_props

class FlatFieldCorrection(Mutate):
    """
    Apply flat-field correction
    """
    def __init__(self, name: str = "Flat-field Correction"):
        #Input message fields
        msg_fields = VarCollection()
    
        #Input message properties
        msg_properties = VarCollection(resolution = "resolution (n,n,n)",)

        #Input host parameters
        host_parameters = VarCollection(op_latency = "op latency (s)",
                                        parallelism = "parallelism (%)",)
        
        inputs = MutationInputs(msg_fields, msg_properties, host_parameters)

        #Output message fields
        msg_fields = VarCollection(ff_latency = "flatfield latency (s)",)

        #Output message properties
        msg_properties = VarCollection()

        #Output host properties
        host_properties = VarCollection(ops = "flatfield ops (n,n)",)

        outputs = MutationOutputs(msg_fields, msg_properties, host_properties)

        super().__init__(name, inputs, outputs)

    def transform(self, message: Message, component: 'Component') -> tuple[dict, dict, dict]:
        #access the required fields/properties/parameters
        resolution = message.properties[self.inputs.msg_properties.resolution]
        ops = np.prod(resolution)
        parallelism = component.parameters[self.inputs.host_parameters.parallelism]
        serial_ops, parallel_ops = serial_parallel_ops(ops, parallelism)
        op_latency = component.parameters[self.inputs.host_parameters.op_latency]
        latency = serial_ops * op_latency
       
        #create the new fields in the message
        msg_fields = {self.outputs.msg_fields.ff_latency: latency}
        msg_props = {}

        #create the new properties in the host
        resolution = message.properties[self.inputs.msg_properties.resolution]
        host_props = {self.outputs.host_properties.ops: (serial_ops, parallel_ops),}

        return msg_fields, msg_props, host_props
    
class MaskCorrection(Mutate):
    """
    Correct pixel values in masked areas
    """
    def __init__(self, name: str = "Mask Correction"):
        #Input message fields
        msg_fields = VarCollection(image_data = "image data (B)")
    
        #Input message properties
        msg_properties = VarCollection(resolution = "resolution (n,n,n)",)

        #Input host parameters
        host_parameters = VarCollection(mask_proportion = "masking proportion (%)",
                                        op_latency = "op latency (s)",
                                        parallelism = "parallelism (%)",
                                        kernel_size = "kernel size (%,%)",)
        
        inputs = MutationInputs(msg_fields, msg_properties, host_parameters)

        #Output message fields
        msg_fields = VarCollection(mask_correction = "masking corrections (B)",
                                   mask_latency = "masking latency (s)",)

        #Output message properties
        msg_properties = VarCollection()

        #Output host properties
        host_properties = VarCollection(ops = "masking operations (n,n)",)
        outputs = MutationOutputs(msg_fields, msg_properties, host_properties)

        super().__init__(name, inputs, outputs)

    def transform(self, message: Message, component: 'Component') -> tuple[dict, dict, dict]:
        #access the required fields/properties/parameters
        image_data = message.fields[self.inputs.msg_fields.image_data]
        resolution = message.properties[self.inputs.msg_properties.resolution]
        resolution_x, resolution_y, _ = resolution
        
        masking = component.parameters[self.inputs.host_parameters.mask_proportion]
        op_latency = component.parameters[self.inputs.host_parameters.op_latency]
        parallelism = component.parameters[self.inputs.host_parameters.parallelism]
        kernel_x, kernel_y = component.parameters[self.inputs.host_parameters.kernel_size]

        is_proportion(masking)
        is_proportion(kernel_x)
        is_proportion(kernel_y)
        #calculate new fields and properties
        #use an X,Y kernel around every masked section (assumption)
        kernel_x = int(resolution_x * kernel_x)
        kernel_y = int(resolution_y * kernel_y)
        center_pixels = int(masking * np.prod(resolution))
        ops = kernel_x * kernel_y * center_pixels
        serial_ops, parallel_ops = serial_parallel_ops(ops, parallelism)
        latency = serial_ops * op_latency

        mask_data = image_data * masking
       
        #create the new fields in the message
        msg_fields = {self.outputs.msg_fields.mask_correction: mask_data,
                      self.outputs.msg_fields.mask_latency: latency,}
        msg_props = {}

        #create the new properties in the host
        host_props = {self.outputs.host_properties.ops: (serial_ops, parallel_ops),}

        return msg_fields, msg_props, host_props
    
class PhaseReconstruction(Mutate):
    """
    A blank template which can be used to define mutations
    """
    def __init__(self, name: str = "Phase reconstruction"):
        #Input message fields
        msg_fields = VarCollection(image_data = "image data (B)",)
    
        #Input message properties
        msg_properties = VarCollection(resolution = "resolution (n,n,n)",)

        #Input host parameters
        host_parameters = VarCollection(op_latency = "op latency (s)",
                                        parallelism = "parallelism (%)",
                                        iterations = "iterations (n)",)
        
        inputs = MutationInputs(msg_fields, msg_properties, host_parameters)

        #Output message fields
        msg_fields = VarCollection(phase_support = "phase data (B)",)

        #Output message properties
        msg_properties = VarCollection(phase = "phase reconstruction (m,n)",
                                       latency = "phase reconstruction latency (s)",)

        #Output host properties
        host_properties = VarCollection(ops = "phase reconstruction ops (n,n)",)
        outputs = MutationOutputs(msg_fields, msg_properties, host_properties)

        super().__init__(name, inputs, outputs)

    def transform(self, message: Message, component: 'Component') -> tuple[dict, dict, dict]:
        #access the required fields/properties/parameters
        #calculate the error between the predicted and 
        n, m, bitdepth = message.properties[self.inputs.msg_properties.resolution]
        parallelism = component.parameters[self.inputs.host_parameters.parallelism]
        op_latency = component.parameters[self.inputs.host_parameters.op_latency]
        iterations = component.parameters[self.inputs.host_parameters.iterations]

        # AD of the error term has to backtrace through the fwd FFT op
        fft_ops = n * m * np.log10(m) + m * n * np.log10(n)
        rvs_ops = n * m
        total_ops = (fft_ops + rvs_ops) * iterations
        support_data = n * m * bitdepth
        serial_ops, parallel_ops = serial_parallel_ops(total_ops, parallelism)
        latency = op_latency * serial_ops
        

        msg_fields = {self.outputs.msg_fields.phase_support: support_data,}
        
        msg_props = {self.outputs.msg_properties.phase: (m,n),
                     self.outputs.msg_properties.latency: latency,}

        host_props = {self.outputs.host_properties.ops: (serial_ops, parallel_ops),}

        return msg_fields, msg_props, host_props