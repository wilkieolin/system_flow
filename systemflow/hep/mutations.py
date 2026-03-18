"""
HEP-specific mutations, merge strategy, and link transport for modeling
CMS-like data acquisition systems using the SystemFlow v2.0 framework.

These mutations replicate the computations from systemflow.graph (v1.0)
in the declarative v2.0 Mutate/Component/ExecutionGraph pattern.
"""

import numpy as np
from functools import reduce

from systemflow.auxtypes import *
from systemflow.merges import *
from systemflow.node import *
from systemflow.classifier import DummyClassifier, ratio_to_reduction, get_passed, get_rejected


class HEPDetector(Mutate):
    """
    Originator mutation for HEP detector nodes.

    Replaces the detector initialization in graph.py's detectors() and the
    detector branch of propagate_statistics() and message_size().

    Produces:
      - Output data (post-compression) for downstream accumulation
      - Initial classification split (passed events) based on global ratio
      - Message size, ops, energy, power as host properties
    """
    def __init__(self, name: str = "HEPDetector"):
        msg_fields = VarCollection()
        msg_properties = VarCollection()

        host_parameters = VarCollection(
            sample_data="sample data (B)",
            sample_rate="sample rate (Hz)",
            data_reduction="data reduction (%)",
            op_efficiency="op efficiency (J/op)",
            routing_latency="routing latency (s)",
            global_ratio="global ratio (1)",
            complexity="complexity (fn)",
        )

        inputs = MutationInputs(msg_fields, msg_properties, host_parameters)

        out_msg_fields = VarCollection(
            output_data="output data (B)",
            passed_events="passed events (2)",
        )
        out_msg_properties = VarCollection(
            message_size="message size (B)",
            output_rate="output rate (Hz)",
        )
        out_host_properties = VarCollection(
            message_size="message size (B)",
            ops="ops (n)",
            energy="energy (J)",
            power="power (W)",
            input_rate="input rate (Hz)",
            output_rate="output rate (Hz)",
            contingency="contingency (2x2)",
            discards="discards (2)",
            error_matrix="error matrix (2x2)",
            classifier_active="classifier active",
            routing_latency="routing latency (s)",
        )

        outputs = MutationOutputs(out_msg_fields, out_msg_properties, out_host_properties)
        super().__init__(name, inputs, outputs)

    def transform(self, message: Message, component: 'Component') -> tuple[dict, dict, dict]:
        p = component.parameters
        hp = self.inputs.host_parameters

        sample_data = p[hp.sample_data]
        sample_rate = p[hp.sample_rate]
        data_reduction = p[hp.data_reduction]
        op_efficiency = p[hp.op_efficiency]
        routing_latency = p[hp.routing_latency]
        global_ratio = p[hp.global_ratio]
        complexity_fn = p[hp.complexity]

        # Message size and ops (replicates message_size() from graph.py)
        msg_size = sample_data
        ops = complexity_fn(msg_size)
        output_data = msg_size * data_reduction

        # Classification split (replicates propagate_statistics() detector branch)
        positives = sample_rate / global_ratio
        negatives = sample_rate - positives
        passed_events = np.array([negatives, positives]).astype("int")
        contingency = np.array([[0, 0], [negatives, positives]]).astype("int")

        # Power (replicates op_power() from graph.py)
        energy = op_efficiency * ops
        power = energy * sample_rate

        of = self.outputs.msg_fields
        op = self.outputs.msg_properties
        oh = self.outputs.host_properties

        msg_fields = {
            of.output_data: output_data,
            of.passed_events: passed_events,
        }
        msg_props = {
            op.message_size: msg_size,
            op.output_rate: sample_rate,
        }
        host_props = {
            oh.message_size: msg_size,
            oh.ops: ops,
            oh.energy: energy,
            oh.power: power,
            oh.input_rate: sample_rate,
            oh.output_rate: sample_rate,
            oh.contingency: contingency,
            oh.discards: get_rejected(contingency),
            oh.error_matrix: np.array([[0.0, 0.0], [1.0, 1.0]]),
            oh.classifier_active: False,
            oh.routing_latency: routing_latency,
        }

        return msg_fields, msg_props, host_props


class HEPProcessor(Mutate):
    """
    Processing/trigger mutation for HEP processor nodes.

    Replaces the processor branch of propagate_statistics(), message_size(),
    and op_power() from graph.py.

    Handles data accumulation, complexity-based ops computation, classifier
    application, and power calculation in a single mutation.
    """
    def __init__(self, name: str = "HEPProcessor"):
        msg_fields = VarCollection(
            output_data="output data (B)",
            passed_events="passed events (2)",
        )
        msg_properties = VarCollection()

        host_parameters = VarCollection(
            sample_data="sample data (B)",
            data_reduction="data reduction (%)",
            op_efficiency="op efficiency (J/op)",
            routing_latency="routing latency (s)",
            reduction_ratio="reduction ratio (1)",
            classifier="classifier (obj)",
            complexity="complexity (fn)",
        )

        inputs = MutationInputs(msg_fields, msg_properties, host_parameters)

        out_msg_fields = VarCollection(
            output_data="output data (B)",
            passed_events="passed events (2)",
        )
        out_msg_properties = VarCollection(
            message_size="message size (B)",
            output_rate="output rate (Hz)",
        )
        out_host_properties = VarCollection(
            message_size="message size (B)",
            ops="ops (n)",
            energy="energy (J)",
            power="power (W)",
            input_rate="input rate (Hz)",
            output_rate="output rate (Hz)",
            contingency="contingency (2x2)",
            discards="discards (2)",
            error_matrix="error matrix (2x2)",
            classifier_active="classifier active",
            routing_latency="routing latency (s)",
        )

        outputs = MutationOutputs(out_msg_fields, out_msg_properties, out_host_properties)
        super().__init__(name, inputs, outputs)

    def transform(self, message: Message, component: 'Component') -> tuple[dict, dict, dict]:
        p = component.parameters
        hp = self.inputs.host_parameters

        sample_data = p[hp.sample_data]
        data_reduction = p[hp.data_reduction]
        op_efficiency = p[hp.op_efficiency]
        routing_latency = p[hp.routing_latency]
        reduction_ratio = p[hp.reduction_ratio]
        classifier = p[hp.classifier]
        complexity_fn = p[hp.complexity]

        inf = self.inputs.msg_fields

        # Data accumulation (replicates message_size() from graph.py)
        input_data = message.fields[inf.output_data]
        msg_size = input_data + sample_data
        ops = complexity_fn(msg_size)
        output_data = msg_size * data_reduction

        # Classification (replicates propagate_statistics() processor branch)
        passed_events = message.fields[inf.passed_events]
        input_rate = np.sum(passed_events)

        reduction = ratio_to_reduction(reduction_ratio)
        contingency = classifier(passed_events, reduction)
        error_matrix = classifier.error_matrix
        discards = get_rejected(contingency)
        output_events = get_passed(contingency)
        output_rate = np.sum(output_events)

        # Power (replicates op_power() from graph.py)
        energy = op_efficiency * ops
        power = energy * input_rate

        is_active = not isinstance(classifier, DummyClassifier)

        of = self.outputs.msg_fields
        op = self.outputs.msg_properties
        oh = self.outputs.host_properties

        msg_fields = {
            of.output_data: output_data,
            of.passed_events: output_events,
        }
        msg_props = {
            op.message_size: msg_size,
            op.output_rate: output_rate,
        }
        host_props = {
            oh.message_size: msg_size,
            oh.ops: ops,
            oh.energy: energy,
            oh.power: power,
            oh.input_rate: input_rate,
            oh.output_rate: output_rate,
            oh.contingency: contingency,
            oh.discards: discards,
            oh.error_matrix: error_matrix,
            oh.classifier_active: is_active,
            oh.routing_latency: routing_latency,
        }

        return msg_fields, msg_props, host_props


class HEPMerge(Merge):
    """
    Merge strategy for HEP pipeline nodes with multiple predecessors.

    - "output data (B)": summed across predecessors (total data arriving)
    - "passed events (2)": averaged across predecessors (collated event stream)

    This replicates the behavior in graph.py where message_size() sums data
    and propagate_statistics() averages the passed event vectors.
    """
    def __init__(self):
        field_merges = {
            "output data (B)": sum,
            "passed events (2)": lambda vals: (reduce(lambda x, y: x + y, vals) / len(vals)).astype("int"),
        }
        # Properties are overwritten by each processor's own mutation, so
        # just take the first value to suppress merge warnings
        property_merges = {
            "message size (B)": lambda vals: vals[0],
            "output rate (Hz)": lambda vals: vals[0],
        }
        super().__init__(field_merges, property_merges)


class HEPLinkTransport(Transport):
    """
    Transport operation for HEP links computing throughput and power.

    Replicates link_throughput() and link_power() from graph.py.
    Reads the transmitting node's message size and output rate from
    the message properties, and link efficiency from link parameters.
    """
    def __init__(self):
        super().__init__(
            msg_fields=[],
            msg_properties=["message size (B)", "output rate (Hz)"],
            parameters=["link efficiency (J/bit)"],
        )

    def transport(self, link: 'Link', message: Message) -> dict:
        msg_size = message.properties["message size (B)"]
        output_rate = message.properties["output rate (Hz)"]
        link_efficiency = link.parameters["link efficiency (J/bit)"]

        # Replicates link_throughput() from graph.py
        throughput = msg_size * output_rate

        # Replicates link_power() from graph.py (bytes to bits conversion)
        energy = link_efficiency * msg_size * 8
        power = link_efficiency * throughput * 8

        return {
            "message size (B)": msg_size,
            "throughput (B/s)": throughput,
            "energy (J)": energy,
            "power (W)": power,
        }
