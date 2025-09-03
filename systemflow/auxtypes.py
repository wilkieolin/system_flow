"""
Copyright 2025, UChicago Argonne LLC. 
Please refer to 'license' in the root directory for details and disclosures.
"""

from collections import namedtuple

# Message handling
#: A named tuple representing a message passed between components.
#:
#: Attributes:
#:  fields (dict): A dictionary containing the data which varies by sample (e.g. fluence, temperature)
#:  properties (dict): A dictionary containing sample-independent data (e.g. resolution, sample rate)
Message = namedtuple("Message", ["fields", "properties"])

empty_message = lambda: Message({}, {})

def merge_message(message: Message, new_fields: dict, new_properties: dict):
    new_message = Message(message.fields | new_fields, message.properties | new_properties)
    return new_message

#Collection to hold named variables as subfields for convenience
class VarCollection:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        if not self.__dict__:
            return f"{self.__class__.__name__}()"

        items_str = ",\n    ".join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}(\n    {items_str}\n)"

class Regex:
    def __init__(self, str):
        self.str = str

def is_proportion(value: float):
    assert 0.0 <= value <= 1.0, "Must be on the domain [0.0, 1.0]"
    