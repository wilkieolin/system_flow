from collections import namedtuple

# Message handling
#: A named tuple representing a message passed between components.
#:
#: Attributes:
#:  fields (dict): A dictionary containing the data which varies by sample (e.g. fluence, temperature)
#:  properties (dict): A dictionary containing sample-independent data (e.g. resolution, sample rate)
Message = namedtuple("Message", ["fields", "properties"])

#Collection to hold named variables as subfields for convenience
class VarCollection:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class Regex:
    def __init__(self, str):
        self.str = str