"""
Copyright 2025, UChicago Argonne LLC. 
Please refer to 'license' in the root directory for details and disclosures.
"""

import numpy as np
from abc import ABC, abstractmethod
from collections import namedtuple
from copy import deepcopy
from functools import reduce
from typing import Callable, Any

from systemflow.auxtypes import *

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