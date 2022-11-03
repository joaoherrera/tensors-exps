import json
import os
import itertools


class CocoAnnotations:
    """Handle coco-like annotations."""
    
    def __init__(self, filepath: str) -> None:
        if not os.path.isfile(filepath):
            raise ValueError(f"Path {filepath} is not a file.")

        self.filepath = filepath
        self.load()
        
    def load(self, inplace: bool = True) -> None | dict:
        with open(self.filepath) as coco_file:
            data = json.load(coco_file)

            if inplace:
                self.data = data
            else: 
                return data
    
    @staticmethod
    def to_dict(data: list, key_type: str) -> dict:
        if not key_type in data[0].keys():
            raise ValueError("Invalid key")
        
        data_dictionary = {}
        for key, group in itertools.groupby(data, key_type):
            data_dictionary[key] = group
        
        return data_dictionary