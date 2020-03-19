
import json
from collections import defaultdict
from pathlib import Path


class PersistentDefaultDict:

    """
    Nested defaultdict that gets synced transparently to disk.

    Init: 
        results = PersistentDefaultDict(<path_to_results_file>)

    Add result:
        results['key1', 'key2'] = <result>
    """    

    def __init__(self, path_to_dict):
        self.path = Path(path_to_dict)
        if self.path.is_file():
            stored_data = json.loads(self.path.read_text())
            self.data = PersistentDefaultDict.redefault_dict(stored_data)
        else:
            self.data = defaultdict(PersistentDefaultDict.rec_default_dict)
            
    def __str__(self):
        return str(json.dumps(self.data, indent=4))
            
    def __setitem__(self, keys, item):
        d = self.data
        if isinstance(keys, str):
            d[keys] = item
        elif isinstance(keys, tuple):
            for key in keys[:-1]:
                d = d[key]
            d[keys[-1]] = item
        else:
            raise NotImplementedError()
        self.path.write_text(json.dumps(self.data, indent=4))

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return self.data.__iter__()

    def as_dict(self):
        if self.path.is_file():
            return json.loads(self.path.read_text())
        else:
            return {}

    @staticmethod
    def rec_default_dict():
        return defaultdict(PersistentDefaultDict.rec_default_dict)
        
    @staticmethod
    def redefault_dict(data):
        if isinstance(data, dict):
            return defaultdict(PersistentDefaultDict.rec_default_dict, {k: PersistentDefaultDict.redefault_dict(v) for k, v in data.items()})
        else:
            return data
