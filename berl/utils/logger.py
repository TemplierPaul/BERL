import matplotlib.pyplot as plt
try:
    from IPython.display import clear_output
except:
    pass


class Logger:
    def __init__(self):
        self._data = {}

    def add(self, name):
        self._data[name] = []

    def add_list(self, name_list):
        for i in name_list:
            self.add(i)

    def log(self, name, value):
        if name not in self._data.keys():
            self.add(name)
        self._data[name].append(value)
#         print("logged", name, value)
        return self

    def log_dict(self, d):
        for k, v in d.items():
            self.log(k, v)
        return self

    def __repr__(self):
        s = "Logger"
        for k, v in self._data.items():
            s += f"\n - {k} ({len(v)})"
        return s

    def __call__(self, *args):
        if type(args[0]) == dict:
            self.log_dict(args[0])
        else:
            self.log(args[0], args[1])

    def __getitem__(self, key):
        return self._data[key]

    def export(self):
        return {k: v[-1] for k, v in self._data.items() if len(v) > 0}

    def last(self, key):
        return self[key][-1] if len(self[key]) > 0 else None
