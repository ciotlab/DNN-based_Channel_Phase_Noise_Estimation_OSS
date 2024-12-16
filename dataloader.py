import numpy as np
from pathlib import Path


class SampleDataLoader(object):
    def __init__(self, file_name, num_repeat=1):
        path = Path(__file__).parents[0].resolve() / 'sample_data' / (file_name + '.npy')
        self.sample_data = np.load(path)
        self.len = self.sample_data.shape[0]
        self.idx = 0
        self.num_repeat = num_repeat
        self.repeat_idx = 0

    def __len__(self):
        return self.len * self.num_repeat

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.len:
            self.repeat_idx += 1
            if self.repeat_idx >= self.num_repeat:
                raise StopIteration
            else:
                self.idx = 0
        data = self.sample_data[self.idx]
        self.idx += 1
        return data