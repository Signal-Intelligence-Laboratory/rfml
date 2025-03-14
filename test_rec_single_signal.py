import numpy as np

from rfml.data.converters.rec_single_signal import SingleSignalDataLoader

arr = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]])

sig = SingleSignalDataLoader()

arr = sig.partition(arr, 7)

print(arr)


