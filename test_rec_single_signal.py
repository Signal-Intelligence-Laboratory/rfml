import numpy as np

from rfml.data.converters.rec_single_signal import SingleSignalDataLoader

x = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]])

sig = SingleSignalDataLoader()

sig.partition(x, 4)









print(x)

