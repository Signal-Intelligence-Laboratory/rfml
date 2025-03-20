# Plotting Includes
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

import numpy as np
import sys

sys.path.insert(0, "/home/garrett/Code/rfml")

from rfml.nbutils import plot_IQ
from rfml.data.converters.rec_urh_single_signal import SingleSignalDataLoader

sig = SingleSignalDataLoader()


# TODO: check that the load() function from SingleSignalDataLoader can load from any of the formats properly



# TEST WITH A SET ARRAY
# arr = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
#               [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]])

# arr = sig.partition(arr, 8)


# Graph the IQ data
# title = "Graph"
# fig = plot_IQ(iq=arr[0], title=title)
# plt.show()


# TEST WITH A .COMPLEX FILE
file = sig.load("/home/garrett/PlutoSDR-20250304_142236-88_9MHz-2_1MSps-2_1MHz.complex")

# convert the 1d array to a 2d array
file = sig.ndarr_to_iq(file)

# returns a list of views of the array partitioned into sections of the determined length
file = sig.partition(file, 128)


print(file.__len__())


# Graph the IQ data
title = "Graph"
fig = plot_IQ(iq=file[0], title=title)
plt.show()


