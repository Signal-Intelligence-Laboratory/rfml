# Plotting Includes
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# Include sys to use modules
import sys
sys.path.insert(0, "/home/garrett/Code/rfml")

# Testing Includes
import unittest
import numpy as np

# Module includes
from rfml.nbutils import plot_IQ
from rfml.data.converters.rec_urh_single_signal import SingleSignalDataLoader 

class TestRecURHExtension(unittest.TestCase):


    def test_load_all_extensions(self):
        sig = SingleSignalDataLoader()

        self.assertEqual(
            sig.load("/home/garrett/Code/rfml/sigint_test/urh_test_signals/PlutoSDR_88.9_mar20.complex").dtype, np.float32
        )
        self.assertEqual(
            sig.load("/home/garrett/Code/rfml/sigint_test/urh_test_signals/PlutoSDR_88.9_mar20.complex16s").dtype, np.int8
        )
        self.assertEqual(
            sig.load("/home/garrett/Code/rfml/sigint_test/urh_test_signals/PlutoSDR_88.9_mar20.complex16u").dtype, np.uint8
        )
        self.assertEqual(
            sig.load("/home/garrett/Code/rfml/sigint_test/urh_test_signals/PlutoSDR_88.9_mar20.complex32s").dtype, np.int16
        )
        self.assertEqual(
            sig.load("/home/garrett/Code/rfml/sigint_test/urh_test_signals/PlutoSDR_88.9_mar20.complex32u").dtype, np.uint16
        )



if __name__ == '__main__':
    unittest.main()