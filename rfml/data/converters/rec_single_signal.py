"""
Data loaders for files with complex array datatypes. 

Base design created with conversion from recorded files obtained from Universal Radio Hacker (https://github.com/jopohl/urh) as the intended input. 
Possible future iterations will include other single signal recording file types. 
"""

"""
Per URH FileOperator Documentation (urh/src/urh/util/FileOperator.py):

SIGNAL_FILE_EXTENSIONS_BY_TYPE = {
    np.int8: ".complex16s",
    np.uint8: ".complex16u",
    np.int16: ".complex32s",
    np.uint16: ".complex32u",
    np.float32: ".complex",
    np.complex64: ".complex",
}

Note from editor: from what I've found in the URH code (urh/src/urh/signalprocessing/IQArray.py) it seems like ".complex" types default to np.float32
"""

# External Includes
import numpy as np
import os

# Internal Includes 
from rfml.data import Dataset, DatasetBuilder

class SingleSignalDataLoader(object):
    """
    Base load function, takes a filepath as a string, checks if the path exists, passes to the actual loader.
    """
    def load(self, path: str) -> np.ndarray:
        if path is not None:
            if not os.path.exists(path):
                raise ValueError(
                    f"If path is provided, it must actually exist.  Provided path: {path}"
                )
            return self.load_by_filetype(path=path)
        
    """
    Accessed by load() function, passes back an NDArray of the data type given by the file extension.
    """
    def load_by_filetype(self, path: str) -> np.ndarray:
        if path.endswith(".complex16u") or path.endswith(".cu8"):
            return np.fromfile(path, dtype=np.uint8)
        elif path.endswith(".complex16s") or path.endswith(".cs8"):
            return np.fromfile(path, dtype=np.int8)
        elif path.endswith(".complex32u") or path.endswith(".cu16"):
            return np.fromfile(path, dtype=np.uint16).conver
        elif path.endswith(".complex32s") or path.endswith(".cs16"):
            return np.fromfile(path, dtype=np.int16)
        else:
            return np.fromfile(path, dtype=np.float32)

    """
    Takes in a 1-D NDArray[] of IQ samples and converts it to be 2-D with I and Q in separate arrays at the same index.
    """
    def ndarr_to_iq(self, arr: np.ndarray) -> np.ndarray:
        if len(arr) % 2 == 0:
            arr = arr.reshape((2, -1), order="C")
        else:  # ignore the last half sample to avoid a conversion error
            arr = arr[:-1].reshape((2, -1), order="C")
        return arr

    
    # def partition(self, nsamps=int):

    # for partition, use a modulo function to find number of samples to be cut off, and cut an even amount off the front and back



