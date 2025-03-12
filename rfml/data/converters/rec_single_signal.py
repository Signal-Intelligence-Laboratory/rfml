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
from warnings import warn

# Internal Includes 
from rfml.data import Dataset, DatasetBuilder

class SingleSignalDataLoader(object):
    def load(self, path: str):
        if path is not None:
            if not os.path.exists(path):
                raise ValueError(
                    "If path is provided, it must actually exist.  Provided path: "
                    "{}".format(path)
                )
            return self.load_by_filetype(path=path)

