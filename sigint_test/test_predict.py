# Most of this taken from /examples/signal_classification.py
# Edited to fit new dataset
import torch
import pandas as pd
import numpy as np

# Include sys to use modules
import sys
sys.path.insert(0, "/home/garrett/Code/rfml")

from rfml.data import build_dataset, DatasetBuilder
from rfml.nn.eval import (
    compute_accuracy,
)
from rfml.nn.model import build_model
from rfml.nn.train import build_trainer, PrintingTrainingListener
from rfml.data.converters.rec_urh_single_signal import SingleSignalDataLoader 

train, val, test, le = build_dataset(dataset_name="RML2016.10a", path="/home/garrett/RML2016.10a/RML2016.10a_dict.pkl")

model = build_model(model_name="CNN", input_samples=128, n_classes=len(le))


print("Model built")

# Trainer commented out for testing

trainer = build_trainer(
    strategy="standard", max_epochs=3, gpu=True
)  # Note: Disable the GPU here if you do not have one
trainer.register_listener(PrintingTrainingListener())
trainer(model=model, training=train, validation=val, le=le)

# Load in file
file = np.fromfile(file="/home/garrett/Code/stdout.dat", dtype=np.float32)

sig = SingleSignalDataLoader()

# Split file into 
# convert the 1d array to a 2d array
file = sig.ndarr_to_iq(file)

# returns a list of views of the array partitioned into sections of the determined length
file = sig.partition(file, 128)

builder = DatasetBuilder()

for x in file:
    builder.add(x, Modulation="WBFM")

dataset = builder.build()

acc = compute_accuracy(model=model, data=dataset, le=le)
print("Overall Testing Accuracy: {:.4f}".format(acc))

print("Done")


