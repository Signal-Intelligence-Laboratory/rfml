# Most of this taken from /examples/signal_classification.py
# Edited to fit new dataset
import torch
import pandas as pd
import numpy as np

from rfml.data import build_dataset, DatasetBuilder
from rfml.nn.eval import (
    compute_accuracy,
    compute_accuracy_on_cross_sections,
    compute_confusion,
)
from rfml.nn.model import build_model
from rfml.nn.train import build_trainer, PrintingTrainingListener

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
file = np.fromfile(file="/home/garrett/PlutoSDR-20250304_142236-88_9MHz-2_1MSps-2_1MHz.complex", dtype=np.float32)

# Split file into 
partitions = file.size / 256
splitfile = np.split(file, partitions)

builder = DatasetBuilder()

for x in splitfile:
    reshape_x = np.reshape(x, (2, 128))
    builder.add(reshape_x, Modulation="WBFM")


dataset = builder.build()

acc = compute_accuracy(model=model, data=dataset, le=le)
print("Overall Testing Accuracy: {:.4f}".format(acc))

print("Done")


