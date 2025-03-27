# Plotting Includes
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

import torch
import pandas as pd
import numpy as np

# Include sys to use modules
import sys
sys.path.insert(0, "/home/garrett/Code/rfml")

# Internal Includes
from rfml.data import DatasetBuilder, Encoder
from rfml.nn.eval import compute_accuracy, compute_confusion
from rfml.nbutils import plot_confusion, plot_IQ
from rfml.nn.model import build_model
from rfml.nn.train import build_trainer, PrintingTrainingListener
from rfml.data.converters.rec_urh_single_signal import SingleSignalDataLoader


# Add all of the samples to a full dataset
loader = SingleSignalDataLoader()
builder = DatasetBuilder()
le = Encoder(["8PSK", "16QAM", "BPSK"], label_name="Modulation")

file8PSK = loader.partition(loader.ndarr_to_iq(loader.load("/home/garrett/Code/gnu_8PSK.dat")), 128) # contains 5679 sections of 128 samples
file16QAM = loader.partition(loader.ndarr_to_iq(loader.load("/home/garrett/Code/gnu_16QAM.dat")), 128) # contains 3071 sections of 128 samples
fileBPSK = loader.partition(loader.ndarr_to_iq(loader.load("/home/garrett/Code/gnu_BPSK.dat")), 128) # contains 8191 sections of 128 samples

# smallest of these is just over 3000 so 3000 will be the max I will use for all

# Add 3000 sections of each modulation type to the dataset
j = 0
while j < 2999:
    for x in file8PSK:
        builder.add(x, Modulation="8PSK")
        j = j + 1
k = 0
while k < 2999:
    for y in file16QAM:
        builder.add(y, Modulation="16QAM")
        k = k + 1
l = 0
while l < 2999:
    for z in fileBPSK:
        builder.add(z, Modulation="BPSK")
        l = l + 1

# Build the dataset
dataset = builder.build()

train, test = dataset.split(frac=0.3, on=["Modulation"])
train, val = train.split(frac=0.05, on=["Modulation"])

# Create the CNN model
model = build_model(model_name="CNN", input_samples=128, n_classes=len(le))


# Train model on dataset
trainer = build_trainer(
    strategy="standard", max_epochs=3, gpu=True
)  # Note: Disable the GPU here if you do not have one
trainer.register_listener(PrintingTrainingListener())
trainer(model=model, training=train, validation=val, le=le)

acc = compute_accuracy(model=model, data=test, le=le)
print("Overall Testing Accuracy: {:.4f}".format(acc))

cmn = compute_confusion(model=model, data=test, le=le)

title = "Confusion Matrix"
fig = plot_confusion(cm=cmn, labels=le.labels, title=title)
plt.show()


