# Most of this taken from /examples/signal_classification.py
# Edited to fit new dataset
import torch
import pandas as pd
import numpy as np

from rfml.data import build_dataset, Dataset
from rfml.nn.eval import (
    compute_accuracy,
    compute_accuracy_on_cross_sections,
    compute_confusion,
)
from rfml.nn.model import build_model
from rfml.nn.train import build_trainer, PrintingTrainingListener

# ADD ALL BELOW BACK WHEN DATA TYPE ISSUE RESOLVED

train, val, test, le = build_dataset(dataset_name="RML2016.10a", path="/home/garrett/RML2016.10a/RML2016.10a_dict.pkl")

model = build_model(model_name="CNN", input_samples=128, n_classes=len(le))

print("Model built")


trainer = build_trainer(
    strategy="standard", max_epochs=3, gpu=True
)  # Note: Disable the GPU here if you do not have one
trainer.register_listener(PrintingTrainingListener())
trainer(model=model, training=train, validation=val, le=le)


file = np.fromfile(file="/home/garrett/PlutoSDR-20250304_142236-88_9MHz-2_1MSps-2_1MHz.complex", dtype=np.float32)

print(file)
tensor_from_file = torch.tensor(file)

model_pass = model(tensor_from_file)


prediction = torch.argmax(model_pass, dim=1) # place file with single recorded radio file here
print(prediction)
print("Done")

# when viewing the base.py predict() implementation, it takes in a torch.Tensor and returns a torch.Tensor
# the input can be a torch.complex64 taken from the array of the .complex type from URH (.complex is of type np.complex64)
# the output is a 1d array with the prediction (will be an int, taken from init of onehot)

#print(prediction)




# acc = compute_accuracy(model=model, data=test, le=le)
# acc_vs_snr, snr = compute_accuracy_on_cross_sections(
#     model=model, data=test, le=le, column="SNR"
# )
# cmn = compute_confusion(model=model, data=test, le=le)

# # Calls to a plotting function could be inserted here
# # For simplicity, this script only prints the contents as an example
# print("===============================")
# print("Overall Testing Accuracy: {:.4f}".format(acc))
# print("SNR (dB)\tAccuracy (%)")
# print("===============================")
# for acc, snr in zip(acc_vs_snr, snr):
#     print("{snr:d}\t{acc:0.1f}".format(snr=snr, acc=acc * 100))
# print("===============================")
# print("Confusion Matrix:")
# print(cmn)

#model.save("cnn.pt")
