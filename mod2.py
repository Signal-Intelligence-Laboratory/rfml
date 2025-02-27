# MAIN POINT OF THIS BRANCH IS TO FIND OUT WHY THIS CODE WON'T WORK
# Having issues with the trainer function

# Plotting Includes
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# External Includes
import numpy as np
from pprint import pprint

from torch.autograd import Variable
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

# Internal Includes
from rfml.data import Dataset, Encoder
from rfml.data.converters import load_RML201610A_dataset

from rfml.nbutils import plot_acc_vs_snr, plot_confusion, plot_convergence, plot_IQ

from rfml.nn.eval import compute_accuracy, compute_accuracy_on_cross_sections, compute_confusion
from rfml.nn.model import CNN, build_model
from rfml.nn.train.standard_with_listeners import StandardWithListeners


gpu = True       # Set to True to use a GPU for training
fig_dir = None   # Set to a file path if you'd like to save the plots generated
data_path = "/home/garrett/RML2016.10a/RML2016.10a_dict.pkl" # Set to a file path if you've downloaded RML2016.10A locally

dataset = load_RML201610A_dataset(path=data_path)
print(len(dataset))
pprint(dataset.get_examples_per_class())

train, test = dataset.split(frac=0.3, on=["Modulation", "SNR"])
train, val = train.split(frac=0.05, on=["Modulation", "SNR"])

print()
print("Training Examples")
print("=================")
pprint(train.get_examples_per_class())
print("=================")
print()
print("Validation Examples")
print("=================")
pprint(val.get_examples_per_class())
print("=================")
print()
print("Testing Examples")
print("=================")
pprint(test.get_examples_per_class())
print("=================")

le = Encoder(["WBFM",
              "AM-DSB",
              "AM-SSB",
              "CPFSK",
              "GFSK",
              "BPSK",
              "QPSK",
              "8PSK",
              "PAM4",
              "QAM16",
              "QAM64"],
             label_name="Modulation")
print(le)


# Plot a sample of the data
# You can choose a different sample by changing
idx = 10
snr = 18.0
modulation = "8PSK"

mask = (dataset.df["SNR"] == snr) & (dataset.df["Modulation"] == modulation)
sample = dataset.as_numpy(mask=mask, le=le)[0][idx,0,:]
t = np.arange(sample.shape[1])

title = "{modulation} Sample at {snr:.0f} dB SNR".format(modulation=modulation, snr=snr)
fig = plot_IQ(iq=sample, title=title)
if fig_dir is not None:
    file_path = "{fig_dir}/{modulation}_{snr:.0f}dB_sample.pdf".format(fig_dir=fig_dir,
                                                                       modulation=modulation,
                                                                       snr=snr)
    print("Saving Figure -> {file_path}".format(file_path=file_path))
    fig.savefig(file_path, format="pdf", transparent=True)
plt.show()


# Now we create the CNN model
# I am not copying and pasting this code from the module because it is all handled in the class CNN

#cnnmodel = CNN(input_samples=128, n_classes=11)
cnnmodel = build_model("CNN", input_samples=128, n_classes=11)
print(cnnmodel)

# And now train it

trainer = StandardWithListeners(max_epochs=3, gpu=gpu)
#trainer = build_trainer("standard", max_epochs=3, gpu=gpu)
print(trainer)

trainer(model=cnnmodel,
        training=train,
        validation=val,
        le=le)
# keep in mind if i want trainer to actually have outputs to like graph, need to make a new training listener (printing_training_listener only prints to stdout)

# # And plot the training results
# title = "Training Results of {model_name} on {dataset_name}".format(model_name="MyCNN", dataset_name="RML2016.10A")
# fig = plot_convergence(train_loss=train_loss, val_loss=val_loss, title=title)
# if fig_dir is not None:
#     file_path = "{fig_dir}/training_loss.pdf"
#     print("Saving Figure -> {file_path}".format(file_path=file_path))
#     fig.savefig(file_path, format="pdf", transparent=True)
# plt.show()

