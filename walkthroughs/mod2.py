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
from rfml.data.converters import load_RML201610B_dataset

from rfml.nbutils import plot_acc_vs_snr, plot_confusion, plot_convergence, plot_IQ

from rfml.nn.eval import compute_accuracy, compute_accuracy_on_cross_sections, compute_confusion
from rfml.nn.model import Model


gpu = True       # Set to True to use a GPU for training
fig_dir = None   # Set to a file path if you'd like to save the plots generated
data_path = "/home/garrett/RML2016.10b/RML2016.10b.dat" # Set to a file path if you've downloaded RML2016.10A locally

dataset = load_RML201610B_dataset(path=data_path)
print(len(dataset))
pprint(dataset.get_examples_per_class())

