# Plotting Includes
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# External Includes
import numpy as np
from pprint import pprint
import pickle
from collections import defaultdict

# Internal Includes
from rfml.data import Dataset, Encoder, DatasetBuilder
from rfml.data.converters import rml_2016

from rfml.nbutils import plot_acc_vs_snr, plot_confusion, plot_convergence, plot_IQ

from rfml.nn.eval import compute_accuracy, compute_accuracy_on_cross_sections, compute_confusion
from rfml.nn.model import Model

gpu = True       # Set to True to use a GPU for training
fig_dir = None   # Set to a file path if you'd like to save the plots generated
data_path = "/home/garrett/RML2016.10a/RML2016.10a_dict.pkl" # Set to a file path if you've downloaded RML2016.10A locally

dataloader = rml_2016.RML2016DataLoader(cache_path="", remote_url="", unpickled_path=data_path, warning_msg="Stop")
builder = DatasetBuilder()


with open(file=data_path, mode="rb") as file:
            data = pickle.load(file, encoding="latin")

            description = defaultdict(list)
            # Declare j just to get the linter to stop complaining about the lamba below
            j = None
            snrs, mods = map(
                lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [1, 0]
            )
            for mod in mods:
                for snr in snrs:
                    description[mod].append(snr)

# print(data)
# print(description)

for mod, snrs in description.items():
        for snr in snrs:
            for iq in data[(mod, snr)]:
                builder.add(iq=iq, Modulation=mod, SNR=snr)
        
dataset = builder.build()

print(dataset.columns)

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
# print(le)


snr = 18.0
modulation = "BPSK"
mask = (dataset.df["SNR"] == snr) & (dataset.df["Modulation"] == modulation)

bpsk = dataset.df[mask]
# print(bpsk)

bpsk.pop("SNR")
bpsk.pop("Modulation")

# print(bpsk.values[0][0])
# print(bpsk.shape[0])

# inphase = []
# quad = []

# for i in range(bpsk.shape[0]):
#       inphase.append(bpsk.to_numpy())
#     #   inphase.append(bpsk.values[i][0])
#     #   quad.append(bpsk.values[i][1])

# print(inphase)
# print("half")
# print(quad)

print(bpsk["I"].to_numpy())








# sample = dataset.as_numpy(mask=mask, le=le)[0][idx,0,:]
# t = np.arange(sample.shape[1])
# sample.tofile("/home/garrett/Code/rml_analysis_bpsk.dat")

# title = "{modulation} Sample at {snr:.0f} dB SNR".format(modulation=modulation, snr=snr)
# fig = plot_IQ(iq=sample, title=title)
# if fig_dir is not None:
#     file_path = "{fig_dir}/{modulation}_{snr:.0f}dB_sample.pdf".format(fig_dir=fig_dir,
#                                                                        modulation=modulation,
#                                                                        snr=snr)
#     print("Saving Figure -> {file_path}".format(file_path=file_path))
#     fig.savefig(file_path, format="pdf", transparent=True)
# plt.show()