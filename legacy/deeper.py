import torch
from torchvision.datasets import MNIST
import torchvision.transforms as tr

import matplotlib.pyplot as plt
from ControlledModules import ControlledNetwork
from tqdm import tqdm
import numpy as np


target_rates = torch.tensor([0., 1.])
controller_precision = 0.01
controller_rate = 0.1
epochs = 1
tau_mem = 1.
tau_stdp = 2.54
lr = 0.001
mom = 0.1
wd = 1e-6
mode = "spiking"
layer_sizes = (784, 10)

plot_path = "./plots/"
plot = False
plot_receptive = True
MAX_SAMPLES = 60000


transform = tr.Compose([tr.ToTensor(), torch.flatten])
dataset = MNIST("./data/", train=True, transform=transform)
dataset_test = MNIST("./data/", train=False, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset)
dataloader_test = torch.utils.data.DataLoader(dataset_test)

net = ControlledNetwork(
    layer_sizes,
    mode=mode,
    tau_mem=tau_mem,
    tau_stdp=tau_stdp,
    controller_rate=controller_rate,
    controller_precision=controller_precision,
)
layer = net.layers[0]

w_evol = []
control_evol = np.zeros(MAX_SAMPLES)
FF_output_evol = np.empty(MAX_SAMPLES)
time_to_targ_evol = np.empty(MAX_SAMPLES)

optim = torch.optim.SGD(
    net.parameters(),
    lr=lr,
    momentum=mom,
    weight_decay=wd,
)

# loop over datapoints
for epoch in range(epochs):
    for idx, (x, y) in enumerate(tqdm(dataloader)):
        if idx >= MAX_SAMPLES:
            break

        target = torch.nn.functional.one_hot(y, num_classes=10).squeeze()
        x = x.squeeze()
        control_target_rate = target_rates[target]

        # FORWARD, with controller controlling
        first_output, n_steps = net.evolve_to_convergence(
            x, target, control_target_rate)
        optim.step()
        optim.zero_grad()

        w_evol.append(layer.ff.weight.data.squeeze().clone().numpy())
        FF_output_evol[idx] = torch.nn.functional.mse_loss(first_output, target)
        time_to_targ_evol[idx] = n_steps
        # if len(input_C) > 1:
        #     control_evol[idx] = input_C[-1].item()


# Print acc
acc = 0
for x, y in dataloader_test:
    target = torch.nn.functional.one_hot(y, num_classes=10).squeeze()
    out = net.feedforward(x.squeeze())
    predicted_label = torch.max(out, dim=-1)[1]
    acc += predicted_label == y

print("Test Accuracy: ", (acc / len(dataset_test)))


if plot:
    plt.figure(figsize=(10, 4))

    plt.subplot(132)
    # control_evol_norm = control_evol / np.max(control_evol)
    # plt.plot(control_evol_norm, label="Feedback")
    plt.plot(FF_output_evol, label="MSE Error")
    plt.xlabel("Sample")
    plt.ylabel("Loss")
    plt.title("Loss functions")
    plt.legend()

    plt.subplot(133)
    plt.plot(time_to_targ_evol, label="Time to target output")
    plt.xlabel("Sample")
    plt.ylabel("N. timesteps")
    plt.title("Time to target output")
    plt.legend()

    plt.show()

if plot_receptive:
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(16, 6))

    for row in range(2):
        for col in range(5):
            w = layer.ff.weight[row * 5 + col]
            axs[row][col].imshow(w.view(28, 28).detach())

    plt.show()
