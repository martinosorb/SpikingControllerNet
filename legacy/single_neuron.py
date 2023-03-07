import torch
import matplotlib.pyplot as plt
from utils import update_weights_rates
from sklearn.datasets import make_blobs
from ControlledModules import ControlledNetwork
from tqdm import tqdm
import numpy as np


v_th = 1
target_rates = [-1, 2]
C_precision = 0.01
timesteps = 2
initial_steps_plot = 5
neuron_noise = 0.

plot_path = "./plots/"
plot = True

dynamic_plot_idxs = [1, 501, 2001, 9001]

data_noise = 0.1
total_points = 10000
X, y = make_blobs(total_points, n_features=3, centers=[[0, 1, 0], [1, 0, 0]], cluster_std=data_noise)
X[:, 2] = 1.

net = ControlledNetwork((3, 1), mode="rate", tau_mem=1., tau_stdp=2.54)
layer = net.layers[0]

w_evol = []
control_evol = []
FF_output_evol = []
time_to_targ_evol = []
DW_DH_list = []
DW_STDP_list = []
list_output_dynamics = []
list_controller_dynamics = []

count_1 = 0
next_dyn_plot_idx = 0
X = torch.from_numpy(X).float()

lr = 0.01
optim = torch.optim.SGD(
    net.parameters(),
    lr=lr,
    momentum=0.0,
    weight_decay=0.0
)


# loop over datapoints
for idx in tqdm(range(len(y))):
    target_rate = y[idx]
    x = X[idx]
    control_target_rate = target_rates[target_rate]
    optim.zero_grad()

    # FORWARD, with controller controlling
    output, input_C = net.evolve_to_convergence(
        x, target_rate, control_target_rate, precision=C_precision)

    R = len(output) - 1  # number of timesteps the controller had to act
    if R > 0:  # avoids learning if the feedback is already good
        DW_STDP_list.append(-lr * layer.ff.weight.grad.clone().numpy())
        optim.step()

        # Calculate (but not use) DH update using Pau's functions
        presynaptic_rates = torch.sigmoid(x).unsqueeze(0).expand((R, -1)).numpy()
        Dw_DH = update_weights_rates(output[:-1], presynaptic_rates)
        DW_DH_list.append(Dw_DH)

        # # Pau's version of STDP
        # Dw_STDP = update_weights_poisson(output[:-1], presynaptic_rates)
        # DW_STDP_list.append(Dw_STDP)

    count_1 += y[idx]
    if count_1 == dynamic_plot_idxs[next_dyn_plot_idx]:
        list_output_dynamics.append(output[1:])
        list_controller_dynamics.append(input_C)
        next_dyn_plot_idx = (next_dyn_plot_idx + 1) % len(dynamic_plot_idxs)

    w_evol.append(layer.ff.weight.data.squeeze().clone().numpy())
    FF_output_evol.append(output[0])
    time_to_targ_evol.append(len(input_C))

    if len(input_C) > 1:
        control_evol.append(input_C[-1])
    else:
        control_evol.append(0)

DW_DH_list = np.asarray(DW_DH_list)
DW_STDP_list = np.asarray(DW_STDP_list)


if plot:
    plt.figure(figsize=(10, 6))
    plt.subplot(231)
    plt.scatter(X[:, 0], X[:, 1], s=2, c=y)
    plt.title("Input Data")

    plt.subplot(232)
    plt.plot(w_evol)
    plt.legend(["$w_{A->C}$", "$w_{B->C}$", "$w_{bias}$"])
    plt.xlabel("Example")
    plt.ylabel("Weights")

    plt.subplot(233)
    max_C = max(control_evol)
    max_T = max(time_to_targ_evol)
    control_evol_norm = [c / max_C for (c, s) in zip(control_evol, y) if s == 1]
    FF_output_evol_norm = [(1 - f) ** 2 for (f, s) in zip(FF_output_evol, y) if s == 1]
    time_to_output_evol_norm = [t / max_T for (t, s) in zip(time_to_targ_evol, y) if s == 1]
    plt.plot(control_evol_norm, label="Feedback")
    plt.plot(FF_output_evol_norm, label="MSE Error")
    plt.plot(time_to_output_evol_norm, label="Time to target output")
    plt.xlabel("Example")
    plt.ylabel("Loss")
    plt.title("Loss functions for class 1")
    plt.legend()

    plt.subplot(234)
    for i in range(len(list_output_dynamics)):
        str_dynamics = "Dynamics at example " + str(dynamic_plot_idxs[i] - 1)
        plt.plot(list_output_dynamics[i], label=str_dynamics)
    plt.ylim([0.0, 1])
    plt.xlabel("Time")
    plt.ylabel("Output rate")
    plt.legend()

    plt.subplot(235)
    for i in range(len(list_output_dynamics)):
        str_dynamics = "Feedback at example " + str(dynamic_plot_idxs[i] - 1)
        plt.plot(list_controller_dynamics[i], label=str_dynamics)

    plt.xlabel("Time")
    plt.ylabel("Feedback strength")
    plt.legend()

    plt.subplot(236)
    # STDP vs dendritic error update
    L_errors = int(len(DW_DH_list) / 2)  # when plotting errors after learning we get noise
    plt.scatter(DW_STDP_list[:L_errors], DW_DH_list[:L_errors], s=2)
    plt.xlabel("STDP weight update")
    plt.ylabel("Dendritic error update")

    plt.savefig(plot_path + "all_plots.pdf", bbox_inches="tight", format="eps")
    plt.show()
