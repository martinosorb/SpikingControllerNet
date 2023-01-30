#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:40:11 2022

@author: pau
"""
# import libraries
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    control_neuron_rate,
    filter_spike_train,
    poisson_spikes,
    spikes_to_raster_plot,
    update_weights_poisson,
    update_weights_rates,
    learnAssociationTask,
)
from scipy.special import expit as sigmoid


v_th = 1
target_rates = [-1, 2]
learn_rate = 0.01
regularizer_rate = 0.0  # 0.00005#
C_precision = 0.01
data_noise = 0.1
timesteps = 1
initial_steps_plot = 5
neuron_noise = 0.

plot_path = "./plots/"


def train_one_example(x, y, w, plot=False):
    input_rate_FF = np.dot(sigmoid(x), w)
    # target_rate = target_rates[y]
    target_rate = y
    output, controller_effect = control_neuron_rate(input_rate_FF, target_rate)
    presynaptic_rate = np.vstack([sigmoid(x)] * len(output))

    postsynaptic_spikes = poisson_spikes(output)
    presynaptic_spikes = []

    DW = []
    for n in range(3):
        pre_spikes = poisson_spikes(presynaptic_rate[:, n])
        presynaptic_spikes.append(pre_spikes)
        filter_pot = filter_spike_train(pre_spikes)
        filter_dep = np.flip(filter_spike_train(np.flip(pre_spikes)))
        filtered_spikes = filter_pot - filter_dep
        DW.append(np.dot(filtered_spikes, postsynaptic_spikes))

    if plot:

        plt.plot(presynaptic_rate[:, 0], label="Presynaptic rate A")
        plt.plot(presynaptic_rate[:, 1], label="Presynaptic rate B")
        plt.plot(output, label="Postsynaptic rate C")
        plt.plot(controller_effect, label="Controller effect")
        plt.title("Single Learning Iteration")
        plt.legend()
        plt.show()
        # axes[1].plot(presynaptic_spikes, label='Presynaptic spikes')
        # axes[1].plot(postsynaptic_spikes, label='Postsynaptic spikes')
        spikes_to_raster_plot(postsynaptic_spikes)
        plt.show()
        # #axes[1].legend()
        # plt.plot(filtered_spikes, label='Postsynaptic Filtering')
        plt.show()

    return DW, controller_effect, postsynaptic_spikes, presynaptic_spikes


def train_input_weights(X, y, w_0, plot=True, dynamic_plot_idxs=[]):
    w = w_0
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

    # loop over datapoints
    for idx in range(len(y)):
        input_rate_FF = np.dot(sigmoid(X[idx, :]), w)  # input times linear in-weights
        target_rate = y[idx]  # output label

        control_target_rate = target_rates[target_rate]
        output, input_C = control_neuron_rate(input_rate_FF, target_rate, control_target_rate, timesteps=timesteps)
        # print(" Idx "+str(idx) + " required out: "+ str(target_rate) +
        #       " controlled out: "+str(round(output[-1],2)) +" control: "+str(input_C[-1])+
        #       " original output: "+str(round(sigmoid(input_rate_FF),2)))
        R = len(output) - 1

        if R > 0:  # avoids learning if the feedback is already good
            presynaptic_rates = np.vstack([sigmoid(X[idx, :])] * R)
            Dw_DH = update_weights_rates(np.array(output[:-1]), presynaptic_rates)
            Dw_STDP = update_weights_poisson(output[:-1], presynaptic_rates)
            # print("    Weight "+str(w)+" Weight update "+str(Dw))
            w = w + Dw_STDP - regularizer_rate * w
            DW_DH_list.append(Dw_DH)
            DW_STDP_list.append(Dw_STDP)

        count_1 += y[idx]
        if count_1 == dynamic_plot_idxs[next_dyn_plot_idx]:
            list_output_dynamics.append(output[1:])
            list_controller_dynamics.append(input_C)
            next_dyn_plot_idx = (next_dyn_plot_idx + 1) % len(dynamic_plot_idxs)

        w_evol.append(w)
        FF_output_evol.append(output[0])
        time_to_targ_evol.append(len(input_C))
        if len(input_C) > 1:
            control_evol.append(input_C[-1])
        else:
            control_evol.append(0)

    print("Output = 1 was shown " + str(count_1) + " times")

    if plot:
        plt.show()

        # ax = plt.axes()
        # ax.set_facecolor("white")
        plt.plot(w_evol)
        plt.legend(["$w_{A->C}$", "$w_{B->C}$", "$w_{bias}$"])
        plt.xlabel("Example")
        plt.ylabel("Weights")
        plt.savefig(
            plot_path + "WeightEvolution.eps", bbox_inches="tight", format="eps"
        )

        plt.show()

        # Benni: Fig 2, panel 1. Normalized to have the maximum at 1
        max_FF = max(FF_output_evol)
        max_C = max(control_evol)
        max_T = max(time_to_targ_evol)
        control_evol_norm = [c / max_C for (c, s) in zip(control_evol, y) if s == 1]
        FF_output_evol_norm = [
            (1 - f) ** 2 for (f, s) in zip(FF_output_evol, y) if s == 1
        ]
        time_to_output_evol_norm = [
            t / max_T for (t, s) in zip(time_to_targ_evol, y) if s == 1
        ]
        plt.plot(control_evol_norm, label="Feedback")
        plt.plot(FF_output_evol_norm, label="MSE Error")
        plt.plot(time_to_output_evol_norm, label="Time to target output")
        plt.xlabel("Example")
        plt.ylabel("Loss")
        plt.title("Loss functions for class 1")
        plt.legend()
        plt.savefig(
            plot_path + "OptimizationFunctions.eps", bbox_inches="tight", format="eps"
        )
        plt.show()

        # Benni: Fig 4 A
        for i in range(len(list_output_dynamics)):
            str_dynamics = "Dynamics at example " + str(dynamic_plot_idxs[i] - 1)
            dynamics = list_output_dynamics[i]
            # init_dynamics, v = runNeuron_rate([0]*initial_steps_plot)
            init_dynamics = [
                dynamics[0] + n * neuron_noise / 2
                for n in np.random.randn(initial_steps_plot)
            ]
            list_to_plot = init_dynamics + dynamics
            plt.plot(list_to_plot, label=str_dynamics)
        # plt.plot(output, label="Dynamics at last example")
        plt.ylim([0.0, 1])
        plt.xlabel("Time")
        plt.ylabel("Output rate")
        plt.legend()
        plt.savefig(
            plot_path + "TemporalDynamics.eps", bbox_inches="tight", format="eps"
        )
        plt.show()

        for i in range(len(list_output_dynamics)):
            str_dynamics = "Feedback at example " + str(dynamic_plot_idxs[i] - 1)
            init_ctr = list_controller_dynamics[i]
            init_control = [
                init_ctr[0] + n * neuron_noise / 2
                for n in np.random.randn(initial_steps_plot)
            ]
            list_to_plot = init_control + list_controller_dynamics[i]
            plt.plot(list_to_plot, label=str_dynamics)

        # plt.plot(input_C, label="Feedback at last example")
        plt.xlabel("Time")
        plt.ylabel("Feedback strength")
        plt.legend()
        plt.show()
        plt.savefig(
            plot_path + "TemporalFeedback.eps", bbox_inches="tight", format="eps"
        )
        # STDP vs dendritic error update
        L_errors = int(
            len(DW_DH_list) / 2
        )  # when plotting errors after learning we get noise
        plt.scatter(DW_DH_list[:L_errors], DW_STDP_list[:L_errors])
        plt.xlabel("STDP weight update")
        plt.ylabel("Dendritic error update")
        plt.savefig(plot_path + "STDP_vs_DH.eps", bbox_inches="tight", format="eps")

        plt.show()
    return w


plt.show()

# # single example
# w = np.array([1,-1,0])
# x = np.array([0, 1,1])
# output_pot, inputC_pot, postsynaptic_pot, presynaptic_pot = train_one_example(x, 1, w, plot=True)
# x = np.array([1, 0,1])
# output_dep, inputC_dep, postsynaptic_dep, presynaptic_dep = train_one_example(x, 0, w, plot=True)


# #train
total_points = 21000
train_points = 20000
w_0 = np.array([-5, 5, -5])  # np.random.randn(3)*0.5
X, y = learnAssociationTask(total_points, data_noise=data_noise)
plt.scatter(X[:, 0], X[:, 1])
plt.title("Input Data")
plt.savefig(plot_path + "input_data.eps", bbox_inches="tight", format="eps")

plt.show()

number_of_examples_for_dynamics = 5
# dynamics_plot_idx = [int(p*train_points)+1 for p in np.linspace(0,0.4,number_of_examples_for_dynamics)]
dynamics_plot_idx = [1, 501, 2001, 9001]
W = train_input_weights(
    X[:train_points, :],
    y[:train_points],
    w_0,
    plot=True,
    dynamic_plot_idxs=dynamics_plot_idx,
)
classifier = np.dot(X[:train_points, :], W)
plt.show()
plt.scatter(y[:train_points], classifier, label="Training data")
classifier = np.dot(X[train_points:, :], W)
plt.scatter(y[train_points:], classifier, label="Testing data")
plt.legend()
plt.show()
