import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

neuron_noise = 0.


# generate a 2-class classification problem with 1,000 data points,
# where each data point is a 2-D feature vector
def learnAssociationTask(points, data_noise):
    y = np.random.randint(2, size=points)
    y[points - 1] = 1  # for plotting purposes
    X = np.zeros([points, 3])
    X[:, 2] = np.ones(points)
    X[:, 0] = y + np.random.randn(points) * data_noise
    X[:, 1] = (np.ones(points) - y) + np.random.randn(points) * data_noise
    return X, y


def runNeuron_rate(input_current, leak=0.9, init_v=0):
    output = []
    v = init_v
    noise = np.random.randn(len(input_current)) * neuron_noise
    t = 0
    for inp in input_current:
        v = inp + v * (1 - leak) + noise[t]
        output.append(sigmoid(v))
        t += 1
    return output, v


def enlarge_vector_interpolation(vector, enlarge_factor):
    idx = np.linspace(0, 1, len(vector))
    spl = UnivariateSpline(idx, vector)
    new_idx = np.linspace(0, 1, int(enlarge_factor * len(vector)))
    expanded = spl(new_idx)
    return expanded


def control_neuron_rate(
    input_rate,
    target_rate,
    control_target_rate,
    timesteps,
    C_precision=0.01,
):
    inputs = input_rate * np.ones(timesteps * 2)
    output, v_0 = runNeuron_rate(inputs)
    output_ff = output[-1]  # np.mean(output[timesteps:])
    output_rate = output_ff
    control_input = 0
    outputs = [output_rate]
    controller_input = []
    controller_effect = []

    # noise = np.random.randn(400)*neuron_noise
    # idx_noise = 0

    while abs(output_rate - target_rate) > C_precision:
        control_input = control_input + 0.1 * (control_target_rate - output_rate)
        new_input = input_rate + control_input
        v_0 = new_input + np.random.randn() * neuron_noise
        output_rate = sigmoid(v_0)
        outputs.append(output_rate)
        controller_input.append(control_input)
        controller_effect.append(output_rate - output_ff)
        # We can also simulate neurons with their own dynamics (but it takes longer)
        # output, v_0 = runNeuron_rate(new_input*np.ones(timesteps), init_v=v_0)
        # output_rate = output[-1]
        # outputs.extend(output)
    return outputs, controller_effect  # controller_input


def poisson_spikes(firing_rate, timesteps=1):
    # convert rates to spikes...
    expanded_rate = np.repeat(firing_rate, timesteps)
    spikes = 1 * (np.random.rand(len(expanded_rate)) < expanded_rate)
    return spikes


def filter_spike_train(spikes, decay_rate=0.8):
    filtered_train = np.zeros(spikes.shape)
    a = 0
    for t in range(len(spikes)):
        a = a * decay_rate + spikes[t]
        filtered_train[t] = a
    return filtered_train


def STDP(input_spikes, output_spikes, tau_STDP=0.5):
    decay_rate = np.exp(-tau_STDP)
    filter_pot = filter_spike_train(input_spikes, decay_rate=decay_rate)
    filter_dep = np.flip(
        filter_spike_train(np.flip(input_spikes), decay_rate=decay_rate)
    )
    STDP_input_filter = filter_pot - filter_dep
    Dw = np.dot(output_spikes, STDP_input_filter)
    return Dw


def update_weights_poisson(output_rates, input_rates, learn_rate=0.01):
    s = input_rates.shape
    Dw = np.zeros(s[1])
    output_spikes = poisson_spikes(output_rates)
    for c in range(s[1]):
        input_spikes = poisson_spikes(input_rates[:, c])
        Dw[c] = STDP(input_spikes, output_spikes)
    return learn_rate * Dw  # /s[0]


def update_weights_rates(output_rates, inputs, learn_rate=0.01):
    T = len(output_rates)
    Doutput = output_rates[1:] - output_rates[: (T - 1)]
    DW = np.dot(Doutput.T, inputs[0 : (T - 1), :])
    return learn_rate * DW  # /T


def spikes_to_raster_plot(spikes):
    s = spikes.shape
    if len(s) == 1:
        T = s[0]
        for t in range(T):
            if spikes[t]:
                plt.vlines(t, 0, 0.9)
    else:
        N = s[1]
        T = s[0]
        for n in range(N):
            for t in range(T):
                if spikes[t, n]:
                    plt.vlines(t, n, n + 0.9, colors="k")
