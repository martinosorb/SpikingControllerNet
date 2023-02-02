import torch
import numpy as np


def spikify(rate):
    return rate > torch.rand_like(rate)


class ControlledLayer(torch.nn.Module):
    def __init__(self, fan_in, fan_out, controller_dim, mode="spiking", leak=0.9, stdp_tau=False):
        super().__init__()

        self.leak = leak
        self.threshold = 1.
        self.ff = torch.nn.Linear(fan_in, fan_out, bias=False)
        self.fb = torch.nn.Linear(controller_dim, fan_out, bias=False)
        self.fan_out = fan_out
        self.reset()

        assert mode == "spiking" or mode == "rate"
        self.mode = mode
        self.dynamics = self._spiking_dynamics if mode == "spiking" else self._rate_dynamics

        self.stdp_tau = stdp_tau
        if stdp_tau:
            self.Apre = torch.zeros(fan_in)
            self.Apost = torch.zeros(fan_out)

    def forward(self, inputs, c):
        ff_input = self.ff(inputs)
        fb_input = self.fb(c)

        # LIF dynamics
        self.v += -self.leak * self.v + ff_input + fb_input
        outputs = self.dynamics()

        if self.stdp_tau:
            if self.ff.weight.grad is None:  # TODO
                self.ff.weight.grad = torch.zeros_like(self.ff.weight)
            if self.mode == "rate":
                input_spikes = spikify(inputs)
                output_spikes = spikify(outputs)
            else:
                input_spikes, output_spikes = inputs, outputs
            self.Apre += -self.Apre / self.stdp_tau + input_spikes.float()
            self.Apost += -self.Apost / self.stdp_tau - output_spikes.float()
            self.ff.weight.grad -= torch.outer(input_spikes, self.Apost).T + torch.outer(output_spikes, self.Apre)

        return outputs

    def _spiking_dynamics(self):
        spikes = self.v > self.threshold
        self.v[spikes] = 0.

        return spikes

    def _rate_dynamics(self):
        return torch.sigmoid(self.v)

    def reset(self):
        self.v = torch.zeros(self.fan_out)


class ControlledNetwork(torch.nn.Module):
    def __init__(self, layers, mode="spiking", leak=0.9, controller_rate=0.1, stdp_tau=False):
        super().__init__()
        self.layers = []
        n_classes = layers[-1]
        self.controller_rate = controller_rate
        self.c = torch.zeros(n_classes)

        for fan_in, fan_out in zip(layers[:-1], layers[1:]):
            self.layers.append(ControlledLayer(fan_in, fan_out, controller_dim=n_classes, mode=mode, leak=leak, stdp_tau=stdp_tau))

        self.seq = torch.nn.Sequential(*self.layers)  # just so that pytorch registers them

    def forward(self, x, c):
        for layer in self.layers:
            x = layer(x, c)
        return x

    def evolve_controller(self, current_output, control_target_rate):
        error = control_target_rate - current_output
        self.c += self.controller_rate * error

    def evolve_to_convergence(self, x, target_rate, control_target_rate, precision=0.01):
        self.reset()
        outputs = []

        while True:
            output_rate = self(x, self.c)
            self.evolve_controller(output_rate, control_target_rate)
            outputs.append(output_rate.detach().numpy())
            if abs(output_rate - target_rate) <= precision: break

        outputs = np.asarray(outputs)
        controller_effect = (outputs - outputs[0])[1:]
        return outputs, controller_effect

    def parameters(self, recurse: bool = True):
        return (l.ff.weight for l in self.layers)

    def reset(self):
        self.c.zero_()
        for layer in self.layers:
            layer.reset()

