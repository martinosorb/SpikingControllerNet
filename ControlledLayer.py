import torch
import numpy as np


class ControlledLayer(torch.nn.Module):
    def __init__(self, fan_in, fan_out, controller_dim, mode="spiking", leak=0.9):
        super().__init__()

        self.leak = leak
        self.threshold = 1.
        self.ff = torch.nn.Linear(fan_in, fan_out, bias=False)
        self.fb = torch.nn.Linear(controller_dim, fan_out, bias=False)
        self.fan_out = fan_out
        self.reset()

        assert mode == "spiking" or mode == "rate"

        self.dynamics = self._spiking_dynamics if mode == "spiking" else self._rate_dynamics

    def forward(self, inputs, c):
        ff_input = self.ff(inputs)
        fb_input = self.fb(c)

        # LIF dynamics
        self.v += -self.leak * self.v + ff_input + fb_input
        return self.dynamics()

    def _spiking_dynamics(self):
        spikes = self.v > self.threshold
        self.v[spikes] = 0.

        return spikes

    def _rate_dynamics(self):
        return torch.sigmoid(self.v)

    def parameters(self, recurse: bool = True):
        return (self.ff.parameters())

    def reset(self):
        self.v = torch.zeros(self.fan_out)


class ControlledNetwork(torch.nn.Module):
    def __init__(self, layers, mode="spiking", leak=0.9, controller_rate=0.1):
        super().__init__()
        self.layers = []
        n_classes = layers[-1]
        self.controller_rate = controller_rate
        self.c = torch.zeros(n_classes)

        for fan_in, fan_out in zip(layers[:-1], layers[1:]):
            self.layers.append(ControlledLayer(fan_in, fan_out, controller_dim=n_classes, mode=mode, leak=leak))

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
        controller_effect = []

        while True:
            output_rate = self(x, self.c)
            self.evolve_controller(output_rate, control_target_rate)
            outputs.append(output_rate.numpy())
            controller_effect.append(self.c.clone().numpy())
            if abs(output_rate - target_rate) <= precision: break

        outputs = np.asarray(outputs)
        controller_effect = (outputs - outputs[0])[1:]
        return outputs, controller_effect

    def reset(self):
        self.c.zero_()
        for layer in self.layers:
            layer.reset()



# ========
# for data in dataset:
#     X, y = data
#     c = 0.

#     for c < error:
#         yhat = net(X)
#         c = net.controller(y, yhat)

#     weight_update()
