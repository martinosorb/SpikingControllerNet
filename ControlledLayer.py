import torch


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

    def reset(self):
        self.v = torch.zeros(self.fan_out)
