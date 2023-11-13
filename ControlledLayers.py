import torch

def spikify(rate):
    return (rate > torch.rand_like(rate)) * 1.


class BaseControlledLayer(torch.nn.Module):
    def __init__(
        self,
        controller_dim,
        batch_size,
        mode="spiking",
        tau_mem=10.,
        tau_stdp=False,
        alpha_stdp=1.0,  # ratio A+/A-
        bias=True,
    ):
        super().__init__()

        self.leak = 1. / tau_mem
        self.threshold = 1.
        self.batch_size = batch_size
        self.bias = bias

        assert mode == "spiking" or mode == "rate"
        self.mode = mode
        self.dynamics = self._spiking_dynamics if mode == "spiking" else self._rate_dynamics

        self.stdp_decay = 1 - 1 / tau_stdp if tau_stdp else False
        if self.stdp_decay:
            self.neg_stdp_amplitude = 2. / (1 + alpha_stdp) / tau_stdp
            self.pos_stdp_amplitude = alpha_stdp * self.neg_stdp_amplitude
        self.v = torch.nn.Parameter(
            torch.zeros((self.batch_size, self.fan_out)),
            requires_grad=False,
        )
        self.reset()

    def forward(self, inputs, c):
        # Add all-ones column to emulate layer biases
        if self.bias:
            inputs = torch.cat(
                (
                    inputs,
                    torch.ones((*inputs.shape[:-1], 1), device=inputs.device),
                ),
                dim=-1,
            )

        ff_input = self.ff(inputs)
        fb_input = self.fb(c)

        # LIF dynamics
        self.v.data += -self.leak * self.v + ff_input + fb_input
        outputs = self.dynamics()

        if self.stdp_decay:
            if self.mode == "rate":
                input_spikes = spikify(inputs)
                output_spikes = spikify(outputs)
            else:
                input_spikes, output_spikes = inputs, outputs
            self.Apre.data = self.Apre * self.stdp_decay + input_spikes.float()
            self.Apost.data = self.Apost * self.stdp_decay + output_spikes.float()
            neg_outer = torch.einsum("bi,bj->bij", (input_spikes, self.Apost)).transpose(1, 2).mean(dim=0)
            pos_outer = torch.einsum("bi,bj->bij", (output_spikes, self.Apre)).mean(dim=0)
            update = self.pos_stdp_amplitude * pos_outer - self.neg_stdp_amplitude * neg_outer
            self.ff.weight.grad -= update
        return outputs

    def _spiking_dynamics(self):
        spikes = self.v > self.threshold
        self.v[spikes] = 0.

        return spikes.float()

    def _rate_dynamics(self):
        return torch.sigmoid(self.v)

    def reset(self):
        if self.stdp_decay:
            self.Apre.zero_()
            self.Apost.zero_()
        if self.ff.weight.grad is None:
            self.ff.weight.grad = torch.zeros_like(self.ff.weight, device=self.ff.weight.device)
        self.v.zero_()

    @property
    def grad(self):
        return self.ff.weight.grad
    

class ControlledLayer(BaseControlledLayer):
    def __init__(
        self,
        fan_in,
        fan_out,
        controller_dim,
        batch_size,
        mode="spiking",
        tau_mem=10.,
        tau_stdp=False,
        alpha_stdp=1.0,  # ratio A+/A-
        bias=True,
    ):
        super().__init__(
            controller_dim=controller_dim,
            batch_size=batch_size,
            mode=mode,
            tau_mem=tau_mem,
            tau_stdp=tau_stdp,
            alpha_stdp=alpha_stdp,
            bias=bias,
        )

        self.fan_out = fan_out
        self.ff = torch.nn.Linear(fan_in + int(bias), fan_out, bias=False)
        self.fb = torch.nn.Linear(controller_dim, fan_out, bias=False)

        if self.stdp_decay:
            self.Apre = torch.nn.Parameter(torch.empty(batch_size, fan_in + int(bias)), requires_grad=False)
            self.Apost = torch.nn.Parameter(torch.empty(batch_size, fan_out), requires_grad=False)


class ControlledConvLayer(BaseControlledLayer):
    def __init__(
        self,
        conv_layer,
        controller_dim,
        batch_size,
        mode="spiking",
        tau_mem=10.,
        tau_stdp=False,
        alpha_stdp=1.0,  # ratio A+/A-
    ):
        super().__init__(
            fan_in=1, # FIXME this is useless
            fan_out=1,
            controller_dim=controller_dim,
            batch_size=batch_size,
            mode=mode,
            tau_mem=tau_mem,
            tau_stdp=tau_stdp,
            alpha_stdp=alpha_stdp,
            bias=False,
        )
        self.ff = conv_layer
