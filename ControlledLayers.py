import torch
from math import prod

def spikify(rate):
    return (rate > torch.rand_like(rate)) * 1.


class BaseControlledLayer(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
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
        self.controller_dim = controller_dim

        self.stdp_decay = 1 - 1 / tau_stdp if tau_stdp else False
        if self.stdp_decay:
            self.neg_stdp_amplitude = 2. / (1 + alpha_stdp) / tau_stdp
            self.pos_stdp_amplitude = alpha_stdp * self.neg_stdp_amplitude
            self.Apre = torch.nn.Parameter(torch.empty(batch_size, *input_shape), requires_grad=False)
            self.Apost = torch.nn.Parameter(torch.empty(batch_size, *output_shape), requires_grad=False)
        
        self.v = torch.nn.Parameter(torch.zeros((self.batch_size, *output_shape)), requires_grad=False)
        assert mode == "spiking" or mode == "rate"
        self.mode = mode
        self.dynamics = self._spiking_dynamics if mode == "spiking" else self._rate_dynamics

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
            self.ff.weight.grad -= self._stdp_update(input_spikes, output_spikes)
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
        input_shape = (fan_in + int(bias), )
        output_shape = (fan_out, )
        print(input_shape, output_shape)

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            controller_dim=controller_dim,
            batch_size=batch_size,
            mode=mode,
            tau_mem=tau_mem,
            tau_stdp=tau_stdp,
            alpha_stdp=alpha_stdp,
            bias=bias,
        )

        self.ff = torch.nn.Linear(input_shape[0], output_shape[0], bias=False)
        self.fb = torch.nn.Linear(self.controller_dim, output_shape[0], bias=False)
        self.reset()

    def _stdp_update(self, input_spikes, output_spikes):
        neg_outer = torch.einsum("bi,bj->bji", (input_spikes, self.Apost)).mean(dim=0)
        pos_outer = torch.einsum("bi,bj->bij", (output_spikes, self.Apre)).mean(dim=0)
        update = self.pos_stdp_amplitude * pos_outer - self.neg_stdp_amplitude * neg_outer
        return update


class ControlledConvLayer(BaseControlledLayer):
    def __init__(
        self,
        conv_layer,
        input_shape,
        output_shape,
        controller_dim,
        batch_size,
        mode="spiking",
        tau_mem=10.,
        tau_stdp=False,
        alpha_stdp=1.0,  # ratio A+/A-
    ):
        output_size = prod(output_shape)
        self.kernel_size = conv_layer.kernel_size

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            controller_dim=controller_dim,
            batch_size=batch_size,
            mode=mode,
            tau_mem=tau_mem,
            tau_stdp=tau_stdp,
            alpha_stdp=alpha_stdp,
            bias=False,
        )

        self.ff = conv_layer
        self.fb = torch.nn.Linear(self.controller_dim, output_size, bias=False)
        self.unfold = torch.nn.Unfold(
            kernel_size=conv_layer.kernel_size,
            dilation=conv_layer.dilation,
            padding=conv_layer.padding,
            stride=conv_layer.stride
        )
        self.reset()

    def _stdp_update(self, input_spikes, output_spikes):
        batch_size, in_ch, _, _ = input_spikes.shape
        batch_size, out_ch, out_h, out_w = input_spikes.shape
        u_shape = (batch_size, in_ch, *self.kernel_size, out_h, out_w)
        u_apre = self.unfold(self.Apre).reshape(u_shape)
        u_input = self.unfold(input_spikes).reshape(u_shape)
        pos_outer = torch.einsum('bcabhw,bkhw->kcab', u_apre, output_spikes)
        neg_outer = torch.einsum('bcabhw,bkhw->kcab', u_input, self.Apost)
        assert pos_outer.shape == (out_ch, in_ch, *self.kernel_size)
        update = self.pos_stdp_amplitude * pos_outer - self.neg_stdp_amplitude * neg_outer
        return update
