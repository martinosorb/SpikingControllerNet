import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np


def spikify(rate):
    return (rate > torch.rand_like(rate)) * 1.


class ControlledLayer(torch.nn.Module):
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
    ):
        super().__init__()

        self.leak = 1. / tau_mem
        self.threshold = 1.
        self.fan_out = fan_out
        self.batch_size = batch_size
        self.ff = torch.nn.Linear(fan_in + 1, fan_out, bias=False)
        self.fb = torch.nn.Linear(controller_dim, fan_out, bias=False)

        assert mode == "spiking" or mode == "rate"
        self.mode = mode
        self.dynamics = self._spiking_dynamics if mode == "spiking" else self._rate_dynamics

        self.stdp_decay = 1 - 1 / tau_stdp if tau_stdp else False
        if self.stdp_decay:
            self.Apre = torch.nn.Parameter(torch.empty(batch_size, fan_in + 1), requires_grad=False)
            self.Apost = torch.nn.Parameter(torch.empty(batch_size, fan_out), requires_grad=False)
            self.neg_stdp_amplitude = 2. / (1 + alpha_stdp) / tau_stdp
            self.pos_stdp_amplitude = alpha_stdp * self.neg_stdp_amplitude
        self.v = torch.nn.Parameter(
            torch.zeros((self.batch_size, self.fan_out)),
            requires_grad=False,
        )
        self.reset()

    def forward(self, inputs, c):
        # Add all-ones column to emulate layer biases
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
        self.v.data[:] += -self.leak * self.v + ff_input + fb_input
        outputs = self.dynamics()

        if self.stdp_decay:
            if self.mode == "rate":
                input_spikes = spikify(inputs)
                output_spikes = spikify(outputs)
            else:
                input_spikes, output_spikes = inputs, outputs
            self.Apre.data[:] = self.Apre * self.stdp_decay + input_spikes.float()
            self.Apost.data[:] = self.Apost * self.stdp_decay + output_spikes.float()
            neg_outer = torch.einsum("bi,bj->bij", (input_spikes, self.Apost)).transpose(1, 2).mean(dim=0)
            pos_outer = torch.einsum("bi,bj->bij", (output_spikes, self.Apre)).mean(dim=0)
            update = self.pos_stdp_amplitude * pos_outer - self.neg_stdp_amplitude * neg_outer
            self.ff.weight.grad[:] = self.ff.weight.grad - update

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


class ControlledNetwork(pl.LightningModule):
    def __init__(
        self,
        layers,
        batch_size,
        mode="spiking",
        tau_mem=10.,
        tau_stdp=False,
        alpha_stdp=1.0,  # ratio A+/A-
    ):
        super().__init__()
        self.automatic_optimization = False

        self.layers = []
        self.batch_size = batch_size
        controller_dim = layers[-1]
        self.controller_dim = controller_dim
        self.c = torch.nn.Parameter(torch.zeros((batch_size, controller_dim)), requires_grad=False)

        for fan_in, fan_out in zip(layers[:-1], layers[1:]):
            layer = ControlledLayer(
                fan_in, fan_out, controller_dim=controller_dim, batch_size=batch_size, mode=mode,
                tau_mem=tau_mem, tau_stdp=tau_stdp, alpha_stdp=alpha_stdp)
            self.layers.append(layer)

        self.initialize_as_dfc()
        self.seq = torch.nn.Sequential(*self.layers)  # so that pytorch registers them

    def evolve_controller(self, current_output, control_target_rate):
        raise NotImplementedError()

    def initialize_as_dfc(self):
        # last layer has Q=identity
        curr_w = torch.eye(self.controller_dim, device=self.c.device)
        with torch.no_grad():
            for layer in self.layers[::-1]:  # layers backwards
                layer.fb.weight.data = curr_w
                curr_w = layer.ff.weight.T[:-1] @ curr_w

    def forward(self, x, c):
        for layer in self.layers:
            x = layer(x, c)
        return x

    def feedforward(self, x):
        return self(x, torch.zeros_like(self.c))

    def parameters(self, recurse: bool = True):
        return (l.ff.weight for l in self.layers)

    def reset(self):
        self.c.zero_()
        for layer in self.layers:
            layer.reset()


class DiffControllerNet(ControlledNetwork):
    def __init__(
        self,
        layers,
        mode="spiking",
        tau_mem=10.,
        tau_stdp=False,
        controller_rate=0.1,
        controller_precision=0.01,
        target_rates=[0., 1.],
        alpha_stdp=1.0,  # ratio A+/A-
    ):
        super().__init__(
            layers=layers, mode=mode, tau_mem=tau_mem, tau_stdp=tau_stdp, alpha_stdp=alpha_stdp)
        self.controller_rate = controller_rate
        self.ctr_precision = controller_precision
        self.target_rates = torch.tensor(target_rates).float()

    def evolve_controller(self, current_output, control_target_rate):
        error = control_target_rate - current_output
        self.c += self.controller_rate * error

    def evolve_to_convergence(self, x, target_rate, control_target_rate):
        self.reset()
        n_iter = 0

        while True:
            output_rate = self(x.float(), self.c)
            self.evolve_controller(output_rate, control_target_rate)
            n_iter += 1
            if n_iter == 1:
                first_output = output_rate.detach()
            if F.l1_loss(output_rate, target_rate) <= self.ctr_precision:
                break

        return first_output, n_iter

    def training_step(self, data, idx):
        optim = self.optimizers().optimizer

        x, y = data
        target = F.one_hot(y, num_classes=10).squeeze()
        x = x.squeeze()
        control_target_rate = self.target_rates[target]

        # FORWARD, with controller controlling
        first_output, n_iter = self.evolve_to_convergence(
            x, target, control_target_rate)
        optim.step()
        optim.zero_grad()

        ffw_mse = F.mse_loss(first_output, target)
        self.log("ffw_mse_train", ffw_mse)
        self.log("iter_to_target", float(n_iter))

    def validation_step(self, data, idx):
        x, y = data
        target = F.one_hot(y, num_classes=10).squeeze()
        x = x.squeeze()

        self.reset()
        out = self.feedforward(x)
        ffw_acc = torch.equal(out, target)
        self.log("ffw_acc_val", ffw_acc)


class EventControllerNet(ControlledNetwork):
    def __init__(
        self,
        layers,
        batch_size=100,
        tau_mem=10.,
        tau_stdp=False,
        controller_rate=0.1,
        max_val_steps=10,
        max_train_steps=10,
        positive_control=0.03,
        alpha_stdp=1.0,  # ratio A+/A-
    ):
        super().__init__(
            layers=layers, batch_size=batch_size, mode="spiking",
            tau_mem=tau_mem, tau_stdp=tau_stdp,
            alpha_stdp=alpha_stdp)

        self.controller_rate = controller_rate
        self.max_val_steps = max_val_steps
        self.max_train_steps = max_train_steps
        self.positive_control = positive_control

    def evolve_controller(self, current_output, one_hot_target, i):
        # this is -1 when there is a spike and it's NOT on the target
        suppressor = current_output * (one_hot_target - 1)
        update = self.controller_rate * suppressor + self.positive_control * one_hot_target
        self.c.data[:] += (0.97**i) * update

    def evolve_to_convergence(self, x, target):
        self.reset()
        for n_iter in range(min(self.max_train_steps, x.shape[1])):
            output = self(x[:, n_iter], self.c)
            self.evolve_controller(output, target, n_iter)

        return n_iter

    def training_step(self, data, idx):
        optim = self.optimizers().optimizer
        optim.zero_grad()

        x, y = data
        target = F.one_hot(y, num_classes=10)
        x = x.float()
        x = self.ensure_time_dim(x, self.max_train_steps)
        target = target.float()

        # FORWARD, with controller controlling
        n_iter = self.evolve_to_convergence(x, target)
        optim.step()

        self.log("iter_to_target", n_iter)
        optim.zero_grad()

    def validation_step(self, data, idx):
        optim = self.optimizers().optimizer

        x, y = data
        target = F.one_hot(y, num_classes=10)
        x = x.float()
        x = self.ensure_time_dim(x, self.max_val_steps)
        target = target.float()

        self.reset()
        spikes = []
        for i in range(min(self.max_val_steps, x.shape[1])):
            out = self.feedforward(x[:, i]).detach().cpu().numpy()
            spikes.append(out)

        spikes = np.array(spikes)
        rates = spikes.mean(axis=0)
        out_ids = np.argmax(rates, axis=-1)
        correct = float((out_ids == y.detach().cpu().numpy()).mean())

        counts = spikes.sum(axis=-1)
        latency = np.mean([
            np.argwhere(bi).min() for bi in (
                np.concatenate((counts, np.ones((1, self.batch_size)))) > 0
            ).T])

        self.log("acc_val", correct)
        self.log("val_latency", latency)

        optim.zero_grad()

    @staticmethod
    def ensure_time_dim(x, tottime):
        if x.dim() == 2:
            return x.unsqueeze(1).expand((-1, tottime, -1))
        elif x.dim() == 3:
            return x
        else:
            raise ValueError("Anomalous number of input dimensions")
