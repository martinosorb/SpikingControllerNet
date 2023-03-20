import torch
import torch.nn.functional as F
import pytorch_lightning as pl


def spikify(rate):
    return (rate > torch.rand_like(rate)) * 1.


class ControlledLayer(torch.nn.Module):
    def __init__(
        self,
        fan_in,
        fan_out,
        controller_dim,
        mode="spiking",
        tau_mem=10.,
        tau_stdp=False,
        alpha_stdp=1.0,  # ratio A+/A-
    ):
        super().__init__()

        self.leak = 1. / tau_mem
        self.threshold = 1.
        self.fan_out = fan_out
        self.ff = torch.nn.Linear(fan_in, fan_out, bias=False)
        self.fb = torch.nn.Linear(controller_dim, fan_out, bias=False)
        self.reset()

        assert mode == "spiking" or mode == "rate"
        self.mode = mode
        self.dynamics = self._spiking_dynamics if mode == "spiking" else self._rate_dynamics

        self.stdp_decay = 1 - 1 / tau_stdp if tau_stdp else False
        if self.stdp_decay:
            self.Apre = torch.zeros(fan_in)
            self.Apost = torch.zeros(fan_out)
            self.neg_stdp_amplitude = 2. / (1 + alpha_stdp) / tau_stdp
            self.pos_stdp_amplitude = alpha_stdp * self.neg_stdp_amplitude

    def forward(self, inputs, c):
        ff_input = self.ff(inputs)
        fb_input = self.fb(c)

        # LIF dynamics
        self.v += -self.leak * self.v + ff_input + fb_input
        outputs = self.dynamics()

        if self.stdp_decay:
            if self.mode == "rate":
                input_spikes = spikify(inputs)
                output_spikes = spikify(outputs)
            else:
                input_spikes, output_spikes = inputs, outputs
            self.Apre = self.Apre * self.stdp_decay + input_spikes.float()
            self.Apost = self.Apost * self.stdp_decay + output_spikes.float()
            self.ff.weight.grad -= -self.neg_stdp_amplitude * torch.outer(input_spikes, self.Apost).T
            self.ff.weight.grad -= +self.pos_stdp_amplitude * torch.outer(output_spikes, self.Apre)

        return outputs

    def _spiking_dynamics(self):
        spikes = self.v > self.threshold
        self.v[spikes] = 0.

        return spikes.float()

    def _rate_dynamics(self):
        return torch.sigmoid(self.v)

    def reset(self):
        if self.ff.weight.grad is None:
            self.ff.weight.grad = torch.zeros_like(self.ff.weight)
        self.v = torch.zeros(self.fan_out)

    @property
    def grad(self):
        return self.ff.weight.grad


class ControlledNetwork(pl.LightningModule):
    def __init__(
        self,
        layers,
        mode="spiking",
        tau_mem=10.,
        tau_stdp=False,
        alpha_stdp=1.0,  # ratio A+/A-
    ):
        super().__init__()

        self.layers = []
        controller_dim = layers[-1]
        self.c = torch.zeros(controller_dim)

        for fan_in, fan_out in zip(layers[:-1], layers[1:]):
            layer = ControlledLayer(
                fan_in, fan_out, controller_dim=controller_dim, mode=mode,
                tau_mem=tau_mem, tau_stdp=tau_stdp, alpha_stdp=alpha_stdp)
            self.layers.append(layer)

        self.initialize_as_dfc()
        self.seq = torch.nn.Sequential(*self.layers)  # so that pytorch registers them

    def evolve_controller(self, current_output, control_target_rate):
        raise NotImplementedError()

    def initialize_as_dfc(self):
        # last layer has Q=identity
        curr_w = torch.eye(len(self.c))
        with torch.no_grad():
            for layer in self.layers[::-1]:  # layers backwards
                layer.fb.weight.data = curr_w
                curr_w = layer.ff.weight.T @ curr_w

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
        tau_mem=10.,
        tau_stdp=False,
        controller_rate=0.1,
        max_val_steps=10,
        max_train_steps=10,
        positive_control=0.03,
        alpha_stdp=1.0,  # ratio A+/A-
        min_spikes_on_target=2,
    ):
        super().__init__(
            layers=layers, mode="spiking",
            tau_mem=tau_mem, tau_stdp=tau_stdp,
            alpha_stdp=alpha_stdp)

        self.controller_rate = controller_rate
        self.max_val_steps = max_val_steps
        self.max_train_steps = max_train_steps
        self.positive_control = positive_control
        self.min_spikes_on_target = min_spikes_on_target

    def evolve_controller(self, current_output, one_hot_target):
        # this is -1 when there is a spike and it's NOT on the target
        suppressor = current_output * (one_hot_target - 1)
        self.c += self.controller_rate * suppressor + self.positive_control * one_hot_target

    def evolve_to_convergence(self, x, target, record=False):
        self.reset()
        spikes_on_target = 0
        outputs, contr = [], []

        for n_iter in range(self.max_train_steps):
            output = self(x.float(), self.c)
            self.evolve_controller(output, target)
            n_output_spikes = output.sum()
            if record:
                outputs.append(output.numpy())
                contr.append(self.c.clone().numpy())

            if n_output_spikes == 0: continue

            # controller stop condition
            if torch.equal(output, target):
                if spikes_on_target >= self.min_spikes_on_target: break
                spikes_on_target += 1
            else:
                spikes_on_target = 0

        if record:
            last_dw = -self.layers[-1].ff.weight.grad.clone().detach().numpy()
            return outputs, contr, last_dw
        return n_iter

    def training_step(self, data, idx):
        optim = self.optimizers().optimizer

        x, y = data
        target = F.one_hot(y, num_classes=10).squeeze()
        x = x.squeeze()

        # FORWARD, with controller controlling
        n_iter = self.evolve_to_convergence(x, target)
        optim.step()
        optim.zero_grad()

        self.log("iter_to_target", float(n_iter))

    def validation_step(self, data, idx):
        x, y = data
        target = F.one_hot(y, num_classes=10).squeeze()
        x = x.squeeze()

        # TTFS evaluation
        self.reset()
        correct = False
        for i in range(self.max_val_steps):
            out = self.feedforward(x)
            if out.sum() > 0:  # This must be changed in case of batches
                correct = torch.equal(out, target)
                break  # if there are any spikes at all

        self.log("ttfs_acc_val", correct)
        self.log("val_latency", i)
