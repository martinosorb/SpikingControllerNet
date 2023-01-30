"""
Created on Fri Nov 25 15:42:47 2022

@author: pau & sander

"""

import torch

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class STDP(metaclass=Singleton):

    def __init__(self, timesteps_ff=500, timesteps_burst=500, device='cpu'):
        self._timesteps_ff = timesteps_ff
        self._timesteps_burst = timesteps_burst
        self._device = device

        self._timesteps_base = timesteps_ff + timesteps_burst
        self._decay_rate = torch.exp(torch.tensor([- 2 / self._timesteps_base], device=device))

    def _STDP_compute_and_filter_prespikes(self, pre_rates):
        shape = (_, n, time) = (pre_rates.size(dim=0), pre_rates.size(dim=1), self._timesteps_base)

        pre_spikes = (torch.rand(shape, device=self._device) < pre_rates.unsqueeze(2).repeat(1, 1, time))*1
        filtered_spikes = torch.zeros(shape, device=self._device)
        v_fwrd, v_bwrd = torch.zeros(n, device=self._device), torch.zeros(n, device=self._device)

        for t in range(time):
            v_fwrd = v_fwrd * self._decay_rate + pre_spikes[:,:,t]
            v_bwrd = v_bwrd * self._decay_rate + pre_spikes[:,:,time-t-1]
            filtered_spikes[:,:,t] += v_fwrd
            filtered_spikes[:,:,time-t-1] -= v_bwrd
        
        return filtered_spikes.double()

    def _STDP_compute_postspikes(self, post_ff_rates, binary_feedback):
        spikes_ff = (torch.rand(post_ff_rates.size(dim=0), post_ff_rates.size(dim=1), self._timesteps_ff, device=self._device) < post_ff_rates.unsqueeze(2).repeat(1, 1, self._timesteps_ff))*1
        # spikes_ff = torch.rand(post_ff_rates.size(dim=1), self._timesteps_ff) < torch.tile(post_ff_rates, (self._timesteps_ff, 1)).T
        spikes_fb = binary_feedback.unsqueeze(2).repeat(1, 1, self._timesteps_burst)
        # spikes_fb = torch.tile(binary_feedback, (self._timesteps_burst, 1)).T

        post_spikes = torch.cat((spikes_ff, spikes_fb), dim=2)

        return post_spikes.double()
        
    def learn_STDP_burst(self, pre_rates, post_ff_rates, binary_feedback):  # former STDP function
        pre_spikes = self._STDP_compute_and_filter_prespikes(pre_rates)
        post_spikes = self._STDP_compute_postspikes(post_ff_rates, binary_feedback)

        dW = torch.bmm(pre_spikes, torch.transpose(post_spikes, 1, 2)) / self._timesteps_base
        
        return dW
