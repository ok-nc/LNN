"""
This is the module where the Lorentz neural network model is defined.
It uses the nn.Module as a backbone to create the network structure
"""

import math
import numpy as np

# Pytorch module
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import pow, add, mul, div, sqrt, square, cos\
    , sin, conj, abs, tan, log, exp, arctan
from numpy import sqrt as root
from numpy import pi

class LorentzDNN(nn.Module):
    def __init__(self, flags):
        super(LorentzDNN, self).__init__()
        self.flags = flags
        self.kill_osc = False

        # Create the constant for mapping the frequency w
        w_numpy = np.arange(flags.freq_low, flags.freq_high,
                            (flags.freq_high - flags.freq_low) / self.flags.num_spec_points)

        # Create the frequency tensor from numpy array, put variables on cuda if available
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.w = torch.tensor(w_numpy).cuda()
            # self.d = torch.tensor([1.5], requires_grad=True).cuda()
        else:
            self.w = torch.tensor(w_numpy)
            # self.d = torch.tensor([1.5], requires_grad=True)

        """
        General layer definitions:
        """
        # Linear Layer and Batch_norm Layer definitions here
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear[0:-1]):               # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1], bias=True))
            self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1], track_running_stats=True, affine=True))

        # self.dropout = nn.Dropout(0.03)           # Can experiment with adding dropout layer
        layer_size = flags.linear[-1]               # Last linear layer size to match to Lorentz layer

        # Last layer is the Lorentzian parameter layer. Four parameters are needed to describe each oscillator for
        # both the relative permittivity (Epsilon) and the permeability (Mu). A single bias is used for each constant
        # term eps_infinity and mu_infinity.
        self.eps_w0 = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.eps_wp = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.eps_g = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.eps_inf = nn.Linear(layer_size, 1, bias=True)
        self.mu_w0 = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.mu_wp = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.mu_g = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.mu_inf = nn.Linear(layer_size, 1, bias=True)

    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The output spectra, with dimension specified in the hyperparameters
        """
        out = G
        batch_size = out.size()[0]

        self.geom = G
        # For the linear layers, leaky relu seems to work better than regular relu for LNN training
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            if ind < len(self.linears) - 0:
            # if ind != len(self.linears) - 1:
                out = F.leaky_relu_(bn(fc(out)))                                   # ReLU + BN + Linear
                # out = self.dropout(out)
            else:
                out = fc(out)

        e_w0 = F.leaky_relu(self.eps_w0(F.leaky_relu(out)))     # Epsilon resonance frequency w0
        e_wp = F.leaky_relu(self.eps_wp(F.leaky_relu(out)))     # Epsilon oscillator strength wp
        e_g = F.leaky_relu(self.eps_g(F.leaky_relu(out)))       # Epsilon damping/linewidth gamma
        e_inf = F.leaky_relu(self.eps_inf(F.leaky_relu(out)))   # Epsilon infinity

        m_w0 = F.leaky_relu(self.mu_w0(F.leaky_relu(out)))      # Mu resonance frequency w0
        m_wp = F.leaky_relu(self.mu_wp(F.leaky_relu(out)))      # Mu oscillator strength wp
        m_g = F.leaky_relu(self.mu_g(F.leaky_relu(out)))        # Mu damping/linewidth gamma
        m_inf = F.leaky_relu(self.mu_inf(F.leaky_relu(out)))    # Mu infinity

        self.eps_params_out = [e_w0, e_wp, e_g, e_inf]          # For easy parameter extraction from model
        self.mu_params_out = [m_w0, m_wp, m_g, m_inf]

        # Expand them to parallelize, (batch_size, # oscillators, # spectrum points)
        e_w0 = e_w0.unsqueeze(2).expand(batch_size, self.flags.num_lorentz_osc, self.flags.num_spec_points)
        e_wp = e_wp.unsqueeze(2).expand_as(e_w0)
        e_g = e_g.unsqueeze(2).expand_as(e_w0)

        m_w0 = m_w0.unsqueeze(2).expand_as(e_w0)
        m_wp = m_wp.unsqueeze(2).expand_as(e_w0)
        m_g = m_g.unsqueeze(2).expand_as(e_w0)

        w_expand = self.w.expand_as(e_w0)       # This frequency variable is for the expanded multi-oscillator array
        w_2 = self.w.expand(batch_size,self.flags.num_spec_points) # This freq. variable is for single calculations

        # Define dielectric functions using Lorentzian oscillator parameters (take abs value, since leaky relu can
        # give negative values
        e1, e2 = lorentzian(w_expand, abs(e_w0), abs(e_wp), abs(e_g))
        mu1, mu2 = lorentzian(w_expand, abs(m_w0), abs(m_wp), abs(m_g))

        if self.kill_osc:       # if oscillators parameters are very small after some time, send them to zero

            for o in range(self.flags.num_lorentz_osc):
                if torch.max(e2[:,o,:]) < 1:
                    # print('Zeroing osc: '+str(o))
                    e2[:, o, :] = torch.zeros_like(e2[:, o, :])
                    e1[:, o, :] = torch.zeros_like(e1[:, o, :])
                if torch.max(mu2[:,o,:]) < 1:
                    # print('Zeroing osc: '+str(o))
                    mu2[:, o, :] = torch.zeros_like(mu2[:, o, :])
                    mu1[:, o, :] = torch.zeros_like(mu1[:, o, :])

        # Combine all oscillators into frequency-dependent eps and mu functions
        e1 = torch.sum(e1, 1).type(torch.cfloat)
        e2 = torch.sum(e2, 1).type(torch.cfloat)
        eps_inf = e_inf.expand_as(e1).type(torch.cfloat)
        e1 += 1+abs(eps_inf)
        mu1 = torch.sum(mu1, 1).type(torch.cfloat)
        mu2 = torch.sum(mu2, 1).type(torch.cfloat)
        mu_inf = m_inf.expand_as(mu1).type(torch.cfloat)
        mu1 += 1+abs(mu_inf)
        j = torch.tensor([0+1j],dtype=torch.cfloat).expand_as(e2)
        if torch.cuda.is_available():
            j = j.cuda()

        eps = add(e1, mul(e2,j))
        mu = add(mu1, mul(mu2, j))

        # Thickness here is pre-defined as height (an input parameter), but can make it a trainable parameter as well
        d_in = G[:, 1]      # Make sure correct input parameter is selected here. May change with geometry design.
        if self.flags.normalize_input:      # Make sure correct geoboundary values are used to de-normalize
            d_in = d_in * 0.5 * (self.flags.geoboundary[5]-self.flags.geoboundary[1]) + (self.flags.geoboundary[5]+self.flags.geoboundary[1]) * 0.5
        self.d_out = d_in
        d = d_in.unsqueeze(1).expand_as(eps)

        # # Spatial dispersion calculation for magnetodielectric slab
        theta = 2 * arctan(0.0033 * pi * w_2 * d * sqrt(mul(eps, mu)))
        adv = div(0.5 * theta, tan(0.5 * theta))                        # Phase advance
        eps_eff = mul(adv, eps)
        mu_eff = mul(adv, mu)
        n_eff = sqrt(mul(eps_eff, mu_eff))
        n = n_eff.real + 1j * abs(n_eff.imag)    # Causality constraint
        z_eff = sqrt(div(mu_eff, eps_eff))
        z = abs(z_eff.real) + 1j * z_eff.imag    # Causality constraint

        self.eps_out = eps
        self.mu_out = mu
        self.eps_eff_out = eps_eff
        self.mu_eff_out = mu_eff
        self.n_out = n
        self.theta_out = theta
        self.adv_out = adv

        r, t, = transfer_matrix(n, z, d, w_2)

        return r, t


def lorentzian(w, w0, wp , g, eps_inf=0):
    num1 = mul(square(wp), add(square(w0), -square(w)))
    num2 = mul(square(wp), mul(w, g))
    denom = add(square(add(square(w0), -square(w))), mul(square(w), square(g)))
    e1 = div(num1, denom + 1e-5)
    e2 = div(num2, denom + 1e-5)
    e1 += eps_inf
    return e1,e2

# Calculates scattering parameters using transfer matrix equations for a homogeneous slab with eps, mu
def transfer_matrix(n,z,d,f):
    c = 3e8                                 # Speed of light
    e0 = (10 ** 7) / (4 * pi * c ** 2)      # Permittivity of free space
    m0 = 4 * pi * 10 ** (-7)                # Permeability of free space
    z0 = root(m0 / e0)                      # Impedance of free space
    d = d * 1e-6                            # Thickness specified in microns
    w = 2 * pi * f * 1e12                   # Angular frequency in units of THz
    k0 = w / c                              # Free space wave vector

    k = div(mul(w, n), c)                   # Wave vector

    M12_TE = 0.5 * 1j * mul((z - div(1, z + 1e-5)), (sin(mul(k, d))))
    M22_TE = cos(mul(k, d)) - 0.5 * 1j * mul((z + div(1, z + 1e-5)), (sin(mul(k, d))))

    # Reflection and transmission coefficients
    r = div(M12_TE,M22_TE + 1e-5)           # Added small number to prevent denominator blowing up
    t = div(1, M22_TE + 1e-5)

    return r,t