"""
This is the module where the model is defined. It uses the nn.Module as a backbone to create the network structure
"""

# Pytorch modules to import
import torch.nn as nn
import torch.nn.functional as F
import torch

class Forward(nn.Module):
    def __init__(self, flags):
        super(Forward, self).__init__()

        self.use_conv = flags.use_conv
        self.flags = flags

        """
        General layer definitions:
        """
        # Linear Layer and Batch_norm Layer definitions here
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear[0:-1]):  # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1]))
            self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1]))

        # Convolutional Layer definitions here
        self.convs = nn.ModuleList([])
        in_channel = 1  # Initialize the in_channel number
        for ind, (out_channel, kernel_size, stride) in enumerate(zip(flags.conv_out_channel,
                                                                     flags.conv_kernel_size,
                                                                     flags.conv_stride)):
            if stride == 2:  # We want to double the number
                pad = int(kernel_size / 2 - 1)
            elif stride == 1:  # We want to keep the number unchanged
                pad = int((kernel_size - 1) / 2)
            else:
                Exception("Now only supports stride = 1 or 2, contact Ben")

            self.convs.append(nn.ConvTranspose1d(in_channel, out_channel, kernel_size,
                                                 stride=stride, padding=pad))  # To make sure L_out doubles each time
            in_channel = out_channel  # Update the out_channel

        self.convs.append(nn.Conv1d(in_channel, out_channels=1, kernel_size=1, stride=1, padding=0))

    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The output spectra (typically dimension 300-1000)
        """
        out = G

        # initialize the out
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            if ind != len(self.linears) - 1:
                out = F.relu(bn(fc(out)))  # ReLU + BN + Linear
            else:
                out = fc(out)
        out = torch.tanh(out)

        if self.use_conv:
            out = out.unsqueeze(1)  # Add 1 dimension to get N,L_in,
            # For the conv part
            for ind, conv in enumerate(self.convs):
                out = conv(out)

            # Final touch, because the input is normalized to [-1,1]
            # out = torch.tanh(out.squeeze())
            out = out.squeeze()
        return out
