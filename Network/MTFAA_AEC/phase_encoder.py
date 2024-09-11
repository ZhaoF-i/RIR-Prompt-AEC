"""
Phase Encoder (PE).

shmzhang@aslp-npu.org, 2022
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=1,
        groups=1,
        causal=True,
        complex_axis=1,
    ):
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels//2
        self.out_channels = out_channels//2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = causal
        self.groups = groups
        self.dilation = dilation
        self.complex_axis = complex_axis
        self.real_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride, padding=[
                                   self.padding[0], 0], dilation=self.dilation, groups=self.groups, bias=False)
        self.imag_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride, padding=[
                                   self.padding[0], 0], dilation=self.dilation, groups=self.groups, bias=False)

        nn.init.normal_(self.real_conv.weight.data + 1e-10, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data + 1e-10, std=0.05)
        # nn.init.constant_(self.real_conv.bias, 1e-10)
        # nn.init.constant_(self.imag_conv.bias, 1e-10)

    def forward(self, inputs):
        if self.padding[1] != 0 and self.causal:
            inputs = F.pad(inputs, [self.padding[1], 0, 0, 0])
        else:
            inputs = F.pad(inputs, [self.padding[1], self.padding[1], 0, 0])

        if self.complex_axis == 0:
            real = self.real_conv(inputs + 1e-10)
            imag = self.imag_conv(inputs + 1e-10)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)

        else:
            # if isinstance(inputs, torch.Tensor):
            #     real, imag = torch.chunk(inputs, 2, self.complex_axis)
            #     if (torch.isnan(real).sum() > 0):
            #         print("  here is nan")
            #     if (torch.isnan(imag).sum() > 0):
            #         print("  here is nan")
            real, imag = torch.chunk(inputs, 2, self.complex_axis)
            if (torch.isnan(real).sum() > 0):
                print("  here is nan")
            if (torch.isnan(imag).sum() > 0):
                print("  here is nan")

            real2real = self.real_conv(real + 1e-10,)
            imag2imag = self.imag_conv(imag + 1e-10,)

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        if (torch.isnan(real).sum() > 0):
            print("  here is nan")
        imag = real2imag + imag2real
        if (torch.isnan(real).sum() > 0):
            print("  here is nan")
        out = torch.cat([real, imag], self.complex_axis)
        if (torch.isnan(out).sum() > 0):
            print("  here is nan")
        return out


def complex_cat(inps, dim=1):
    reals, imags = [], []
    for inp in inps:
        real, imag = inp.chunk(2, dim)
        reals.append(real)
        imags.append(imag)
    reals = torch.cat(reals, dim)
    if (torch.isnan(reals).sum() > 0):
        print("  here is nan")
    imags = torch.cat(imags, dim)
    if (torch.isnan(imags).sum() > 0):
        print("  here is nan")
    return reals, imags


class ComplexLinearProjection(nn.Module):
    def __init__(self, cin):
        super(ComplexLinearProjection, self).__init__()
        self.clp = ComplexConv2d(cin, cin)

    def forward(self, real, imag):
        """
        real, imag: B C F T
        """
        inputs = torch.cat([real, imag], 1)
        outputs = self.clp(inputs)
        if (torch.isnan(outputs).sum() > 0):
            print("  here is nan")
        real, imag = outputs.chunk(2, dim=1)
        outputs = torch.sqrt(real**2+imag**2+1e-8)
        if (torch.isnan(outputs).sum() > 0):
            print("  here is nan")
        return outputs


class PhaseEncoder(nn.Module):
    def __init__(self, cout, n_sig, cin=2, alpha=0.5):
        super(PhaseEncoder, self).__init__()
        self.complexnn = nn.ModuleList()
        for _ in range(n_sig):
            self.complexnn.append(
                nn.Sequential(
                    nn.ConstantPad2d((2, 0, 0, 0), 0.0),
                    ComplexConv2d(cin, cout, (1, 3))
                )
            )
        self.clp = ComplexLinearProjection(cout*n_sig)
        self.alpha = alpha

    def forward(self, cspecs):
        """
        cspec: B C F T
        """
        outs = []
        for idx, layer in enumerate(self.complexnn):
            outs.append(layer(cspecs[idx]))
            if (torch.isnan(layer(cspecs[idx])).sum() > 0):
                print("  here is nan")
        real, imag = complex_cat(outs, dim=1)
        amp = self.clp(real, imag)
        if (torch.isnan(amp).sum() > 0):
            print("  here is nan")
        return amp**self.alpha


if __name__ == "__main__":
    net = PhaseEncoder(cout=4, n_sig=1)
    # 32ms@48kHz, concatenation of [real, imag], dim=1
    inps = torch.randn(3, 2, 769, 126)
    outs = net([inps])
    print(outs.shape)