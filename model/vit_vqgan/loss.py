import torch
from torch import nn
from torch.nn import functional as F

def adopt_weight(weight, global_step, threshold=0, value=0.):
    return weight if global_step >= threshold else value

def hinge_d_loss(logits_real, logits_fake):
    return 0.5 * (torch.mean(F.relu(1. - logits_real)) + torch.mean(F.relu(1. + logits_fake)))

def vanilla_d_loss(logits_real, logits_fake):
    return 0.5 * (torch.mean(F.softplus(-logits_real)) + torch.mean(F.softplus(logits_fake)))

def hinge_g_loss(logits_fake): return -torch.mean(logits_fake)

def vanilla_g_loss(logits_fake): return torch.mean(F.softplus(-logits_fake))

def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        norm_layer = nn.BatchNorm2d
        use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*sequence)

    def forward(self, x):
        return self.main(x)


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, 
                 disc_start: int, 
                 disc_weight: float,
                 disc_num_layers: int, 
                 disc_in_channels: int, 
                 disc_ndf: int, 
                 disc_loss_type: str = "hinge",
                 disc_conditional: bool = False,
                 disc_factor: float = 1.0
                 ):
        super().__init__()
        
        if disc_loss_type not in ["hinge", "vanilla"]:
            raise ValueError(f"Unknown GAN loss type '{disc_loss_type}'.")

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels, 
            n_layers=disc_num_layers, 
            ndf=disc_ndf
        ).apply(weights_init)
        
        self.discriminator_iter_start = disc_start
        
        if disc_loss_type == "hinge":
            self.disc_loss_fn = hinge_d_loss
            self.gen_loss_fn = hinge_g_loss
        elif disc_loss_type == "vanilla":
            self.disc_loss_fn = vanilla_d_loss
            self.gen_loss_fn = vanilla_g_loss
        
        print(f"VQLPIPSWithDiscriminator running with {disc_loss_type} loss.")
        
        self.disc_factor = disc_factor
        self.adaptive_disc_weight_factor = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer_params):
        if not last_layer_params:
            return torch.tensor(0.0, device=nll_loss.device)

        nll_grads = torch.autograd.grad(nll_loss, last_layer_params, retain_graph=True, allow_unused=True)
        g_grads = torch.autograd.grad(g_loss, last_layer_params, retain_graph=True, allow_unused=True)

        nll_grads_filtered = [g for g in nll_grads if g is not None]
        g_grads_filtered = [g for g in g_grads if g is not None]

        if not nll_grads_filtered or not g_grads_filtered:
            return torch.tensor(0.0, device=nll_loss.device)

        norm_nll_grads = torch.norm(torch.cat([g.flatten() for g in nll_grads_filtered]))
        norm_g_grads = torch.norm(torch.cat([g.flatten() for g in g_grads_filtered]))
        
        d_weight = norm_nll_grads / (norm_g_grads + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.adaptive_disc_weight_factor
        return d_weight

    def forward(self, inputs, reconstructions, optimizer_idx, global_step, last_layer_params=None, cond=None, split="train"):
        rec_loss_values = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        nll_loss = torch.mean(rec_loss_values)

        # GAN part
        if optimizer_idx == 0:
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            
            g_loss = self.gen_loss_fn(logits_fake)

            if last_layer_params:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer_params=last_layer_params)
                except RuntimeError as e:
                    if not self.training:
                        d_weight = torch.tensor(0.0, device=inputs.device)
                    else:
                        print(f"RuntimeError calculating adaptive weight: {e}. Defaulting d_weight to 0.")
                        d_weight = torch.tensor(0.0, device=inputs.device)
            else:
                d_weight = torch.tensor(0.0, device=inputs.device)

            current_disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            
            # Total generator loss
            loss = nll_loss + d_weight * current_disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(current_disc_factor, device=inputs.device),
                   "{}/g_loss".format(split): g_loss.detach().mean(),}
            return loss, log

        # Discriminator part
        if optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            current_disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = current_disc_factor * self.disc_loss_fn(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()}
            return d_loss, log