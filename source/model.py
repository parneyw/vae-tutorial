"""
Implementation of a Variational Autoencoder.
[Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114), Kingma & Welling.
"""
from dataclasses import dataclass
import math
from typing import Union
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class VAEOutput:
    """
    Dataclass for VAE output.

    Attributes:
        z_dist (torch.distributions.Distribution): Latent space probability distribution.
        z_sample (torch.Tensor): Sample from z_dist.
        x_recon (torch.Tensor): Reconstruction of the input.
        loss (torch.Tensor): loss_recon + loss_kl.
        loss_recon (torch.Tensor): Reconstruction loss; P(x|z).
        loss_kl (torch.Tensor): KL divergence of z_dist from standard multivariate normal; how different they are.
    """
    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_recon: Union[torch.Tensor, torch.distributions.Distribution]

    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor

@dataclass
class VAEConfig:
    """
    Dataclass for VAE hyperparameter configuration.

    Attributes:
        input_dim (int): Dimensionality of input.
        hidden_dim (int): Dimensionality of hidden layers.
        latent_dim (int): Dimensionality of latent space.
        depth (int): Number of hidden layers, each with half the nodes of the preceding layer.
        act_fn (FunctionType): Activation function for hidden layers.
    """
    input_dim: int
    hidden_dim: int
    latent_dim: int
    depth: int
    act_fn: nn.Module


class EncoderMLP(nn.Module):
    """
    VAE Encoder MLP, ``depth`` layers each shrinking by a factor of 2.

    Attributes:
        hidden_dim (int): Dimensionality of the first hidden layer.
    """
    def __init__(self, config: VAEConfig) -> None:
        super(EncoderMLP, self).__init__()
        assert(config.hidden_dim//2**(config.depth-1) >= 1)
        if config.depth == 1:
            self.mlp = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dim),
                config.act_fn,
                nn.Linear(config.hidden_dim, 2*config.latent_dim)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dim),
                config.act_fn,
                nn.Sequential(*[
                    nn.Sequential(
                        nn.Linear(config.hidden_dim//2**(i-1), config.hidden_dim//2**i),
                        config.act_fn,
                    )
                    for i in range(1, config.depth)
                ]),
                nn.Linear(config.hidden_dim//2**(config.depth-1), 2*config.latent_dim)
            )

    def forward(self, x) -> torch.Tensor:
        """ Compute bias and log-variance for distribution of latents. """
        return self.mlp(x)


class GaussianEncoder(nn.Module):
    """ VAE Gaussian Encoder (Appendix C.2). """
    def __init__(self, config: VAEConfig) -> None:
        super(GaussianEncoder, self).__init__()
        self.mlp = EncoderMLP(config)
        

    def forward(self, x) -> torch.distributions.Distribution:
        """ Compute distribution of latents for given input x. """
        x = self.mlp(x)
        bias, logvar = torch.chunk(x, 2, dim=-1)
        var = torch.exp(logvar)
        scale_tril = torch.diag_embed(var)
        z_dist = torch.distributions.MultivariateNormal(loc=bias, scale_tril=scale_tril)
        return z_dist


class DecoderMLP(nn.Module):
    """ 
    Generic class for VAE Decoder MLP with ``depth`` layers each growing by a factor of 2.

    Attributes:
        hidden_dim (int): Dimensionality of the last hidden layer.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, depth, act_fn):
        super(DecoderMLP, self).__init__()
        assert(hidden_dim//2**(depth-1) >= 1)
        if depth == 1:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                act_fn,
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim//2**(depth-1)),
                act_fn,
                nn.Sequential(*[
                    nn.Sequential(
                        nn.Linear(hidden_dim//2**(i+1), hidden_dim//2**i),
                        act_fn,
                    )
                    for i in range(depth-2, -1, -1)
                ]),
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x) -> torch.Tensor:
        """ Compute bias and log-variance for distribution of latents. """
        return self.mlp(x)


class BernoulliDecoder(nn.Module):
    """ VAE Bernoulli Decoder (Appendix C.1). """
    def __init__(self, config: VAEConfig) -> None:
        super(BernoulliDecoder, self).__init__()
        self.mlp = DecoderMLP(config.latent_dim, config.hidden_dim, config.input_dim,
                              config.depth, config.act_fn)


    def forward(self, z) -> torch.Tensor:
        """ Compute reconstructed input from latent z. """
        z = self.mlp(z)
        z = F.sigmoid(z)
        return z


class GaussianDecoder(nn.Module):
    """ VAE Gaussian Encoder (Appendix C.2). """
    def __init__(self, config: VAEConfig) -> None:
        super(GaussianDecoder, self).__init__()
        self.mlp = DecoderMLP(config.latent_dim, config.hidden_dim, 2*config.input_dim,
                              config.depth, config.act_fn)

    def forward(self, z) -> torch.distributions.Distribution:
        """ Compute distribution of latents for given input x. """
        z = self.mlp(z)
        bias, logvar = torch.chunk(z, 2, dim=-1)
        var = torch.exp(logvar) # todo: change to softplus
        scale_tril = torch.diag_embed(var)
        z_dist = torch.distributions.MultivariateNormal(loc=bias, scale_tril=scale_tril)
        return z_dist


class BVAE(nn.Module):
    """ Variational Autoencoder for binary data (Section 3). """
    def __init__(self, config: VAEConfig) -> None:
        super(BVAE, self).__init__()
        self.encoder = GaussianEncoder(config)
        self.decoder = BernoulliDecoder(config)

    def encode(self, x) -> torch.distributions.Distribution:
        return self.encoder(x)

    def decode(self, z) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x, bias_correction = None, compute_loss: bool = True) -> VAEOutput:
        """
        Compute reconstructed input and compute loss.

        x (torch.Tensor): Input data (may be adjusted as for MNIST x = x_true-0.5).
        bias_correction (torch.Tensor): Bias correction term; required to be set if compute_loss==True.
        compute_loss (bool): Flag indicating whether to compute loss (for training) or not (for inference).
        """
        z_dist = self.encode(x)
        z_sample = z_dist.rsample()
        x_recon = self.decoder(z_sample)

        if not compute_loss:
            return VAEOutput(
                z_dist=z_dist,
                z_sample=z_sample,
                x_recon=x_recon,
                loss=None,
                loss_recon=None,
                loss_kl=None,
            )
        
        assert(bias_correction is not None) # Give explicit bias correction term (for MNIST or other datasets)

        # Compute reconstruction loss: how different is the reconstruction from the input?
        loss_recon = F.binary_cross_entropy(x_recon, x + bias_correction, reduction="none").sum(-1).mean()
        std_normal = torch.distributions.MultivariateNormal( # our standard Gaussian prior
            torch.zeros_like(z_sample, device=z_sample.device),
            scale_tril=torch.diag_embed(torch.ones_like(z_sample, device=z_sample.device)),
        )
        loss_kl = torch.distributions.kl.kl_divergence(z_dist, std_normal).mean()

        loss = loss_recon + loss_kl

        return VAEOutput(
            z_dist=z_dist,
            z_sample=z_sample,
            x_recon=x_recon,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )
    


class SGVAE(nn.Module):
    """ Stochatstic Gaussian Variational Autoencoder (Gaussian decoder). """
    def __init__(self, config: VAEConfig) -> None:
        super(SGVAE, self).__init__()
        self.encoder = GaussianEncoder(config)
        self.decoder = GaussianDecoder(config)

    def encode(self, x) -> torch.distributions.Distribution:
        return self.encoder(x)

    def decode(self, z) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x, compute_loss: bool = True) -> VAEOutput:
        """
        Compute reconstructed input and compute loss.

        x (torch.Tensor): Input data (may be adjusted as for MNIST x = x_true-0.5).
        x_true (torch.Tensor): Input data
        """
        z_dist = self.encoder(x)
        z_sample = z_dist.rsample()
        x_recon_dist = self.decoder(z_sample)

        if not compute_loss:
            return VAEOutput(
                z_dist=z_dist,
                z_sample=z_sample,
                x_recon=x_recon_dist,
                loss=None,
                loss_recon=None,
                loss_kl=None,
            )

        # Compute reconstruction loss: how different is the reconstruction from the input?
        bias, var = x_recon_dist.mean, x_recon_dist.variance
        loss_recon = F.gaussian_nll_loss(bias, x, var, reduction="none").sum(-1).mean()
        std_normal = torch.distributions.MultivariateNormal( # our standard Gaussian prior
            torch.zeros_like(z_sample, device=z_sample.device),
            scale_tril=torch.diag_embed(torch.ones_like(z_sample, device=z_sample.device)),
        )
        loss_kl = torch.distributions.kl.kl_divergence(z_dist, std_normal).mean()

        loss = loss_recon + loss_kl

        return VAEOutput(
            z_dist=z_dist,
            z_sample=z_sample,
            x_recon=x_recon_dist,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )