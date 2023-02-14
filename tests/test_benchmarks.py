import pytest

from trainable import Trainable, TrainableHParams
from experiments.benchmarks import *

import torch
import torch.distributions as D

import matplotlib.pyplot as plt

import FrEIA.framework as ff
import FrEIA.modules as fm
from .subnet_factory import SubnetFactory


def test_circles():
    ds = CirclesDataset()

    samples = ds.distribution.sample((1000,)).numpy()

    plt.scatter(*samples.T)
    plt.title("Circles Distribution")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("figures/circles.png")
    plt.show()


def test_moons():
    ds = MoonsDataset()

    samples = ds.distribution.sample((1000,)).numpy()

    plt.scatter(*samples.T)
    plt.title("Moons Distribution")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("figures/moons.png")
    plt.show()


def test_hypershells():
    ds = HypershellsDataset(shells=3, batch_size=32)

    samples = ds.distribution.sample((1000,)).numpy()

    plt.scatter(*samples.T)
    for radius in ds.distribution._component_distribution.radii:
        circle = plt.Circle(xy=(0, 0), radius=radius, color="red", fill=False, lw=1)
        plt.gca().add_patch(circle)

    plt.title("Hypershells Distribution")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("figures/hypershells.png")
    plt.show()


def test_hypersphere_mixture():
    ds = HypersphereMixtureDataset()

    samples = ds.distribution.sample((1000,)).numpy()

    plt.scatter(*samples.T)
    means = ds.distribution._component_distribution.base_dist.loc
    plt.scatter(*means.T, color="red", marker="x")
    plt.title("Hyperspheres Distribution")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("figures/hypersphere_mixture.png")
    plt.show()


@pytest.mark.slow
def test_benchmarks():

    class GenerativeHParams(TrainableHParams):
        loss: str = "negative_log_likelihood"

        inputs: int
        layers: int
        bins: int = 32
        subnet_widths: list
        activation: str = "relu"

    class GenerativeModel(Trainable):
        """ Example Generative Model as a Normalizing Flow """
        def __init__(self, hparams, **kwargs):
            if not isinstance(hparams, GenerativeHParams):
                hparams = GenerativeHParams(**hparams)
            super().__init__(hparams, **kwargs)

            self.flow = ff.SequenceINN(self.hparams.inputs)
            subnet_constructor = SubnetFactory(kind="dense", widths=self.hparams.subnet_widths, activation=self.hparams.activation)

            for _ in range(self.hparams.layers):
                self.flow.append(
                    fm.RationalQuadraticSpline, subnet_constructor=subnet_constructor, bins=self.hparams.bins
                )

            self.latent_distribution = D.Independent(D.Normal(torch.zeros(self.hparams.inputs), torch.ones(self.hparams.inputs)), 1)

        def compute_metrics(self, batch, batch_idx) -> dict:
            z, log_jac_det = self.flow(batch)
            log_prob = self.latent_distribution.log_prob(z.cpu()).to(log_jac_det.device)

            nll = log_jac_det + log_prob

            return dict(
                log_prob=log_prob.mean(),
                log_jac_det=log_jac_det.mean(),
                negative_log_likelihood=nll.mean(),
            )

    hparams = GenerativeHParams(
        inputs=2,
        layers=10,
        subnet_widths=[128, 256, 128],
        max_epochs=10,
        batch_size=32,
    )
    benchmark_grid([GenerativeModel], [hparams], [
        CirclesDataset(),
        HypershellsDataset(),
        HypersphereMixtureDataset(),
        MoonsDataset(),
    ], trainer_kwargs=dict(limit_train_batches=1, limit_val_batches=1))
