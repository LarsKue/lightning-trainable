from lightning_trainable.datasets import *
import matplotlib.pyplot as plt


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
    plt.close()


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
    plt.close()


def test_hypershells():
    ds = HypershellsDataset(dimensions=2, shells=3)

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
    plt.close()


def test_hypersphere_mixture():
    ds = HypersphereMixtureDataset(dimensions=2)

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
    plt.close()


def test_normal():
    ds = NormalDataset(dimensions=2)

    samples = ds.distribution.sample((1000,)).numpy()

    plt.scatter(*samples.T)
    plt.title("Normal Distribution")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("figures/normal.png")
    plt.close()


def test_uniform():
    ds = UniformDataset(dimensions=2)

    samples = ds.distribution.sample((1000,)).numpy()

    plt.scatter(*samples.T)
    plt.title("Uniform Distribution")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("figures/uniform.png")
    plt.close()
