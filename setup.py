from setuptools import setup, find_packages

setup(
    name="lightning-trainable",
    version="0.0.1",
    description="A light-weight trainable module for pytorch-lightning.",
    author="Lars Kühmichel",
    author_email="lars.kuehmichel@stud.uni-heidelberg.de",
    url="https://github.com/LarsKue/lightning-trainable",
    install_requires=["pytorch-lightning"],
    python_requires=">=3.10.0",
    packages=find_packages(),
)
