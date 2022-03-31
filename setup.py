from setuptools import setup, find_packages

setup(
    name="project",
    version="0.0.1",
    description="RSA for Valence-Arousal",
    author="Kleanthis Avramidis, Georgios Chochlakis,"
    " Hyunjun Choi, Meghana Kolasani, Roshni Lulla",
    author_email="chochlak@usc.edu",
    packages=find_packages(),
    install_requires=["torch", "numpy", "torchvision", "scikit-learn", "tqdm"],
    extras_require={"dev": ["black", "pytest"]},
)
