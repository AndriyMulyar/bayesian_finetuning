'''Setup for neural network training'''
from setuptools import setup, find_packages

setup(
    name='bayesian_finetuning',
    version='0.0.1',
    url='https://github.com/AndriyMulyar/bayesian_finetuning',
    description='Experiments for improving neural network fine tuning with bayesian inspired priors.',
    packages=find_packages(),
    install_requires=[
        'pytorch-lightning',
        'torch',
        'wandb',
        'ruamel.yaml',
        'black',
        'coverage',
        "pylint",
        "pytest",
        "attrdict",
        "joblib",
        "datasets",
        "transformers",
        "scipy",
        "sklearn"
    ],
    include_package_data=True
)



