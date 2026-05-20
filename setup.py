from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='dlbs_instance_segmentation',
    version='0.0.0',
    description='A library for instance segmentation using ultralitics.',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'pandas',
        'numpy',
        'pyyaml',
        'h5py',
        'scipy',
        'Pillow',
        'opencv-python',
        'pycocotools',
        'scikit-image',
        'matplotlib',
        'seaborn',
        'wandb',
        'ultralytics'
    ],  

    extras_require = {
    # FHNW slurm cluster (CUDA 12.6)
    "calculon": [
        "torch==2.9.1+cu126",
        "torchvision==0.24.1+cu126",
        "torchaudio==2.9.1+cu126",
        "torchmetrics>=1.4",
    ],

    # Local GPU (your workstation CUDA 12.8)
    "local": [
        "torch==2.9.1+cu128",
        "torchvision==0.24.1+cu128",
        "torchaudio==2.9.1+cu128",
        "torchmetrics>=1.4",
    ],

    # CPU / CI / Mac / testing
    "cpu": [
        "torch",
        "torchvision",
        "torchaudio",
        "torchmetrics>=1.4",
    ],
},
    entry_points={
        'console_scripts': [
            'dlbs_instance_segmentation=dlbs_instance_segmentation.cli:main',
        ],
    },
    author='Florian Baumgartner, Juan Ruiz',
    author_email='your.email@example.com',
    url='https://github.com/JuanFelipeRuiz/dlbs',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
