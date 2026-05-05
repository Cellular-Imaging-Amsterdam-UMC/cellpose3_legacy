import setuptools
from setuptools import setup

install_deps = [
    'numpy>=1.20.0,<2.1',
    'scipy',
    'natsort',
    'tifffile',
    'tqdm',
    'numba>=0.53.0',
    'llvmlite',
    'torch>=1.6',
    'opencv-python-headless',
    'fastremap',
    'imagecodecs',
    'roifile',
    'fill-voids',
]

try:
    import torch
    a = torch.ones(2, 3)
    from importlib.metadata import version
    ver = version("torch")
    major_version, minor_version, _ = ver.split(".")
    if major_version == "2" or int(minor_version) >= 6:
        install_deps.remove("torch>=1.6")
except:
    pass

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="cellpose3_legacy",
    license="BSD",
    author="Marius Pachitariu and Carsen Stringer",
    author_email="stringerc@janelia.hhmi.org",
    description="Cellpose 3 legacy fork (cellpose <4) for coexistence with cellpose >=4",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cellular-Imaging-Amsterdam-UMC/cellpose3_legacy",
    setup_requires=['setuptools_scm'],
    packages=setuptools.find_packages(),
    use_scm_version=True,
    install_requires=install_deps,
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ),
    entry_points={'console_scripts': ['cellpose3_legacy = cellpose3_legacy.__main__:main']},
)
