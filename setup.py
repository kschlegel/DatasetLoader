from setuptools import setup, find_packages

setup(
    name="datasetloader",
    version="0.0.1",
    author="Example Author",
    author_email="author@example.com",
    description="A small example package",
    url="https://github.com/pypa/sampleproject",
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'tqdm',
        'scipy',
        'h5py'
    ],
)
