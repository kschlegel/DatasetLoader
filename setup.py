from setuptools import setup, find_packages

setup(
    name="datasetloader",
    version="0.0.1",
    author="Kevin Schlegel",
    author_email="kevinschlegel@cantab.net",
    description=
    "A utility project to provide a convenient and consistent access to various datasets.",
    url="https://github.com/kschlegel/DatasetLoader",
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.18.5', 'tqdm>=4.46.1', 'scipy>=1.4.1', 'h5py>=2.10.0'
    ],
)
