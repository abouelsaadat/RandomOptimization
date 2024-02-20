"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

import setuptools

setuptools.setup(
    version="1.0.0",
    url="https://github.com/abouelsaadat/RandomOptimization.git",
    author="Mohamed Abouelsaadat",
    author_email="mohamed.abouelsaadat@gmail.com",
    description="Description of my package",
    packages=["randoptma"],
    install_requires=[
        "numpy >= 1.20",
        "matplotlib",
        "pgmpy",
        "multiprocess",
        "setuptools >=42",
        "dill >= 0.3.8",
    ],
    python_requires=">=3.8, <=3.11",
)
