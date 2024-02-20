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
    install_requires=["numpy >= 1.20", "matplotlib", "pgmpy"],
    python_requires=">=3.9, <=3.11",
)
