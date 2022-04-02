import os

from setuptools import setup


def _read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


exec(open("firelink/_version.py").read())

setup(
    name="firelink",
    version=__version__,
    author="Chengran (Owen) Ouyang",
    author_email="chengranouyang@gmail.com",
    description=(
        "Firelink is based on scikit-learn pipeline and adding the functionality to store the pipeline in `.yaml` or `.ember` file for production."
    ),
    license="Apache License",
    keywords="firelink documentation tutorial",
    packages=["firelink"],
    long_description=_read("README.md"),
)
