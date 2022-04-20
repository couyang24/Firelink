import os

from setuptools import setup


def _read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="firelink",
    version="0.1.2",
    author="Chengran (Owen) Ouyang",
    author_email="chengranouyang@gmail.com",
    description=(
        "Firelink is based on scikit-learn pipeline and adding the functionality to store the pipeline in `.yaml` or `.ember` file for production."
    ),
    license="Apache License",
    keywords="firelink documentation tutorial",
    packages=["firelink"],
    install_requires=["pandas", "scikit-learn", "numpy"],
    long_description=_read("README.md"),
    long_description_content_type="text/markdown",
)
