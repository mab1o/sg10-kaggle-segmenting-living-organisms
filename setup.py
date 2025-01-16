from setuptools import setup, find_packages

setup(
    name="kaggle-segmenting-living-organisms",
    version="0.1",
    packages=find_packages(include=["torchtmpl", "torchtmpl.*"]),
)
