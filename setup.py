from setuptools import find_packages, setup

setup(
    name="bm25_pt",
    version="0.0.2",
    description="bm25 search algorithm in pytorch",
    author="Jack Morris",
    author_email="jxm3@cornell.edu",
    packages=find_packages(),
    install_requires=open("requirements.txt").readlines(),
)
