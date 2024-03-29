from os import path

import setuptools

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

required_pypi = [
    'numpy',
    'scikit-learn',
    'pandas',
    'tqdm',
    'dict_hash', # required for caching
    'imodelsx',
    'transformers',
    'datasets',
    'spacy',
    'accelerate',

    'datasets', # optional, required for getting NLP datasets
    'pytest', # optional, required for running tests
]

setuptools.setup(
    name="tprompt",
    version="0.01",
    author="Jack Morris, Chandan Singh, Yuntian Deng",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/tree-prompt",
    packages=setuptools.find_packages(
        exclude=['tests', 'tests.*', '*.test.*']
    ),
    install_requires=required_pypi,
    python_requires='>=3.9',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
