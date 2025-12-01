from setuptools import setup, find_packages

setup(
    name='src',
    version='0.1',
    python_requires='>=3.9',
    packages=find_packages(),
    install_requires=[
        'gurobipy==11.0.3',
        'matplotlib',
        'numpy',
        'sympy',
        'scipy',
        'tqdm',
    ],
)

