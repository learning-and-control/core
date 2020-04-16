from setuptools import find_packages, setup

setup(
    name='core',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'cvxpy',
        'keras'
    ],
    extra_require={
        'dev': ['pytest']
    }
)
