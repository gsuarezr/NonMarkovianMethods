from setuptools import find_packages
from setuptools import setup
from Cython.Build import cythonize


setup(
    name='nmm',
    version='0.0.1',
    install_requires=[
        'numpy',
        'jax',
        'jaxlib',
        'multipledispatch',
        'scipy; python_version > "3.7"',
        'cython',
    ],
    ext_modules=cythonize(["nmm/cumulant/cum.pyx"],include_path=["nmm/cumulant/"]),
    packages=find_packages(),
    extras_require={"Full": ['qutip']},
    include_dirs=['.']
)
