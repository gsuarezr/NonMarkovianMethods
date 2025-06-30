![Continuous Integration](https://github.com/mcditoos/NonMarkovianMethods/actions/workflows/continous_integration.yml/badge.svg)

[![codecov](https://codecov.io/github/gsuarezr/NonMarkovianMethods/graph/badge.svg?token=80B4ABOUYR)](https://codecov.io/github/gsuarezr/NonMarkovianMethods)

[![Documentation](https://github.com/gsuarezr/NonMarkovianMethods/actions/workflows/documentation.yml/badge.svg)](https://github.com/gsuarezr/NonMarkovianMethods/actions/workflows/documentation.yml)

# NonMarkovian Methods

This package has a a goal to collect some of the alternative approaches to open quantum systems into one package with unified notation. For now it only supports the cumulant equation (Aka Refined Weak Coupling limit), and the Redfield equation. The API documentation  and examples from papers can be found in https://gsuarezr.github.io/NonMarkovianMethods/api/

This is undergoing development and major changes are expected, if you find it useful and use it in your research please cite [ Dynamics of the Non-equilibrium spin Boson Model: A Benchmark of master equations and their validity ](https://arxiv.org/abs/2403.04488) (the code necessary for the simulations in the paper is in example 1)

using

```bibtex 
@article{Su_rez_2024,
   title={Dynamics of the nonequilibrium spin-boson model: A benchmark of master equations and their validity},
   volume={110},
   ISSN={2469-9934},
   url={http://dx.doi.org/10.1103/PhysRevA.110.042428},
   DOI={10.1103/physreva.110.042428},
   number={4},
   journal={Physical Review A},
   publisher={American Physical Society (APS)},
   author={Suárez, Gerardo and Łobejko, Marcin and Horodecki, Michał},
   year={2024},
   month=oct }
```
The other examples come from 

```bibtex 
@misc{suárez2025makingnonmarkovianmasterequations,
      title={Making Non-Markovian master equations accessible with approximate environments}, 
      author={Gerardo Suárez and Micha\l{} Horodecki},
      year={2025},
      eprint={2506.22346},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2506.22346}, 
}
```

Many features of this code depend on QuTiP's enviroment class, see the 
[documentation and tutorials](https://qutip.readthedocs.io/en/latest/guide/guide-environments.html) 
to get familiar with it.

For the linear algebra computations, the code can use QuTiP or jax. Which one is 
used depends on which objects are passed to the solver, either a QuTiP Object, 
or a nmm.Qobj which mimics a QuTiP QObj in jax. The former is recommended, and 
now supports jax easily through the [QuTiP-jax plugin](https://github.com/qutip/qutip-jax).
