# NonMarkovian methods documentation

Provides a solver class for the Refined Weak Coupling Limit/Cumulant equation 
and the $TCL_{2}$ Redfield equation. The solverś here are only valid for Bosonic environments, support for fermionic environmets will come soon. The results for the 
Refined Weak Coupling Limit are always in the interaction picture while the ones for Redfield can be in Schrodinder's or interaction picture, it is selected using the picture argument ('S' or 'I').

More about the cumulant equation can be found in [1,2,3,4]. The notation used for $TCL_{2}$ can be found in [1,5]
We also provide a way to speed up their simulation using approximate environments as described in [2]. The approach in that paper is used when the argument Matsubara is set to True, otherwise numerical integration is used. 

All the examples in this code require QuTiP, as the methods here are validated using QuTiP's HEOM solver [6,7]. The examples also required QuTiP  the environment class [7].

1. [ Suárez, G., Łobejko, M., Horodecki, M. (2024). Dynamics of the Non-equilibrium spin Boson Model: A Benchmark of master equations and their validity.](https://arxiv.org/abs/2403.04488) . This article can be reproduced with the code from example one 
2. [Suárez, G., Horodecki, M. (2025). Making Non-Markovian master equations accessible with approximate environments](https://arxiv.org/abs/2506.22346). The examples in this article can be reproduced with the code from examples 2-5
3. [Rivas, Á. (2016). Refined Weak Coupling Limit: Coherence, Entanglement and Non-Markovianity.](https://arxiv.org/abs/1611.01483)
4. [Winczewski, M., Mandarino, A., Suarez, G., Horodecki, M., Alicki, R. (2021). Intermediate Times Dilemma for Open Quantum System: Filtered Approximation to The Refined Weak Coupling Limit.](https://arxiv.org/abs/2106.05776)
5. [Łobejko, M., Winczewski, M., Suárez, G., Alicki, R., Horodecki, M. (2022). Corrections to the Hamiltonian induced by finite-strength coupling to the environment.](https://arxiv.org/abs/2204.00643)
6. [Lambert, N., Raheja, T., Cross, S., Menczel, P., Ahmed, S., Pitchford, A., Burgarth, D., Nori, F. (2020). QuTiP-BoFiN: A bosonic and fermionic numerical hierarchical-equations-of-motion library with applications in light-harvesting, quantum control, and single-molecule electronics](https://arxiv.org/abs/2010.10806)
7. [Lambert, N., Giguère, E., Menczel, P., Li, B., Hopf, P., Suárez, G., Gali, M., Lishman, J., Gadhvi, R., Agarwal, R., Galicia, A., Shammah, N., Nation, P., Johansson, J.R., Ahmed, S., Cross, S., Pitchford, A., Nori, F. (2024). QuTiP 5: The Quantum Toolbox in Python](https://arxiv.org/abs/2412.04705)