<h1 align="center">Variationally Enhanced Adjoint Schrödinger Bridge Sampler</h1>

We implement **Variationally Enhanced Adjoint Schrödinger Bridge Sampler**, which continues idea of **Well-Tempered Adjoint Schrödinger Bridge Sampler (WT-ASBS)**
about sampling of chemical systems, using low-dimensional features of atomic coordinates (collective variables).

In original article [Enhancing Diffusion-Based Sampling with Molecular Collective Variables](https://arxiv.org/abs/2510.11923) authors
used collective variables to get Gaussian-style bias, which helped in mode exploration, because default [Adjoint Schrödinger Bridge Sampler](https://arxiv.org/pdf/2506.22565) have difficulties to sample low-energy conformations. But in **WT-ASBS** authors used many hyperparameters which is not solve original problem of
amortized conformers sampling, so we trained additional neural network, which was proposed by [Neural Networks Based Variationally Enhanced Sampling](https://arxiv.org/pdf/1904.01305) to have less parameters.

## License

`ve-asbs` code, model, and checkpoints are licensed under the [FAIR Chem License](LICENSE). Data is licensed under CC-BY-4.0. Dependencies have their own licences.
