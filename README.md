# Training spiking neural networks with a controller and STDP

This is an ongoing research project at the Institute of Neuroinformatics (INI) of the university of Zürich and ETH Zürich.
[This paper](https://www.frontiersin.org/articles/10.3389/fncom.2023.1136010/full) showed how to train neurons and networks using a controller coupled with spike-timing-dependent plasticity (STDP).
We try to extend this to spiking neural networks, with more realistic performance benchmarks (e.g. MNIST) and deeper networks, to see if this is indeed a viable training algorithm.

The work is in progress. Code is in Python, based on pytorch and pytorch-lightning.

### Credits
- Previous code this was based on: Pau Vilimelis Aceituno, Sander de Haan
- New code in pytorch and pytorch lightning: Martino Sorbaro, Sander de Haan
- Extension, debugging, new experiments on deep networks: Alexander Efremov
- Supervision: Pau Vilimelis Aceituno, Benjamin Grewe
