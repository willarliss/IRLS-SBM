# IRLS-SBM

Stochastic Block Model estimation with (latent variable) Iteratively Reweighted Least Squares. See [write-up](write_up.pdf) for more details.

This method impliments approximate Fisher scoring updates (similar to IRLS) using a one-hot projection of a latent variable representing the partition. The partition is iteratively estimated by IRLS while the structure matrix has a closed form MLE. Funtionality supports graphs, multi-graphs, and weighted graphs. There is a "slow" implementation (`sbm_slow`) which implements the algorithm exactly as written in the write-up as well as a fast implementation (`sbm_fast`) which employs a number of computational improvements.

<img src="sbm_animation.gif" width="500">
