# Large-Scale Methods for Distributionally Robust Optimization

Code for the paper [*Large-Scale Methods for Distributionally Robust
Optimization*](https://arxiv.org/pdf/2010.05893.pdf) by Daniel Levy*, Yair Carmon*, John C. Duchi and Aaron
Sidford, to appear at NeurIPS 2020.

## Dependencies

This code is written in python, dependencies are:
* Python >= 3.6
* [PyTorch](http://pytorch.org) >= 1.1.0
* torchvision >= 0.3.0
* numpy
* pandas
* **for unit tests only**: [CVXPY](https://www.cvxpy.org/) and MOSEK

## Datasets

For ImageNet and MNIST, we use the datasets provided by `torchvision`. For the
typed-written digits, see [Campos, Babu & Varma,
2009](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/). We provide the details of
how we use the datasets in Section F.1 of the Appendix of the paper. The code
extracting features can be found in `./features`.

[Features for the Digits experiments](https://www.dropbox.com/s/e0puwp86vyh4dkg/digits_data.zip?dl=0)

## Robust Losses

The file `robust_losses.py` implements the gradient estimators we consider in the paper.
In particular, it includes the two main estimators we study: Mini-batch and
and the multilevel Monte Carlo (MLMC). It also includes implementations for the "baselines" methods
we consider: dual-SGM and primal-dual. Our code relies on PyTorch for auto-differentiation and 
is usable in any existing (PyTorch) training code. We show
in Appendix F.3 how to integrate it in less than 3 lines of code.

## Training and Evaluation
The training and evaluation code are contained in `train.py`. Here is an example
command to run for the ImageNet dataset, for the $\chi^2$ uncertainty set of
size 1 and running with a batch size of 500, momentum of 0.9 and learning rate
of 0.01 for 30 epochs.
```
python train.py --algorithm batch --dataset imagenet --data_dir ../data --epochs 30 --momentum 0.9 --lr_schedule constant --averaging constant_3.0 --wd 1e-3 --geometry chi-square --size 1.0 --batch_size 500 --lr 1e-2 --output_dir ../output-dir
```

## Hyperparameters

The hyperparameters (including seeds) for all the experiments we show in the paper
are detailed in the `./hyperparameters` folder. We describe our search strategy
(a coarse-to-fine grid) in Appendix F.2.

## Reference
```
@inproceedings{levy2020large,
  title={Large-Scale Methods for Distributionally Robust Optimization},
  author={Levy, Daniel and Carmon, Yair and Duchi, John C and Sidford, Aaron},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```
