# vae-tutorial
Implementation of a Variational Autoencoder (VAE) in Pytorch based on the [tutorial by Hunter Heidenreich](https://hunterheidenreich.com/?url=https%3A%2F%2Fhunterheidenreich.com%2Fposts%2Fmodern-variational-autoencoder-in-pytorch%2F).

This is a really simple project which I used to learn more about VAEs. It trains a VAE on the MNIST dataset of hand-written digits.

## Setup
1. Clone this repo and navigate to the root directory.
2. (Optional) Create a virtual environment called ``.venv``. Alternatively you can just install pip packages globally, up to you really.
```
python -m venv .venv
```
4. Activate the virtual environment (Windows).
```
.\.venv\Scripts\activate
```
5. Install [pytorch](https://pytorch.org/get-started/locally/) separately. If you want to use a GPU make sure you have CUDA installed and all that jazz.
6. Install requirements with pip.
```
pip install -r requirements.txt
```

## Usage
You can train a model by running the ``train.py`` file in the ``source`` directory.
```
python .\train.py
```
This will train for 10 epochs by default and creates a Tensorboard log of training, as well as a checkpoint at the end so the model can be loaded for inference or further training. The logging and training code all comes from the tutorial by Hunter Heidenreich, so please go check that out. It even shows samples of ground truth images with their reconstructions and randomly generated images from your models while they train!
The optimizer is included in the checkpoint as well as hyperparameter configurations for training and the model.
View the Tensorboard training logs in VSCode by installing the Tensorboard extension by Microsoft.

To change the model or training hyperparameters you can pass in command-line arguments like so:
```
python .\train.py --lr 2e-3 --latent_dim 3
```
Run ``python .\train.py --help`` for info on cli args.

## Suggestions
1. Train a few models with different hyperparameters, see if you can beat my loss of 132.02!
1. Create a representation of the MNIST dataset in latent space. This would be appropriate for 1D/2D/3D latent space, but feel free to have a go at 4D+ one day! One way of doing this is to place a point in N-space for each encoded image and colour the points based on the digit. This was done in [the tutorial](https://hunterheidenreich.com/?url=https%3A%2F%2Fhunterheidenreich.com%2Fposts%2Fmodern-variational-autoencoder-in-pytorch%2F) by Hunter Heidenreich but I do not think he gives the python code for it, so give it a go!
1. Use other datasets, such as FashionMNIST or your own custom dataset. Try to create a nice visualisation of latent space.
1. Create some animations or other art. E.g. an animation of one digit morphing into another continuously by definining a path through the latent space.