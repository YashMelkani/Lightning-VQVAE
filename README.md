# VQVAE for Track Tokenization
This repo contains code for tokenizing detector hits (3d space points) in HEP experiments into discrete tokens. It uses data from the Kaggle TrackML challenge. Two frameworks for tokenization are provided: Conditional and Unconditional.


## Conditional Tokenization
In the conditional framework, the tokenization of a space point is dependent on all the other space points in the input. Transformer encoders are used to encode and decode space point sequences to and from the latent space. For particle tracks, it is natural to tokenize each detector hit conditioned on all the other detector hits in the track. Two types of datasets are provided in this implementation. One uses the PyTorch IterableDataset module whereas the other uses the standard map-style Dataset


## Unconditional Tokenization
In the unconditional framework, the tokenization of a space point is independent of the other space points in the input. As a result, a space point will always map to the same token in the codebook/vocabulary. A simple MLP is used to encode and decode space points to and from the latent dimension. 

## Dataset
Code for creating track data from the TrackML challenge dataset is included in the create_event_data.py file. For loading tracks, a IterableDataset and a map-style Dataset are provided. My dataset was small enough that I could load everything to memory during training. I am hoping to scale up to larger datasets in the future which is why I included an IterableDataset so that data could be loaded on file at a time. Note that I have not fully tested the IterableDataset so I suggest verifying its implementation before using it... :). 

When training, configurations for the training and validation datasets are specified in the `config.yaml` file. A copy of this file is saved to the lightning_logs after training.

## Model Training
I used a simple shell script `run.sh` to source DistributedDataParallel variables and to launch training. The file `train_vqvae.py` initializes the dataloaders + the model and runs training.
 
## Monitoring + Eval
A jupyter notebook is provided to launch a tensorboard session. I also included a notebook to evaluate model performance for both the conditional and unconditional models.

## Adapting
To adapt this project for other implmentations, make sure to change the datasets as well as the `create_loader()` and `create_dataset()` methods in `utils/data.py`. Changes to the encoder/decoder architectures should be made in `utils/vqvae.py`. Code for the vector-quantization block is in `utils/quantizers.py`

## Acknowledgements
Vector quantizer code in utils.quantizers taken from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/distributed/distributed.py and https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb 

