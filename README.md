# music-vae
Music AutoEncoders (AEs) and Variational AutoEncoders (VAEs)

## Setup
Clone the repo:
```
git clone git@github.com:rshube/music-vae.git
```
Install the requirements from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Additionally, one will need to [install torch](https://pytorch.org/get-started/locally/).


## Fetch the data
In the top level of the repo, fetch the data sources and generate clips of `.wav` (this will take a few minutes):
```bash
python3 generate_wav_clips.py
```


## Train a model
In the top level of the repo, train and evaluate a model with:
```bash
python3 main.py --{optional_args}
```
The arguments you can add: 
| Flag      | Description | Default |
| ---------- | - | - |
| -h      | Show options and exit       | N/A|
| --model   | The model type to train. Must be one of: <br />{SimpleAutoEncoder, SimpleVarAutoEncoder, AutoEncoder, <br />SimpleConvAutoEncoder, ConvAutoEncoder, ImageConvAutoEncoder}      |SimpleAutoEncoder |
|--num_epochs | The number of epochs to complete during training | 10|
|--lr | The learning rate | 1e-03|
|--batch_size| The batch_size| 32|


