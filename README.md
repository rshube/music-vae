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
The data files are saved in `music-vae/data` by default. The data save location can be changed by changed the `DATA_PATH` variable in `music-vae/utils/consts.py`


## Train a model
In the top level of the repo, train and evaluate a model with:
```bash
python3 main.py --{optional_args}
```
The arguments you can add: 
| Flag      | Description | Default |
| ---------- | - | - |
| -h      | Show options and exit       | N/A|
| --model   | The model type to train. Must be one of: <br /> <ul><li>SimpleAutoEncoder</li><li>SimpleVarAutoEncoder</li><li>ComplexAutoEncoder</li><li>ComplexVarAutoEncoder</li><li>SimpleConvAutoEncoder</li><li>ComplexConvAutoEncoder</li><li>ImageConvAutoEncoder</li></ul> |SimpleAutoEncoder |
|--num_epochs | The number of epochs to complete during training | 10|
|--lr | The learning rate | 1e-03|
|--batch_size| The batch_size| 32|

By default, the model is trained on 10s clips of Lo-Fi music. Running this will save the a plot of the evaluation loss over each epoch of training, the evaluation loss against different types of data sets, and the model weights. The default save location is `music-vae/results/{model type}/{date-time}/`. This can be changed by changing the `save_dir` variable in the `main` function of `main.py`.

