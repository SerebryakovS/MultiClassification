## Muti-Class Binary classification
This project contains 3 models(LSTM,BERT,GPT) implemented for multi-classification task resolving.
It also contains the data preparation tool for easily dataset preparation for each of used model.
Project's main output contains the train/valid data and also generate plots for all used metrics Acccuracy/Loss,Recall,Presition and AUC

## Installation guidance (may differ, include primary steps)
```
conda create --name nlpenv python=3.9
conda activate nlpenv
conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
pip install --upgrade pip setuptools wheel
pip install tensorflow
pip install datasets transformers
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```
Sign out and sign back in via SSH or close and re-open your terminal window. Reactivate your conda session.
```
conda activate tf
python3 -m pip install tensorflow==2.10
```
### Verify installation:
```
python3 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"
```
