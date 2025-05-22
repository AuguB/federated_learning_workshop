# federated_learning_workshop

Setup instructions - Linux and Mac only, Windows users please use WSL. 

1. Execute these commands in a terminal
```
conda create -n federated_learning_workshop python=3.12
conda activate federated_learning_workshop
pip install git+https://github.com/amarquand/PCNtoolkit.git@v1.alpha.7
conda install ipykernel graphviz libcxx
python -m ipykernel install --user --name=federated_learning_workshop
```
2. Clone this repo
3. Run the transfer.ipynb using the `federated_learning_workshop` environment
