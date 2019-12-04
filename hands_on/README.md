# FLOW19_MLSchool
Code for the hands-on sessions of the 2019 FLOW School on Machine Learning 

## Regression session
### To run the notebook
* Using Docker (as root):  
```docker run -i -t -p 8888:8888 -v "$PWD:/home/" continuumio/anaconda3 /bin/bash -c "/opt/conda/bin/jupyter notebook --notebook-dir=/home/ --ip='0.0.0.0' --port=8888 --no-browser --allow-root"```  
* Using local Anaconda installation (with Jupyter notebook installed)   
  Clone the repository, then launch from terminal  
  ```jupyter-notebook```  
  Browser should open automatically  
* Using Anaconda installation (with Jupyter notebook installed) on a remote computer  
  Clone the repository on the remote computer, then launch from remote terminal  
  ```jupyter-notebook --no-browser--port=XXXX```  
  The port needs to be forwarded to connect to the notebook from the local computer. Connect to the remote computer using
  ```ssh -N -f -L localhost:YYYY:localhost:XXXX remoteuser@remotehost```  
  Connect to the notebook using the browser at the URL *"localhost:YYYY"*

**fit-sine.ipynb**  
Noisy samples from a sine functions are taken and fit with a polynomial functions of increasing order. The concepts of overfitting and L2 regularization, as well as stochastic gradient descent (SGD) are introduced. 

## RNN Session
### To run the notebook
* Using Docker (as root):
    1. Launch the Anaconda container  
    ```docker run -i -t -p 8888:8888 -v "$PWD:/home/" continuumio/anaconda3 /bin/bash```  
    2. Install PyTorch  
    ```conda install pytorch cpuonly -c pytorch```
    3. Run Jupyter notebook  
    ```/opt/conda/bin/jupyter notebook --notebook-dir=/home/ --ip='0.0.0.0' --port=8888 --no-browser --allow-root```

* Using Anaconda installation, either local or remote. Before launching the notebook, make sure that PyTorch is installed with  
  ```conda install pytorch cpuonly -c pytorch```  
  
**pytorch_introduction.ipynb**  
Guide to the PyTorch and implementation of neural networks with this framework 

**generate-lorenz.py**  
The script integrates Lorenz equations and saves the output in a file called *series_xxxx.npz* in the folder *./datasets/*.
The length of the simulation and the time interval between samples can be adjusted.  
There are two saved time series in the *./datasets/* folder:  
* *series_0000.npz*: Simulation length: 14,000 (5,000 training, 5,000 validation, 4,000 testing) - dt: 0.05
* *series_0001.npz*: Simulation length: 104,000 (50,000 training, 50,000 validation, 4,000 testing) - dt: 0.05

**train.py**  
Trains the RNN network.  
Training parameters can be adjusted at the beginning of the script.
