# core
Python simulation and hardware library for learning and control

#Papers Implemnted in this Library:

- [Learning to Control an Unstable System with One Minute of Data:
Leveraging Gaussian Process Differentiation in Predictive Control](https://arxiv.org/pdf/2103.04548.pdf)

## macOS setup
Set up virtual environment
```
python3 -m venv .venv
```
Activate virtual environment
```
source .venv/bin/activate
```
Upgrade package installer for Python
```
pip install --upgrade pip
```
Install requirements
```
pip3 install -r requirements.txt
```
Create IPython kernel
```
python3 -m ipykernel install --user --name .venv --display-name "Virtual Environment"
```
