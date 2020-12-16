# VQA
Visual question and answer


## Quick Start Notes
The instructions below detail how to activate the python 3 virtual environment
to ensure you have the correct dependencies. Virtual environments will run python dependencies
from the directories `venv` if on Mac or Linux, and `win-venv` if on Windows. Alternatively, you could
also install the required dependencies locally and run without a virtual environment (not recommended since some dependencies are
fairly large)

### On Linux and Mac
```
source activate-python-venv.sh
```


### On Windows
We recommend you use VSCode to run the scripts via the powershell.
At the bottom left of your opened VSCode screen there should be a blue bar, click on the environment. \
![Image of Yaktocat](https://github.com/nicklauscyc/vqa/blob/main/readme-images/bottom-left.png)

Then, select the virtual environment 'win-venv' from the drop-down menu at the top of your screen \
![Image of Yaktocat](https://github.com/nicklauscyc/vqa/blob/main/readme-images/popup-env.png)

To run python scripts, for example `<my-script-name>.py` simply ensure that you are not in `bash`, and run
```
python <my-script-name>.py
```

## Data Set
Data used for training can be found [here](https://vizwiz.org/tasks-and-datasets/vqa/).
Be sure to download the `test`, `val`, `train` datasets and put them in this
directory. Don't worry the `.gitignore` has been set to not track these
folders since they contain data way past the Github limit

## Running the code
This model is based off https://github.com/Cyanogenoid/pytorch-vqa

We run the scripts in the following order to achieve the best results

image_preprocessing_CNN.py

This extracts features from the image

preprocess-QA.py

This extracts features from the questions asked

train_models_cnn.py

This trains the model for 50 epochs with L2 regularization 0.0005

view-logs.py logs/<log name>

This prints out the results

