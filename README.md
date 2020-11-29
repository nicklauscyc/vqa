# VQA
Visual question and answer

## Quick Start Notes
The instructions below detail how to activate the python 3 virtual environment
to ensure you have the correct dependencies
### On Linux and Mac
```
source activate-python-venv.sh
```


### On Windows
```
win-venv\Scripts\Activate.ps1
```
If the above does not work in VSCode's powershell, first run this command, then re-run
the above command:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
Alternative is to run `bash` in VSCode's terminal and then run the Linux version of commands.

## Data Set
Data used for training can be found [here](https://vizwiz.org/tasks-and-datasets/vqa/). 
Be sure to download the `test`, `val`, `train` datasets and put them in this 
directory. Don't worry the `.gitignore` has been set to not track these 
folders since they contain data way past the Github limit
