# Online Grounding of Symbolic Planning Domains in Unknown Environments

This [repository](https://github.com/LamannaLeonardo/OGAMUS) contains the official code of the Online Grounding of Action Models in Unknown Situations (OGAMUS) algorithm that will be presented at the 19th International Conference on Principles of Knowledge Representation and Reasoning (KR-2022 Special Session of KR and Robotics), for details about the method please see the [paper](https://arxiv.org/abs/2112.10007).


## Installation
The following instructions have been tested on Ubuntu 20.04.


1. Clone this repository
```
 git clone https://github.com/LamannaLeonardo/OGAMUS.git
```

2. Create a Python 3.9 virtual environment using conda or pip.
```
 conda create -n ogamus python=3.9
```

3. Activate the environment
```
 conda activate ogamus
```

4. Install pip in the conda environment
```
 conda install pip
```

5. Install [PyTorch](https://pytorch.org/get-started/locally/) (tested with version 1.11.0)

6. Install [AI2THOR](https://ai2thor.allenai.org/ithor/documentation) (tested with version 4.2.0) 
```
  pip install ai2thor
```

7. Install the following dependencies
```
pip install matplotlib
```

8. Download the pretrained neural network models available at this [link](https://drive.google.com/drive/folders/1UjADpBeBOMUKXQt-qSULIP3vM90zr_MR?usp=sharing), and move all the downloaded files into the directory "Utils/pretrained_models"

9. Check everything is correctly installed by executing the command
```
  python main.py
```


## Issues
1. This github repository already contains the [FastForward](https://fai.cs.uni-saarland.de/hoffmann/ff.html) planner in the directory "OGAMUS/Plan/PDDL/Planners/FF". If you face any issue, you can compile it from scratch as follows: from the [offical FastForward site](https://fai.cs.uni-saarland.de/hoffmann/ff.html), download FF-v2.3.tgz (you can directly download it from this [link](https://fai.cs.uni-saarland.de/hoffmann/ff/FF-v2.3.tgz)), move it into the "Planners/FF" directory, extract the archive ```tar -xf FF-v2.3.tgz```, go into the installation directory with ```cd FF-v2.3``` and compile FastForward with ```make```. Finally move the "ff" executable in the parent directory through the command ```mv ff ../```, go to the parent directory ```cd ../``` and delete unnecessary files with ```rm -r FF-v2.3``` and ```rm FF-v2.3.tgz```.



## Execution

### Running OGAMUS
The OGAMUS algorithm can be run over the following tasks: on, open, close, object goal navigation (for further details about the tasks, please see the [paper](https://arxiv.org/abs/2112.10007)). 
To run OGAMUS on a specific task w/o ground truth object detections, there are two options:
    
a) -t xxx where "xxx" is the task you want to test, available tasks are: on, open, close, ogn, ogn_ithor
    
b) -obj (or -o), when you pass this option, the agent uses ground truth object detections

e.g. to run OGAMUS on the task "on" with ground truth object detections, execute the command: "python main.py -t on -o"


## Log and results
When you execute OGAMUS, a new directory with all logs and results is created in the "Results" folder. For instance, the logs and results are stored in the folder "Results/test_set_X_stepsY", where X is the task name provided as input and Y the number of steps (which equals 200 for all tasks but object goal navigation in RoboTHOR). One subdirectory is created for each episode, which consists of a run in a single environment. Each episode subdirectory contains evaluation and log files relative to a single episode.
If you want to generate a summarized evaluation of all episodes in a directory named "DIR", open the script "Utils/ResultsPlotter.py"
and change the value of the "DIR" variable (at the beginning of the script) with the path of the results directory you want
to evaluate.
    e.g. after running "python main.py -t on -o", in ResultsPlotter.py set DIR = "Results/test_set_on_steps200" and execute
    the command "python ResultsPlotter.py"
    
For the object goal navigation task, if you want to generate the additional metric SPL, look at the end of the file
ResultsPlotter.py, comment "generate_plots()" and uncomment "ogn_metrics()", then run the script as above.



## Citations
Coming soon

## License
This project is licensed under the MIT License - see the [LICENSE](/LICENSE) file for details.
