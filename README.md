# Climate Trends of Tropical Cyclone Intensity and Energy Extremes Revealed by Deep Learning

In this repository, we provide codes to reproduce the result describe in the following paper:

### Climate Trends of Tropical Cyclone Intensity and Energy Extremes Revealed by Deep Learning

Buo-Fu Chen,
Center for Weather Climate and Disaster Research, National Taiwan University, Taiwan


## Requirements

To install requirements:

0. install pipenv (if you don't have it installed yet)
```setup
pip install pipenv
```
1. use pipenv to install dependencies:
```
pipenv install
```
2. install tensorflow **in the** pipenv shell
(choose compatible tensorflow version according to your cuda/cudnn version)
```
pipenv run pip install tensorflow
pipenv run pip install tensorflow_addons
```

## Training

To run the experiments, run this command:

```train
pipenv run python main.py <experiment_path>

<experiment_path>:

# CNN TC size estimation
experiments/IR_only/History_IR1.yml

# CNN-GAN TC size estimation
experiments/IR_VIS/History_3_stage_VIS_stage1.yml
experiments/IR_VIS/History_3_stage_VIS_stage2.yml
experiments/IR_VIS/History_3_stage_VIS_stage3.yml

```

***Notice that on the very first execution, it will download and extract the dataset before saving it into a folder "TCSA_data_2004_2018/".
This demands approximately 19GB space on disk as well as about 20 min preprocessing time, please be patient. :D***

### Some usful aguments

#### To limit GPU usage
Add *GPU_limit* argument, for example:
```args
pipenv run python train main.py <experiment_path> --GPU_limit 3000
```

#### Continue from previous progress
An experiemnt contains several sub_exp's.

Once the experiemnt get interrupted, we probably want to continue from the completed part.
For example, when the *History_IR1* experiment get interrupted when executing sub-exp #3 (*M03*), we want to restart from the beginning of sub-exp #3 instead of sub-exp #1.

We can do this to save times:

1. Remove partially done experiment's log.
```
rm -r logs/History_IR1/M03/ 
```

2. Restart experiment with argument: *omit_completed_sub_exp*.
```
pipenv run python train main.py experiments/IR_only/History_IR1.yml --omit_completed_sub_exp
```

## Evaluation

All the experiments are evaluated automaticly by tensorboard and recorded in the folder "logs".
To check the result:

```eval
pipenv run tensorboard --logdir logs

# If you're running this on somewhat like a workstation, you could bind port like this:
pipenv run tensorboard --logdir logs --port=1234 --bind_all
```

Curves can be obtained from the **[valid] regressor: blending_loss** in the scalar tab.

![tensorboard](figs/way_to_obtain_fig7.png)
