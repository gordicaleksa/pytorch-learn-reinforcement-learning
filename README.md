## Reinforcement Learning (PyTorch) :robot: + :cake: = :heart:
This repo contains PyTorch implementation of various RL algorithms. <br/>
It's aimed at making it **easy** to start playing and learning about RL. <br/>

## Table of Contents
* [Policy gradient](#policy-gradient)
* [DQN](#dqn)
* [PPO](#ppo)
* [Setup](#setup)
* [Usage](#usage)
* [Hardware requirements](#hardware-requirements)
* [Learning material](#learning-material)
    
## Policy gradient

## DQN

## PPO

## Setup

Let's get this thing running! Follow the next steps:

1. `git clone https://github.com/gordicaleksa/pytorch-learn-reinforcement-learning`
2. Open Anaconda console and navigate into project directory `cd path_to_repo`
3. Run `conda env create` from project directory (this will create a brand new conda environment).
4. Run `activate pytorch-gat` (for running scripts from your console or setup the interpreter in your IDE)

That's it! It should work out-of-the-box executing environment.yml file which deals with dependencies. <br/>

-----

PyTorch pip package will come bundled with some version of CUDA/cuDNN with it,
but it is highly recommended that you install a system-wide CUDA beforehand, mostly because of the GPU drivers. 
I also recommend using Miniconda installer as a way to get conda on your system.
Follow through points 1 and 2 of [this setup](https://github.com/Petlja/PSIML/blob/master/docs/MachineSetup.md)
and use the most up-to-date versions of Miniconda and CUDA/cuDNN for your system.

## Usage

#### Option 1: Jupyter Notebook

Just run `jupyter notebook` from you Anaconda console and it will open up a session in your default browser. <br/>
Open `Annotated RL.ipynb` and you're ready to play!

---

**Note:** if you get `DLL load failed while importing win32api: The specified module could not be found` <br/>
Just do `pip uninstall pywin32` and then either `pip install pywin32` or `conda install pywin32` [should fix it](https://github.com/jupyter/notebook/issues/4980)!

#### Option 2: Use your IDE of choice

You just need to link the Python environment you created in the [setup](#setup) section.

## Hardware requirements

## Learning material

Here are some more advanced videos:

<p align="left">
<a href="https://www.youtube.com/watch?v=H1NRNGiS8YU" target="_blank"><img src="https://img.youtube.com/vi/H1NRNGiS8YU/0.jpg" 
alt="DQN paper explained" width="480" height="360" border="10" /></a>
</p>

I have some more videos which could further help you understand RL:
* [DeepMind: AlphaGo](https://www.youtube.com/watch?v=Z1BELqFQZVM)
* [DeepMind: AlphaGo Zero and AlphaZero](https://www.youtube.com/watch?v=0slFo1rV0EM)
* [OpenAI: Solving Rubik's Cube with a Robot Hand](https://www.youtube.com/watch?v=eTa-k1pgvnU)
* [DeepMind: MuZero](https://www.youtube.com/watch?v=mH7f7N7s79s)

## Acknowledgements

## Citation

If you find this code useful, please cite the following:

```
@misc{Gordić2021PyTorchLearnReinforcementLearning,
  author = {Gordić, Aleksa},
  title = {pytorch-learn-reinforcement-learning},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gordicaleksa/pytorch-learn-reinforcement-learning}},
}
```

## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/gordicaleksa/pytorch-learn-reinforcement-learning/blob/master/LICENCE)