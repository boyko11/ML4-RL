Code at:
https://github.com/boyko11/ML4-RL

Assumes Python 3.5+

Open AI Gym  
* pip install gym  

Numpy  
* pip install numpy

tqdm - status bar during long iterations
* pip install tqdm

Tensorflow
* pip install tensorflow==1.8.0

Keras
* pip install keras

matplotlib
* pip install matplotlib

## FrozenLake-v0 ##

#### Value Iteration ####
python frozen_lake.py 4 v  
or  
python frozen_lake.py 8 v  
* the number specifies the size of the grid - 4: 4x4, 8: 8x8
* prints the calculated V matrix as a result of VALUE Iteration
* prints a visualization of the policy calculated based on the V matrix. The states are printed as numeric  
* prints a visualization of the policy with states as documented in the Open AI Gym documentation for the environment  
  
To Generate The Gamma Plot  
python frozen_lake.py 4 v gamma-plot  
* this will take a longer time
---

#### Policy Iteration ####
python frozen_lake.py 4 p   
or  
python frozen_lake.py 8 p  
* prints a visualization of the policy as a result of policy iteration. The states are printed as numeric  
* prints a visualization of the policy with states as documented in the Open AI Gym documentation for the environment  
  
To Generate The Gamma Plot  
python frozen_lake.py 4 p gamma-plot  
* this will take a longer time  
---

#### Q Learning ####
Deterministic:  
python frozen_lake.py 4 q d  
or  
python frozen_lake.py 8 q d 

Stochastic:  
python frozen_lake.py 4 q s  
or  
python frozen_lake.py 8 q s  

* both print the calculated Q matrix as a result of Q-Learning
* both print a visualization of the policy calculated based on the Q matrix. The states are printed as numeric  
  
To Generate The Gamma Plot  
python frozen_lake.py 4 q s gamma-plot  
* this will take a longer time  
---

#### Q Learning Miguel Morales ####
python frozen_lake_q_miguel_morales.py 4  
or  
python frozen_lake_q_miguel_morales.py 8  
---


## CartPole-v0 ##
#### Q-Learning with function approximation ####
python cart_pole.py  
To replicate the episode/reward plot:  
python plotting_service.py  

## GridWorld from AI Modern Approach ##
#### Value Iteration ####
python mdp.py small v  
or  
python mdp.py large v  
* 'small' will run the 3x4 grid described in our lecture and AI - Modern Approach  
* 'large' will run a 300x400 grid  
* prints the calculated V matrix as a result of VALUE Iteration  
* prints a visualization of the policy  

