# SIR_mouse
Agent-based SIR Model for the mouse brain 

This script runs the agent-based Susceptible-Infected-Removed (SIR) Model.
Authors:
    Ying-Qiu Zheng, Shady Rahayel
    
For running the model, run:
    
    python abm.py --retro True --speed 10 --spreading-rate 0.01 --time 30000
        --delta-t 0.1 --seed -1 --seed-amount 1
        
--retro True specifies a retrograde spreading

--speed is the spreading speed of agents in edges

--spreading-rate is the probability of staying inside a region

--time is the spreading time of agents

--delta-t is the size of timesteps

--seed is an integer that refers to the list of regions listed
alphabetically from the Allen Mouse Brain Atlas (see params_nature_retro.pickle)
CP = 35, ACB = 3, and CA1 = 24

--seed-amount is the initial injected amount of infected agents
        
This generates arrays containing the number of normal and infected agents
at each iteration for every region of the Allen Mouse Brain Atlas.
The distribution of normal agents can be found in .s_region_history
The distribution of infected agents can be found in .i_region_history
