# Cores

This directory defines the `Core` managers that run the different algorithms within 
Marco Polo. Simulations begin with `core.PopulationManager()` importing the 
appropriate `Core` manager, which will then setup the desired namespace. The idea 
is to create an appropriate `Core` for every evolution algorithm desired with optimization 
managers, models, enviroments flexibly swapable from the command line. 
