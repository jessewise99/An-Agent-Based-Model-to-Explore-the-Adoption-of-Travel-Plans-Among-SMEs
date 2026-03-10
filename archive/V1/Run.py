############################################
#     Toy model prototype - Run            #
#     Date: 2025-07-01                     #
#     Author: Jesse Wise                   #
#     Purpose: To learn how Mesa works     #
#     This will run the model              #
############################################

# Be aware that MESA will take 3.11 or higher #pip install --upgrade mesa[rec] 
# ter Hoeven, E., Kwakkel, J., Hess, V., Pike, T., Wang, B., rht, & Kazil, J. (2025). Mesa 3: Agent-based modeling with Python in 2025. Journal of Open Source Software, 10(107), 7668. https://doi.org/10.21105/joss.07668

# This file is used to run the model. To do so, you must import the model and agents from their respective files.

from Model import AdoptionModel # From the file Model.py, import the AdoptionModel class.
from Agents import FirmAgent # From the file Agents.py, import the FirmAgent class.


# These parameters will need to be tuned and calibrated

T =  20 										# Set how many  time steps the program will run for
N = 100								# Set how many agents there are in the model, THIS DOES NOT ACTUALLY CHANGE NUM_AGENTS YET

#learning_rate = 0.3									# This is the rate at which firms learn from other firms
#realism_pull_constraints = 0.3								# For time and money constraints set the realism pull as higher for these very objective concepts
#realism_pull_sociallyInfluencedVars = 0.1						# For benefits, costs, and knowledge, the realism pull is lower as these are more subjective likely to be swayed by social influence

time_min = 0.2										# This is the time threshold, if exceeded they may be able to adopt
money_min = 0.3										# This is money threshold, if exceeded they may be able to adopt
knowledge_min = 0.4									# This is the knowledge threshold, if exceeded they may be able to adopt




model = AdoptionModel(num_agents= N)  # Create an instance of the AdoptionModel with N agents.
### You need to finish passing the rest of the parameters once you've desinged the model. This is just a placeholder.


for i in range(T):
    model.step()


# You need to run data collection and the batch runner too see here https://mesa.readthedocs.io/latest/overview.html