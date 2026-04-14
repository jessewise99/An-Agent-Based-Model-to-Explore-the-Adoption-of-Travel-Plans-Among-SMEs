############################################
#     Model  - Batch Runner                #
#     Date: 2026-04-08                     #
#     Author: Jesse Wise                   #
#     Purpose: Implementing Pseudocode V2  #
############################################

# Note: I have used Chat GPT to help me learn how to use Mesa, and to write this code. All errors are my own. 
# Be aware that MESA will take 3.11 or higher #pip install --upgrade mesa[rec] 
# ter Hoeven, E., Kwakkel, J., Hess, V., Pike, T., Wang, B., rht, & Kazil, J. (2025). Mesa 3: Agent-based modeling with Python in 2025. Journal of Open Source Software, 10(107), 7668. https://doi.org/10.21105/joss.07668

# This file is used to run the model. To do so, you must import the model and agents from their respective files.
# This file should choose parameters, activate shocks, run the model and plot results.

from Model import AdoptionModel # From the file Model.py, import the AdoptionModel class.
from Agent_Kof4FeasbilityRule import FirmAgent # From the file Agents.py, import the FirmAgent class.
import numpy as np #Has multi-dimensional arrays and matrices. Has a large collection of mathematical functions to operate on these arrays.
import pandas as pd # Data manipulation and analysis.
import seaborn as sns # Data visualization tools.
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.figure import Figure
import mesa
import pyreadr # To write an .rds file
from collections import defaultdict
from collections import Counter


######################################################################### Parameter Sweeps with Batch Runner #########################################################################
# You need to run data collection and the batch runner too see here https://mesa.readthedocs.io/latest/overview.html
# These parameters will need to be tuned and calibrated: learning_rate, realism_pull_constraints, realism_pull_sociallyInfluencedVars, competitor_inference_increment

T =  29 										# The program runs for 28 years because I have data from 1997 to 2025.
N =  500                                        # Set how many agents there are in the model. 

#--- Setting the parameters for the batch runner ---
params = {"learning_rate":  [0.4, 0.5, 0.6],								# This is the rate at which firms learn from other firms
          "competitor_inference_increment":  [0.01, 0.03, 0.05], # This is how much an agent's perceived benefits increases or decreases depening on their compeitors adoption stage. (at the moment = to learning rate* learning)
          "realism_pull_constraints" :  [0.5, 0.7, 0.9],								# For time and money constraints set the realism pull as higher for these very objective concepts
          "init_positive_shift" :[0, 0.01],                                  # This is used for calibration of initial distributions of beliefs
          "collect_agent_data": False, # False while doing such large sweeps
          ## These are not changing, but I have to pass them in anyway
          "num_agents": N, # Set how many agents there are in the model. This needs to be <= the number of firms in the data file.
          "organisationalReadiness_min": 0.4367,										# This is the organisational readiness threshold, if exceeded they may be able to adopt
          "publicTransport_min": 0.5883,										# This is the public transport threshold, if exceeded they may be able to adopt
          "resource_min" :.5683,										# This is resource threshold, if exceeded they may be able to adopt
          "knowledge_min": 0.4667,									# This is the knowledge threshold, if exceeded they may be able to adopt
          "obj_net_benefit_min":	188,					# This is the lower threshold for the net benefits (£) an SME can expect per employee per year, according to the RAS project
          "obj_net_benefit_max" :	250,					# This is the upper threshold for the net benefits (£) an SME can expect per employee per year, according to the RAS project
          "active_shocks" : None, #{"caseStudy", "subsidy"} # These are the policies in effect. Needs to be a set.
          "shock_parameters" : None#{"accreditationAward": 0.25} # These are the strengths of the policies, it needs to be a dictionary. It will look like this {"caseStudy": 0.3, "subsidy": 0.2}
          } 


#--- Running the batch runner ---
results = mesa.batch_run(
     AdoptionModel,
     parameters=params,
     iterations=5, # The number of iterations to run each parameter combination for. Optional. If not specified, defaults to 1. 10 is the minimum really.
     max_steps=T, # How many steps to run the model for (needs 28 years @1 ticks per year = 28)
     number_processes=1,
     data_collection_period=1, # The length of the period (number of steps) after which the model and agent reporters collect data. Optional. If not specified, defaults to -1, i.e. only at the end of each episode.
     display_progress=True,
 )

#--- Analysis and visualisation of batch results ---
results_df = pd.DataFrame(results)
print(f"The results have {len(results)} rows.")
print(f"The columns of the data frame are {list(results_df.keys())}.")

pyreadr.write_rds("batch_results.rds", results_df) # Write it to an .rds file so I can analyse it in R.
print("Finished saving .rds file")