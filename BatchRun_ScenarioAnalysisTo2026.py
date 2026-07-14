############################################
#     Model  - Batch Runner                #
#     Date: 2026-06-15                     #
#     Author: Jesse Wise                   #
#     Purpose: Scenario Analysis 1996 to 2025 #
############################################

# Note: I have used Chat GPT to help me learn how to use Mesa, and to write this code. All errors are my own. 
# Be aware that MESA will take 3.11 or higher #pip install --upgrade mesa[rec] 
# ter Hoeven, E., Kwakkel, J., Hess, V., Pike, T., Wang, B., rht, & Kazil, J. (2025). Mesa 3: Agent-based modeling with Python in 2025. Journal of Open Source Software, 10(107), 7668. https://doi.org/10.21105/joss.07668

# This file is used to run the model. To do so, you must import the model and agents from their respective files.
# This file should choose parameters, activate shocks, run the model and plot results.

from Model_NoRealismPullCompInfInc4MotPBonly import AdoptionModel # From the file Model.py, import the AdoptionModel class.
from Agents_Optimising import FirmAgent # From the file Agents.py, import the FirmAgent class.
import numpy as np #Has multi-dimensional arrays and matrices. Has a large collection of mathematical functions to operate on these arrays.
import pandas as pd # Data manipulation and analysis.
import seaborn as sns # Data visualization tools.
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.figure import Figure
import mesa
import random
import pyreadr # To write an .rds file
from collections import defaultdict
from collections import Counter


######################################################################### Parameter Sweeps with Batch Runner #########################################################################
# You need to run data collection and the batch runner too see here https://mesa.readthedocs.io/latest/overview.html

# I now run the model 20 times to take account of stochasticity. I run it for 1 extra tick to take it to 2026.

T =  31 										# The program runs for 31 years because I step the model forward 1 tick.
N =  500                                        # Set how many agents there are in the model.
N_RUNS= 100 

#- Using these to capture all the policies and their efficacies in one file
policy_names = [
    "infrastructureInvestment",
    "caseStudy",
    "proofOfROI",
    "subsidy",
    "policyChampion",
    "accreditationAward",
    ]

efficacies = [0.05, 0.10, 0.15]

scenario_results = []

#--- Setting the parameters for the batch runner ---
for policy in policy_names:
    for efficacy in efficacies:
      for seed in range(N_RUNS):
        print(f"Running {policy}, efficacy={efficacy}, seed={seed + 1}/{N_RUNS}") 
        model = AdoptionModel(
          learning_rate=  0.65,								# This is the rate at which firms learn from other firms
          competitor_inference_increment=  0.04, # This refers to mimetic isomorphism
          init_positive_shift=0.1,                                  # This is used for calibration of initial distributions of beliefs
          B_min_time= 2,
          C_min_time= 2,
          D_min_time= 4,
          B_constraints= 2,
          D_constraints= 3,
          logit_pivot= 180, # This forms part of a function which converts perceived net benefits into a p(adopt) score
          logit_steepness=0.04, # Ditto
          cap_first_tick_at="D. Has a WTP",
          collect_agent_data= False, # False while doing such large sweeps
          ## These are not changing, but I have to pass them in anyway
          num_agents= N, # Set how many agents there are in the model. 
          organisationalReadiness_min= 0.4367,										# This is the organisational readiness threshold, if exceeded they may be able to adopt
          publicTransport_min= 0.5883,										# This is the public transport threshold, if exceeded they may be able to adopt
          resource_min= 0.5683,										# This is resource threshold, if exceeded they may be able to adopt
          knowledge_min= 0.4667,									# This is the knowledge threshold, if exceeded they may be able to adopt
          obj_net_benefit_min= 188,					# This is the lower threshold for the net benefits (£) an SME can expect per employee per year, according to the RAS project
          obj_net_benefit_max= 250,					# This is the upper threshold for the net benefits (£) an SME can expect per employee per year, according to the RAS project
          seed=seed,
          active_shocks= {policy}, # These are the policies in effect. Needs to be a set.
          shock_parameters= {policy: efficacy}# These are the strengths of the policies, it needs to be a dictionary. 
          ) 
        
        for _ in range(T):
                    model.step()

        df = model.datacollector.get_model_vars_dataframe().reset_index()
        df["policy"] = policy
        df["efficacy"] = efficacy
        df["Seed"] = seed

        scenario_results.append(df)

#--- Analysis and visualisation of batch results ---
results_df = pd.concat(scenario_results, ignore_index=True)
print(f"The results have {len(results_df)} rows.")
print(f"The columns of the data frame are {list(results_df.keys())}.")

pyreadr.write_rds("batch_results_ScenarioAnalysis_woutAgentData_100Iterations.rds", results_df) # Write it to an .rds file so I can analyse it in R.
print("Finished saving .rds file")