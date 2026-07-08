############################################
#     Model  - Batch Runner                #
#     Date: 2026-07-17                     #
#     Author: Jesse Wise                   #
#     Purpose: Sensitivity Analysis -LHC   #
############################################

# Note: I have used Chat GPT to help me learn how to use Mesa, and to write this code. All errors are my own. 
# Be aware that MESA will take 3.11 or higher #pip install --upgrade mesa[rec] 
# ter Hoeven, E., Kwakkel, J., Hess, V., Pike, T., Wang, B., rht, & Kazil, J. (2025). Mesa 3: Agent-based modeling with Python in 2025. Journal of Open Source Software, 10(107), 7668. https://doi.org/10.21105/joss.07668

# This file is used for sensitivity analysis using latin hypercube sampling

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
import pyreadr # To write an .rds file
from collections import defaultdict
from collections import Counter
from itertools import product
from scipy.stats import qmc
import pandas as pd
import numpy as np

N_SAMPLES = 100     # number of parameter sets
N_RUNS = 20         # stochastic repeats per set
N=500               # Numgber of agents
T=31                # Number of sets

param_ranges = {
    "learning_rate": (0.2, 0.8),
    "competitor_inference_increment": (0.01, 0.4),
    "logit_pivot": (120, 220),
    "logit_steepness": (0.02, 0.08),
}

sampler = qmc.LatinHypercube(d=len(param_ranges), seed=123)
sample = sampler.random(n=N_SAMPLES)

lower_bounds = [v[0] for v in param_ranges.values()]
upper_bounds = [v[1] for v in param_ranges.values()]

scaled_sample = qmc.scale(sample, lower_bounds, upper_bounds)

param_sets = pd.DataFrame(
    scaled_sample,
    columns=param_ranges.keys()
)

print(param_sets.head())

# Enter the sampled parameter sets into my model

results = []

for sample_id, row in param_sets.iterrows():

    for run in range(N_RUNS):

        print(f"Running {row['learning_rate']}, {row['competitor_inference_increment']}, {row['logit_pivot']}, {row['logit_steepness']} run={run + 1}/{N_RUNS}")
        model = AdoptionModel(
            learning_rate=row["learning_rate"],
            competitor_inference_increment=row["competitor_inference_increment"],
            logit_pivot=row["logit_pivot"],
            logit_steepness=row["logit_steepness"],

            init_positive_shift=0.1,
            B_min_time=2,
            C_min_time=2,
            D_min_time=4,
            B_constraints=2,
            D_constraints=3,
            cap_first_tick_at="D. Has a WTP",
            collect_agent_data=False,
            num_agents=N,

            organisationalReadiness_min=0.4367,
            publicTransport_min=0.5883,
            resource_min=0.5683,
            knowledge_min=0.4667,
            obj_net_benefit_min=188,
            obj_net_benefit_max=250,
            active_shocks=None,
            shock_parameters=None
        )

        for year in range(T):
            model.step()

        df = model.datacollector.get_model_vars_dataframe().reset_index()

        df["SampleId"] = sample_id
        df["RunId"] = run

        for param in param_ranges.keys():
            df[param] = row[param]

        results.append(df)

results_df = pd.concat(results, ignore_index=True)

pyreadr.write_rds("batch_results_LHS_SensitivityAnalysis.rds", results_df)