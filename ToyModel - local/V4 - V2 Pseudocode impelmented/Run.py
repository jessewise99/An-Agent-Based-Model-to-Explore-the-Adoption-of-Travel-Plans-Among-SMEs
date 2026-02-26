############################################
#     Toy model prototype - Run            #
#     Date: 2026-02-04                     #
#     Author: Jesse Wise                   #
#     Purpose: Implementing Pseudocode V2  #
############################################

# Note: I have used Chat GPT to help me learn how to use Mesa, and to write this code. All errors are my own. 
# Be aware that MESA will take 3.11 or higher #pip install --upgrade mesa[rec] 
# ter Hoeven, E., Kwakkel, J., Hess, V., Pike, T., Wang, B., rht, & Kazil, J. (2025). Mesa 3: Agent-based modeling with Python in 2025. Journal of Open Source Software, 10(107), 7668. https://doi.org/10.21105/joss.07668

# This file is used to run the model. To do so, you must import the model and agents from their respective files.
# This file should choose parameters, activate shocks, run the model and plot results.

from Model import AdoptionModel # From the file Model.py, import the AdoptionModel class.
from Agents import FirmAgent # From the file Agents.py, import the FirmAgent class.
import numpy as np #Has multi-dimensional arrays and matrices. Has a large collection of mathematical functions to operate on these arrays.
import pandas as pd # Data manipulation and analysis.
import seaborn as sns # Data visualization tools.
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import mesa
import pyreadr # To write an .rds file
from collections import defaultdict
from collections import Counter

######################################################################### Running the model #########################################################################

# These parameters will need to be tuned and calibrated: learning_rate, realism_pull_constraints, realism_pull_sociallyInfluencedVars, competitor_inference_increment

model = AdoptionModel(
    num_agents= 100, # Set how many agents there are in the model. This needs to be <= the number of firms in the data file.
    learning_rate = 0.9,									# This is the rate at which firms learn from other firms
    competitor_inference_increment=0.01, # This is how much an agent's perceived benefits increases or decreases depening on their compeitors adoption stage. (at the moment = to learning rate* learning)
    realism_pull_constraints = 0.05,								# For time and money constraints set the realism pull as higher for these very objective concepts
    realism_pull_sociallyInfluencedVars = 0.01,						# For benefits, costs, and knowledge, the realism pull is lower as these are more subjective likely to be swayed by social influence
    organisationalReadiness_min = 0.4367,										# This is the organisational readiness threshold, if exceeded they may be able to adopt
    publicTransport_min = 0.5883,										# This is the public transport threshold, if exceeded they may be able to adopt
    resource_min = 0.5683,										# This is resource threshold, if exceeded they may be able to adopt
    knowledge_min = 0.4667,									# This is the knowledge threshold, if exceeded they may be able to adopt
    obj_net_benefit_min =	246,					# This is the lower threshold for the net benefits (£) an SME can expect per employee per year, according to the RAS project
    obj_net_benefit_max =	413,					# This is the upper threshold for the net benefits (£) an SME can expect per employee per year, according to the RAS project
    active_shocks = None, #{"caseStudy", "subsidy"} # These are the policies in effect. Needs to be a set.
    shock_parameters = None#{"accreditationAward": 0.25} # These are the strengths of the policies, it needs to be a dictionary. It will look like this {"caseStudy": 0.3, "subsidy": 0.2}
    )  # Create an instance of the AdoptionModel with the above parameters.

T =  10 										# Set how many  time steps the program will run for
for i in range(T):                              # For each tick in the time range....
    model.step()                                #... take a step in the model

######################################################################### Visualisations #########################################################################

# --- Distribution of likelihood of adoption ---
agent_data = model.datacollector.get_agent_vars_dataframe().reset_index() #Retrieve agent-level data
beginning_data = agent_data[agent_data["Step"] == 1]
last_step = agent_data["Step"].max()
final_data = agent_data[agent_data["Step"] == last_step]

# Plot histogram of adoption probabilities at beginning
plt.figure(figsize=(16, 10))
sns.histplot(beginning_data["Adoption Probability"], bins=20)
plt.title("Distribution of Intention to Adopt a Workplace Travel Plan at Beginning of Simulation")
plt.xlabel("Probability of adopting a workplace travel plan")
plt.ylabel("Number of agents")
plt.show()

# Plot histogram of adoption probabilities at end
plt.figure(figsize=(16, 10))
sns.histplot(final_data["Adoption Probability"], bins=20)
plt.title("Distribution of Intention to Adopt a Workplace Travel Plan at Final Tick")
plt.xlabel("Probability of adopting a workplace travel plan")
plt.ylabel("Number of agents")
plt.show()

# Plot histogram of perceived NBs at beginning
plt.figure(figsize=(16, 10))
sns.histplot(beginning_data["Perceived Net Benefit"], bins=20)
plt.title("Distribution of Perceived Net Benefit of Adoption at Beginning of Simulation")
plt.xlabel("Perceived Net Benefit of adopting a workplace travel plan")
plt.ylabel("Number of agents")
plt.show()

# Plot histogram of perceived NBs at end
plt.figure(figsize=(16, 10))
sns.histplot(final_data["Perceived Net Benefit"], bins=20)
plt.title("Distribution of  Perceived Net Benefit of Adoption at Final Tick")
plt.xlabel("Perceived Net Benefit of adopting a workplace travel plan")
plt.ylabel("Number of agents")
plt.show()

# --- Adopters in the network ---
G = model.G

# Align probabilities to graph node order (important!)
probs_by_id = {a.unique_id: a.prob_adoption for a in model.agents}
probs = [probs_by_id[n] for n in G.nodes()]

norm = plt.Normalize(vmin=0, vmax=1)
cmap = cm.get_cmap("RdYlGn")
node_colors = [cmap(norm(p)) for p in probs]
node_sizes = [max(30, p * 800) for p in probs]  # avoid invisible nodes

# Diagnostics: connected components
comp_sizes = sorted([len(c) for c in nx.connected_components(G)], reverse=True)
print("Connected components:", len(comp_sizes))
print("Largest component sizes:", comp_sizes[:10])

# Choose what you want to label by
group_attr = "postcode"  # change to "region" only after you actually add it to nodes

# Check attribute exists
attrs = nx.get_node_attributes(G, group_attr)
print(f"Nodes with {group_attr} set:", len(attrs), "out of", G.number_of_nodes())

# If nothing is set, stop early with a clear message
if len(attrs) == 0:
    raise ValueError(
        f"No node attribute '{group_attr}' found. "
        f"Add it when creating nodes in build_network_from_agents, or choose a different attribute."
    )

counts = Counter(attrs.values())
print("Number of groups:", len(counts))
print("Top 20 group sizes:")
for k, v in counts.most_common(20):
    print(k, v)

# Layout and draw
plt.figure(figsize=(16, 16))
pos = nx.spring_layout(G, seed=42)

nx.draw(
    G, pos,
    with_labels=False,
    node_color=node_colors,
    node_size=node_sizes,
    edge_color="gray",
    width=0.3,
    alpha=0.25
)

# Group nodes and label group centroids
groups = defaultdict(list)
for node, data in G.nodes(data=True):
    key = data.get(group_attr, "Unknown")
    groups[key].append(node)

min_group_size = 5
for key, nodes in groups.items():
    if len(nodes) < min_group_size:
        continue

    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    cx, cy = float(np.mean(xs)), float(np.mean(ys))

    plt.text(
        cx, cy, str(key),
        fontsize=12,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85),
        zorder=10
    )

plt.title("Firm Network: adoption probability")
plt.axis("off")
plt.show()


# --- Number of adopters over time ---
model_data = model.datacollector.get_model_vars_dataframe().reset_index() #Retrieve model-level data

# Plot Adoption over time
plt.figure(figsize=(16, 10))
sns.lineplot(x="index", y="Num_Developers", data=model_data, marker="o") # This isn't plotting correctly for some reason?
plt.title("Adoption Curve: Number of Firms Developing a WTP Over Time")
plt.xlabel("Step")
plt.ylabel("Number of Firms Developing a WTP")
plt.show()

plt.figure(figsize=(16, 10))
sns.lineplot(x="index", y="Num_Adopters", data=model_data, marker="o")
plt.title("Adoption/Infection Curve: Number of Firms with a WTP Over Time")
plt.xlabel("Step")
plt.ylabel("Number of Firms Who Have Adopted a WTP")
plt.show()

# Plot S-curve of probability of adoption over time
# Compute average adoption probability per step
avg_prob = agent_data.groupby("Step")["Adoption Probability"].mean().reset_index()

# Plot it
plt.figure(figsize=(16, 10))
sns.lineplot(x="Step", y="Adoption Probability", data=avg_prob, marker="o")
plt.title("Average Probability of Adoption Over Time")
plt.xlabel("Step")
plt.ylabel("Average Probability")
plt.show()

# ######################################################################### Parameter Sweeps with Batch Runner #########################################################################
# You need to run data collection and the batch runner too see here https://mesa.readthedocs.io/latest/overview.html

#--- Setting the parameters for the batch runner ---
# params = {"num_agents": 1000, # Set how many agents there are in the model. This needs to be <= the number of firms in the data file.
#     "learning_rate":  [0.5, 0.8],								# This is the rate at which firms learn from other firms
#     "competitor_inference_increment":  [0.5,0.8], # This is how much an agent's perceived benefits increases or decreases depening on their compeitors adoption stage. (at the moment = to learning rate* learning)
#     "realism_pull_constraints" :  [0.02,0.05],								# For time and money constraints set the realism pull as higher for these very objective concepts
#     "realism_pull_sociallyInfluencedVars" :  [0.02,0.05],						# For benefits, costs, and knowledge, the realism pull is lower as these are more subjective likely to be swayed by social influence
#     "organisationalReadiness_min" : 0.4367,										# This is the organisational readiness threshold, if exceeded they may be able to adopt
#     "publicTransport_min" : 0.5883,										# This is the public transport threshold, if exceeded they may be able to adopt
#     "resource_min" : 0.5683,										# This is resource threshold, if exceeded they may be able to adopt
#     "knowledge_min" : 0.4667,									# This is the knowledge threshold, if exceeded they may be able to adopt
#     "obj_net_benefit_min" :	246,					# This is the lower threshold for the net benefits (£) an SME can expect per employee per year, according to the RAS project
#     "obj_net_benefit_max" :	413,					# This is the upper threshold for the net benefits (£) an SME can expect per employee per year, according to the RAS project
#     "active_shocks" : None, #{"caseStudy", "subsidy"} # These are the policies in effect. Needs to be a set.
#     "shock_parameters" : None #{"accreditationAward": 0.25} # These are the strengths of the policies, it needs to be a dictionary. It will look like this {"caseStudy": 0.3, "subsidy": 0.2}
# } 


# #--- Running the batch runner ---
# results = mesa.batch_run(
#     AdoptionModel,
#     parameters=params,
#     iterations=1, # The number of iterations to run each parameter combination for. Optional. If not specified, defaults to 1.
#     max_steps=10, # How many steps to run the model for
#     number_processes=1,
#     data_collection_period=1, # The length of the period (number of steps) after which the model and agent reporters collect data. Optional. If not specified, defaults to -1, i.e. only at the end of each episode.
#     display_progress=True,
# )

# #--- Analysis and visualisation of batch results ---
# results_df = pd.DataFrame(results)
# print(f"The results have {len(results)} rows.")
# print(f"The columns of the data frame are {list(results_df.keys())}.")

# pyreadr.write_rds("batch_results.rds", results_df) # Write it to an .rds file so I can analyse it in R.
# print("Finished saving .rds file")