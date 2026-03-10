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
import numpy as np #Has multi-dimensional arrays and matrices. Has a large collection of mathematical functions to operate on these arrays.
import pandas as pd # Data manipulation and analysis.
import seaborn as sns # Data visualization tools.
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm

######################################################################### Running the model #########################################################################

# These parameters will need to be tuned and calibrated

model = AdoptionModel(
    num_agents= 1000, # Set how many agents there are in the model. This needs to be <= the number of firms in the data file.
    learning_rate = 0.1,									# This is the rate at which firms learn from other firms
    realism_pull_constraints = 0.5,								# For time and money constraints set the realism pull as higher for these very objective concepts
    realism_pull_sociallyInfluencedVars = 0.05,						# For benefits, costs, and knowledge, the realism pull is lower as these are more subjective likely to be swayed by social influence
    time_min = 0.2,										# This is the time threshold, if exceeded they may be able to adopt
    money_min = 0.2,										# This is money threshold, if exceeded they may be able to adopt
    knowledge_min = 0.2,									# This is the knowledge threshold, if exceeded they may be able to adopt
    competitor_inference_increment=0.05, # This is how much an agent's perceived benefits increases or decreases depening on their compeitors adoption stage.
    )  # Create an instance of the AdoptionModel with the above parameters.
### You need to finish passing the rest of the parameters once you've desinged the model. This is just a placeholder.

T =  20 										# Set how many  time steps the program will run for
for i in range(T):
    model.step()


# You need to run data collection and the batch runner too see here https://mesa.readthedocs.io/latest/overview.html

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

# --- Adopters in the network ---

# Access the underlying graph
G = model.G  # This is your network from build_network()

# Assign colours based on adoption probability
probs = [agent.prob_adoption for agent in model.agents]
norm = plt.Normalize(vmin=0, vmax=1)
cmap = cm.get_cmap('RdYlGn')  # Explicitly get the RdYlGn colormap

node_colors = [cmap(norm(p)) for p in probs]
node_sizes = [p * 800 for p in probs]

# Draw the network
plt.figure(figsize=(16, 16))
pos = nx.spring_layout(G, seed=42)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=node_colors,
    node_size=node_sizes,
    font_size=8,
    edge_color="gray"
)
plt.title("Firm Network: WTP Adoption Probability (Red= Non-adopter, Green= High Probability of Adoption)")
plt.show()

# --- Number of adopters over time ---
model_data = model.datacollector.get_model_vars_dataframe().reset_index() #Retrieve model-level data

# Plot S-curve of adoption over time
plt.figure(figsize=(16, 10))
sns.lineplot(x="index", y="Num_Developers", data=model_data, marker="o")
plt.title("Adoption Curve: Number of Firms Developing a WTP Over Time")
plt.xlabel("Step")
plt.ylabel("Number of Firms Developing a WTP")
plt.show()

plt.figure(figsize=(16, 10))
sns.lineplot(x="index", y="Num_Adopters", data=model_data, marker="o")
plt.title("Adoption Curve: Number of Firms Adpopting Over Time")
plt.xlabel("Step")
plt.ylabel("Number of Firms Who Have Adopted a low-efficacy WTP")
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

# ######################################################################### Batch Runner #########################################################################
# # See tutorial here https://mesa.readthedocs.io/latest/tutorials/7_batch_run.html

# #--- Setting the parameters for the batch runner ---
# params = {T=10, # T= number of time steps
#           N= 100, # N= number of agents 
#           learning_rate = 0.3, # This is the rate at which firms learn from other firms 
#           realism_pull_constraints = 0.3, ## For time and money constraints set the realism pull as higher for these very objective concepts
#           realism_pull_sociallyInfluencedVars = 0.1, # For benefits, costs, and knowledge, the realism pull is lower as these are more subjective likely to be swayed by social influence
#           time_min = 0.2, # this is the time threshold, if exceeded they may be able to adopt
#           money_min = 0.3, # This is money threshold, if exceeded they may be able to adopt
#           knowledge_min = 0.4, #This is the knowledge threshold, if exceeded they may be able to adopt
#           "n": range(5, 105, 5)} 


# #--- Running the batch runner ---
# results = mesa.batch_run(
#     AdoptionModel,
#     parameters=params,
#     iterations=5, # The number of iterations to run each parameter combination for. Optional. If not specified, defaults to 1.
#     max_steps=100, # Run each model for 100 steps. I'm not sure if I need this because I set T above.
#     number_processes=1,
#     data_collection_period=1, # The length of the period (number of steps) after which the model and agent reporters collect data. Optional. If not specified, defaults to -1, i.e. only at the end of each episode.
#     display_progress=True,
# )

# #--- Analysis and visualisation of batch results ---
# results_df = pd.DataFrame(results)
# print(f"The results have {len(results)} rows.")
# print(f"The columns of the data frame are {list(results_df.keys())}.")

