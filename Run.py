############################################
#     Model  - Run                         #
#     Date: 2026-04-02                     #
#     Author: Jesse Wise                   #
#     Purpose: Implementing Pseudocode V2  #
############################################

# Note: I have used Chat GPT to help me learn how to use Mesa, and to write this code. All errors are my own. 
# Be aware that MESA will take 3.11 or higher #pip install --upgrade mesa[rec] 
# ter Hoeven, E., Kwakkel, J., Hess, V., Pike, T., Wang, B., rht, & Kazil, J. (2025). Mesa 3: Agent-based modeling with Python in 2025. Journal of Open Source Software, 10(107), 7668. https://doi.org/10.21105/joss.07668

# This file is used to run and visualise the model. To do so, you must import the model and agents from their respective files.
# This file should choose parameters, activate shocks, run the model and plot results.

from Model import AdoptionModel # From the file Model.py, import the AdoptionModel class.
from Agents import FirmAgent # From the file Agents.py, import the FirmAgent class.
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


######################################################################### Running the model #########################################################################

# These parameters will need to be tuned and calibrated: learning_rate, realism_pull_constraints, realism_pull_sociallyInfluencedVars, competitor_inference_increment

T =  28 										# The program runs for 28 years because I have data from 1997 to 2025.
N =  500                                        # Set how many agents there are in the model. 

model = AdoptionModel(
    num_agents= N, 
    learning_rate = 1,									# This is the rate at which firms learn from other firms
    competitor_inference_increment=0.40, # This is how much an agent's perceived benefits increases or decreases depening on their compeitors adoption stage.
    realism_pull_constraints = 0.05,								# Higher number means that beliefs as less influenced.
    init_positive_shift = 0.3,                                  # This is used for calibration of initial distributions of beliefs
    collect_agent_data= True,
    organisationalReadiness_min= 0.4367,										# This is the organisational readiness threshold, if exceeded they may be able to adopt
    publicTransport_min= 0.5883,										# This is the public transport threshold, if exceeded they may be able to adopt
    resource_min=.5683,										# This is resource threshold, if exceeded they may be able to adopt
    knowledge_min= 0.4667,									# This is the knowledge threshold, if exceeded they may be able to adopt
    obj_net_benefit_min =	188,					# This is the lower threshold for the net benefits (£) an SME can expect per employee per year, according to the RAS project
    obj_net_benefit_max =	250,					# This is the upper threshold for the net benefits (£) an SME can expect per employee per year, according to the RAS project
    active_shocks = None,# {"subsidy", "proofOfROI"}, # These are the policies in effect. Needs to be a set.
    shock_parameters = None, #{"subsidy": 0.5, "proofOfROI":0.3} # These are the strengths of the policies, it needs to be a dictionary. It will look like this {"caseStudy": 0.3, "subsidy": 0.2}
    )  # Create an instance of the AdoptionModel with the above parameters.

######################################################################### Visualisations #########################################################################

# --- Visualsing the model ---
G = model.G
pos = nx.spring_layout(G, seed=42)  # compute once

stage_colours = {
    "A. No intention": "red",
    "B. May consider": "orange",
    "C. Is developing a WTP": "yellow",
    "D. Has a WTP": "green"}

fig, ax = plt.subplots(figsize=(10, 10))

def draw_frame(frame):
    ax.clear() # Wipes each frame clean so the next one can be drawn

    # One tick only
    model.step() # The animation advances the step, so th emodel is actively playing while the animation runs
  
    node_colors = [stage_colours[model.grid.get_cell_list_contents([n])[0].adoption_stage] for n in G.nodes()] # Iterating over every node, dispaly the colour based on adoption stage
    node_sizes  = [max(50, model.grid.get_cell_list_contents([n])[0].prob_adoption*500) for n in G.nodes()] # Gives each node a size based on p(adopt)
    pos = nx.spring_layout(G, k=1.5, seed=42)
    nx.draw(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color="gray",
        with_labels=False,
    ) # draws the graph
    ax.set_title(f"Tick {frame + 1}") # adds a frame counter in the title

ani = animation.FuncAnimation(fig, draw_frame, frames=T, repeat=False, interval=500) # creates the animation. FuncAnimation calls draw_frame(frame) once per frame -> which calls model.step()
plt.show() # Displays the animattion

model_data = model.datacollector.get_model_vars_dataframe().reset_index() # extracts datacollector output after animation or model.step() wasn't called in every frame

# Sanity checks. If this prints fewer than T rows, it usually means either the animation exited early
print("Collected steps:", len(model_data))
print(model_data.head())
print(model_data.tail())

# --- Gather data ---

agent_data = model.datacollector.get_agent_vars_dataframe().reset_index() #Retrieve agent-level data
beginning_data = agent_data[agent_data["Step"] == 1]
last_step = agent_data["Step"].max()
final_data = agent_data[agent_data["Step"] == last_step]

model_data = model.datacollector.get_model_vars_dataframe().reset_index() #Retrieve model-level data
print("Describing the number of adopters", model_data["Num_Adopters"].describe())
print("Any adopters at all:", (model_data["Num_Adopters"] > 0).any())
agent_data = model.datacollector.get_agent_vars_dataframe()
print("Descibe adpoption probability",agent_data["Adoption Probability"].describe())
print("The proportion of agents with p(adopt)>=.79 is",(agent_data["Adoption Probability"] >= 0.79).mean())
print("The proportion of agents with p(adopt)>=.85 is", (agent_data["Adoption Probability"] >= 0.85).mean())

# --- Visualsing the model Adopters in the network ---
# Plot histogram of adoption probabilities at beginning
plt.figure(figsize=(16, 10))
sns.histplot(beginning_data["Adoption Probability"], bins=20)
plt.title("Distribution of Intention to Adopt a Workplace Travel Plan at Beginning of Simulation")
plt.xlabel("Probability of adopting a workplace travel plan")
plt.ylabel("Number of agents")
plt.grid()
plt.ylim(0, N)
plt.show()

# Plot histogram of adoption probabilities at end
plt.figure(figsize=(16, 10))
sns.histplot(final_data["Adoption Probability"], bins=20)
plt.title("Distribution of Intention to Adopt a Workplace Travel Plan at Final Tick")
plt.xlabel("Probability of adopting a workplace travel plan")
plt.ylabel("Number of agents")
plt.grid()
plt.ylim(0, N)
plt.show()

# Plot histogram of perceived NBs at beginning
plt.figure(figsize=(16, 10))
sns.histplot(beginning_data["Perceived Net Benefit"], bins=20)
plt.title("Distribution of Perceived Net Benefit of Adoption at Beginning of Simulation")
plt.xlabel("Perceived Net Benefit of adopting a workplace travel plan")
plt.ylabel("Number of agents")
plt.grid()
plt.ylim(0, N)
plt.show()

# Plot histogram of perceived NBs at end
plt.figure(figsize=(16, 10))
sns.histplot(final_data["Perceived Net Benefit"], bins=20)
plt.title("Distribution of  Perceived Net Benefit of Adoption at Final Tick")
plt.xlabel("Perceived Net Benefit of adopting a workplace travel plan")
plt.ylabel("Number of agents")
plt.ylim(0, N)
plt.grid()
plt.show()

# --- Number of adopters over time ---
# Plot Adoption over time
plt.figure(figsize=(16, 10))
sns.lineplot(x="index", y="Num_Developers", data=model_data, marker="o") 
plt.title("Adoption Curve: Number of Firms Developing a WTP Over Time")
plt.xlabel("Step")
plt.ylabel("Number of Firms Developing a WTP")
plt.grid()
plt.ylim(0, N)
plt.show()

plt.figure(figsize=(16, 10))
sns.lineplot(x="index", y="Num_Adopters", data=model_data, marker="o")
plt.title("Adoption/Infection Curve: Number of Firms with a WTP Over Time")
plt.xlabel("Step")
plt.ylabel("Number of Firms Who Have Adopted a WTP")
plt.grid()
plt.ylim(0, N)
plt.show()

# Compute average adoption probability per step
avg_prob = agent_data.groupby("Step")["Adoption Probability"].mean().reset_index()

# Plot it
plt.figure(figsize=(16, 10))
sns.lineplot(x="Step", y="Adoption Probability", data=avg_prob, marker="o")
plt.title("Average Probability of Adoption Over Time")
plt.xlabel("Step")
plt.ylabel("Average Probability")
plt.grid()
plt.ylim(0, 1)
plt.show()