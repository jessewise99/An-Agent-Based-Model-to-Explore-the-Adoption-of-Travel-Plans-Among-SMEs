############################################
#     Toy model prototype - Model          #
#     Date: 2025-07-01                     #
#     Author: Jesse Wise                   #
#     Purpose: To learn how Mesa works     #
#     and to test model file               #
############################################

from mesa import Model, DataCollector
from mesa.space import NetworkGrid # NetworkGrid is mesa's modern tools for discrete spatial and network structures
import networkx as nx  # This is used to create and manipulate networks https://networkx.org/documentation/stable/reference/introduction.html
from Agents import FirmAgent  # Import the FirmAgent class from the agent file
import numpy as np
import pandas as pd


class AdoptionModel(Model): # Everything idented inside the class is part of the class. Remember, Mesa is OOP, so the model itself is an object
    """A toy model to explore how Mesa works, and the basis for my ABM. 

    All agents have some capability, opportunity, and motivation to adopt a workplace travel plan.
    At each time step they learn from their peers, and update their own beliefs, including feasibility (time and knowledge) and their motivation (=costs-benefits).
    They re-assess what level of adoption is right for them, and change their status accordingly.
    We want to test whether this theory can explain an s-shaped adoption curve.
        
    Attributes:
        num_agents (int): The number of agents in the model.
        grid (NetworkGrid): The space in which agents act.
        running (bool): Whether the model should continue running.
        datacollector (DataCollector): Collects data for analysis.
    """

    def __init__(self, num_agents, learning_rate, realism_pull_sociallyInfluencedVars, realism_pull_constraints, time_min, money_min, knowledge_min, competitor_inference_increment, seed=None): #_init_ means the model is being intialised
        super().__init__(seed=seed)  # This *initialises* the parent Model class and sets the random seed
        #self.seed = seed
        self.num_agents = num_agents
        self.running = True

        # Set learning parameters
        self.competitor_inference_increment = competitor_inference_increment
        self.learning_rate = learning_rate
        self.realism_pull_sociallyInfluencedVars = realism_pull_sociallyInfluencedVars
        self.realism_pull_constraints = realism_pull_constraints
        self.time_min = time_min
        self.money_min = money_min
        self.knowledge_min = knowledge_min


        self.G = self.build_network()  # Create your network (spatial + business links)
        self.grid = NetworkGrid(self.G)  # Place agents on a network grid

        # Create agents and place them on nodes
        for i, node in enumerate(self.G.nodes):
            if i >= self.num_agents:
                break  # Safety check if graph has more nodes than num_agents
            attributes = self.G.nodes[node]
            agent = FirmAgent(
                self,
                sector=attributes["sector"],
                postcode=attributes["postcode"],
                network_membership=attributes["network"],
                size=attributes["size"]
            )

            self.grid.place_agent(agent, node) # You place the agent on its corresponding network node in the Mesa NetworkGrid. [Correction: This doesn't mean the agent can move like in a physical grid. Instead, it means the agent is anchored to a node, and get_neighbors() gives it access to peers via edges.]

        # Create a DataCollector to collect data for analysis
        self.datacollector = DataCollector(
            model_reporters={
                "Num_Developers": lambda m: self.count_adoption_stage("C. Is developing a WTP"),
                "Num_Adopters": lambda m: self.count_adoption_stage("D. Has a low efficacy WTP")
            },
            agent_reporters={
                # Add other variables as needed for debugging/plotting
                "Adoption Stage": "adoption_stage",
                "Adoption Probability": "prob_adoption"
            }
        )

        self.datacollector.collect(self)  # Collect initial state (step 0)

    def step(self): # Now we are defining the methods, the things the model can do. This one advancces time
        """Advance the model by one step."""
        self.agents.shuffle_do("step")  # Random activation of all agents
        self.datacollector.collect(self)  # Collect data after each step

    def build_network(self): # This is a helper methods, to construct my structure
        """Builds a network of agents with spatial and business links.

        [Correction: This creates a NetworkX graph where each node represents a firm and edges represent potential social influence.]
        """
        G = nx.Graph()  # Create an empty undirected graph

        self.firm_data = pd.read_excel("Test_Data1000Firms.xlsx", sheet_name="Firms") # This loads my data on firm types

        # Assign static attributes for each firm node
        for _, row in self.firm_data.iterrows():
            firm_id = int(row["ID"])
            sector = row["Sector"]
            postcode = row["Postcode"]
            network = row["Network"]
            size = row["Size"]

            G.add_node(firm_id, postcode=postcode, sector=sector, network=network, size=size)  # Add nodes with attributes see tutorial here https://memgraph.github.io/networkx-guide/basics/#reading-graphs

        # Now we add edges based on the attributes of the nodes
        for i in G.nodes:
            for j in G.nodes:
                if i >= j:
                    continue  # Avoid self-loops and duplicates

                p1, s1, n1 = G.nodes[i]["postcode"], G.nodes[i]["sector"], G.nodes[i]["network"]
                p2, s2, n2 = G.nodes[j]["postcode"], G.nodes[j]["sector"], G.nodes[j]["network"] # This may cause issues because each edge uses at least 100 bytes of memory https://memgraph.github.io/networkx-guide/faq#load-graph

                spatial_link = p1 == p2  # Firms are co-located
                network_link = n1 != "None" and n1 == n2  # Firms are in same business network

                if spatial_link or network_link:
                    # Edge type logic
                    if (spatial_link and s1 != s2) or network_link:
                        edge_type = "peer"
                    else:
                        edge_type = "competitor"
                    G.add_edge(i, j, type=edge_type)  # Add edge with type metadata

        return G

    def count_adoption_stage(self, stage_label):
        """Counts how many agents are at a specific adoption stage."""
        return sum(1 for agent in self.agents if agent.adoption_stage == stage_label)

