############################################
#     Toy model prototype - Model          #
#     Date: 2026-02-04                     #
#     Author: Jesse Wise                   #
#     Purpose: Implementing Pseudocode V2  #
############################################

# Think of this file like the 'world' - what exists globally and what rules apply system wide?
# It shoud store global parameters and baselines, store which shocks are active, define how policies modify constraints, and provide ehlper functions that agents can query.
# Note: I have used Chat GPT to help me learn how to use Mesa, and to write this code. All errors are my own. 
import itertools
from mesa import Model, DataCollector
from mesa.space import NetworkGrid # NetworkGrid is mesa's modern tools for discrete spatial and network structures
import networkx as nx  # This is used to create and manipulate networks https://networkx.org/documentation/stable/reference/introduction.html
from Agents import FirmAgent  # Import the FirmAgent class from the agent file
import numpy as np
import pandas as pd


class AdoptionModel(Model): # Everything idented inside the class is part of the class. Remember, Mesa is OOP, so the model itself is an object
    """Continuing to build on my toy model. 

    All agents have some capability, opportunity, and motivation to adopt a workplace travel plan - defined by 6 subdomains + awareness.
    At each time step they learn from their peers, and update their own beliefs, including feasibility (resources, knowledge, access to PT, and organisational readiness) and their motivation (=costs-benefits).
    They re-assess what level of adoption is right for them, and change their status accordingly.
    We want to test whether this theory can explain observed adoption patterns.
        
    Attributes:
        num_agents (int): The number of agents in the model.
        grid (NetworkGrid): The space in which agents act.
        running (bool): Whether the model should continue running.
        datacollector (DataCollector): Collects data for analysis.
        active_shocks
    """

    def __init__(self, num_agents, learning_rate, realism_pull_sociallyInfluencedVars, realism_pull_constraints, 
                 obj_net_benefit_min, obj_net_benefit_max, organisationalReadiness_min, publicTransport_min, knowledge_min, 
                 resource_min, competitor_inference_increment, active_shocks, shock_parameters, seed=None): #_init_ means the model is being intialised
        super().__init__(seed=seed)  # This *initialises* the parent Model class and sets the random seed

        self.num_agents = num_agents
        self.running = True

        # Exogenous shocks
        self.active_shocks = set(active_shocks) if active_shocks is not None else set() # Should be set in run.py
        self.shock_efficacy = dict(shock_parameters) if shock_parameters is not None else {} # Should be set in run.py

        # Set learning parameters
        self.competitor_inference_increment = competitor_inference_increment
        self.learning_rate = learning_rate
        self.realism_pull_sociallyInfluencedVars = realism_pull_sociallyInfluencedVars
        self.realism_pull_constraints = realism_pull_constraints
        self.organisationalReadiness_min = organisationalReadiness_min
        self.publicTransport_min = publicTransport_min
        self.knowledge_min = knowledge_min
        self.resource_min = resource_min
        self.resource_min_base = resource_min
        self.organisationalReadiness_min_base = organisationalReadiness_min
        self.learning_rate_base = learning_rate
        self.obj_net_benefit_min = obj_net_benefit_min
        self.obj_net_benefit_max = obj_net_benefit_max

        # 1. Create agents and place them on nodes
        for i, node in enumerate(self.G.nodes):
            if i >= self.num_agents:
                break  # Safety check if graph has more nodes than num_agents
            attributes = self.G.nodes[node]
            agent = FirmAgent(
                self,
                sector=attributes["sector"],
                postcode=attributes["postcode"],
                network=attributes["network"],
                size=attributes["size"]
            )

            self.grid.place_agent(agent, node) # The agent is anchored to a node in Mesa's network grid, and get_neighbors() gives it access to peers via edges.]

        #2. Build network from these agents
        self.G = self.build_network()  # Create your network (spatial + business links)
        #3. Create a grid fro mthe network
        self.grid = NetworkGrid(self.G)  # Place agents on a network grid
        #4. Place agents on nodes

        #5. debug and collect data
        print("Num agents created:", self.num_agents)
        print("Num graph nodes:", self.G.number_of_nodes())
        print("Num graph edges:", self.G.number_of_edges())
        print("Agents placed on grid:", len(self.grid.get_all_cell_contents()))

        # Create a DataCollector to collect data for analysis
        self.datacollector = DataCollector(
            model_reporters={
                "Num_Developers": lambda m: m.count_adoption_stage("C. Is developing a WTP"), # Count (using a helper function) how many agents are in the 'developing a WTP' stage
                "Num_Adopters": lambda m: ( 
                    m.count_adoption_stage("D. Has a low efficacy WTP") # Count how many agents have a poorly implemented WTP
                    + m.count_adoption_stage("E. Has an effective WTP") # Count (and add) how many agents have a well implemented WTP
                    )
            },
            agent_reporters={
                # Add other variables as needed for debugging/plotting
                "Adoption Stage": "adoption_stage",
                "Adoption Probability": "prob_adoption"
            }
        ) # This is collecting a lot of data and causing bloat, may need to find a way to reduce this.

        self.datacollector.collect(self)  # Collect initial state (step 0)

    def step(self): # Now we are defining the methods, the things the model can do. This one advancces time
        """Advance the model by one step."""
        self.apply_active_shocks()         # Apply exogenous shocks
        # Activate agents
        self.agents.shuffle_do("step")  # Random activation of all agents
        self.datacollector.collect(self)  # Collect data after each step


    def build_network(self): # This is a helper methods, to construct my structure
        """ This creates a NetworkX graph where each node represents a firm and edges represent potential social influence."""
        G = nx.Graph()  # Create an empty undirected graph

        for a in self.agents:  # for all my agents, add a node to the graph, inclduing their attributes https://memgraph.github.io/networkx-guide/basics/#reading-graphs
            G.add_node(
                a.unique_id,
                postcode=a.postcode,
                sector=a.sector,
                network=a.network,
                size=a.size,
            )
        
        # Group ids by linkage keys
        by_postcode = {}
        by_network = {}

        for a in self.agents:
            by_postcode.setdefault(a.postcode, []).append(a.unique_id)
            if a.network is not None and a.network != "None":
                by_network.setdefault(a.network, []).append(a.unique_id)

    # Helper to add edges within a group
    def connect_group(id_list, reason):
        # reason is either "spatial" if they are in the same postcode or "network" if they are in the same network
        for i, j in itertools.combinations(id_list, 2):
            p1, s1, n1 = G.nodes[i]["postcode"], G.nodes[i]["sector"], G.nodes[i]["network"]
            p2, s2, n2 = G.nodes[j]["postcode"], G.nodes[j]["sector"], G.nodes[j]["network"]

            spatial_link = (p1 == p2) # If they are in the same postcode, then spatial link = TRUE
            network_link = (n1 is not None and n1 != "None" and n1 == n2) # If they are in the same network, then they have a network link

            if not (spatial_link or network_link):
                continue

            # Your existing edge type logic
            if (spatial_link and s1 != s2) or network_link: # If they are in network, or same geography and different sectors, then they are peers. Otherwise competitor
                edge_type = "peer"
            else:
                edge_type = "competitor"

            # Avoid overwriting an existing edge type in a way that loses information
            if G.has_edge(i, j):
                existing = G.edges[i, j].get("type")
                if existing != edge_type:
                    G.edges[i, j]["type"] = "peer"
                    G.edges[i, j]["types"] = sorted(set([existing, edge_type]))
                else:
                    G.add_edge(i, j, type=edge_type)
        return G

    # Connect within each postcode clique
    for ids in by_postcode.values():
        if len(ids) > 1:
            connect_group(ids, reason="spatial")

    # Connect within each business network clique
    for ids in by_network.values():
        if len(ids) > 1:
            connect_group(ids, reason="network")

    print("Agents in schedule:", len(self.schedule.agents))
    print("Agents on grid:", len(self.grid.get_all_cell_contents()))

        
        # # Now we add edges based on the attributes of the nodes
        # for i in G.nodes:
        #     for j in G.nodes:
        #         if i >= j:
        #             continue  # Avoid self-loops and duplicates

        #         p1, s1, n1 = G.nodes[i]["postcode"], G.nodes[i]["sector"], G.nodes[i]["network"]
        #         p2, s2, n2 = G.nodes[j]["postcode"], G.nodes[j]["sector"], G.nodes[j]["network"] # This may cause issues because each edge uses at least 100 bytes of memory https://memgraph.github.io/networkx-guide/faq#load-graph

        #         spatial_link = p1 == p2  # Firms are co-located
        #         network_link = n1 != "None" and n1 == n2  # Firms are in same business network

        #         if spatial_link or network_link:
        #             # Edge type logic
        #             if (spatial_link and s1 != s2) or network_link:
        #                 edge_type = "peer"
        #             else:
        #                 edge_type = "competitor"
        #             G.add_edge(i, j, type=edge_type)  # Add edge with type metadata

        # return G

    def count_adoption_stage(self, stage_label):
        """Counts how many agents are at a specific adoption stage."""
        return sum(1 for agent in self.agents if agent.adoption_stage == stage_label)
    
    ### Defining exogenous shocsks
    def activate_shock(self, name, efficacy): # This function activates a shock
        self.active_shocks.add(name) # add the name to the list of active shocks
        self.shock_efficacy[name]= efficacy # store its efficacy in the dictionary

    def deactivate_shock(self, name): # Deactivates a shock (reverse of above)
        self.active_shocks.discard(name)
        self.shock_efficacy.pop(name, None)

    def apply_active_shocks(self): # Need to do this for perceptual shocks
        for agent in self.agents:
            if "caseStudy" in self.active_shocks:
                self.caseStudy(agent, self.shock_efficacy["caseStudy"])
            if "proofOfROI" in self.active_shocks:
                self.proofOfROI(agent, self.shock_efficacy["proofOfROI"])

    ## Perceptual shocks - those which modify internal agents states
    def caseStudy(self, agent, policy_efficacy):
        exposure = 1 if agent.beliefs["awareness"] else 0.0 # An agent is exposed if at least one of their connections has a plan (a firm is aware if someone adopts near them (apart from at time step 0))
        sensitivity = agent.shock_sensitivity["caseStudy"]["value"]
        delta = policy_efficacy * exposure * sensitivity
        agent.beliefs["knowledge"] = np.clip(agent.beliefs["knowledge"] + delta, 0.0, 1.0)

    def proofOfROI(self, agent, policy_efficacy, exposure =1.0):
        sensitivity = agent.shock_sensitivity["proofOfROI"]["value"]
        delta = policy_efficacy * exposure * sensitivity
        agent.beliefs["motivations"] = np.clip(agent.beliefs["motivations"] * (1 + delta), 0.0, 1.0)  # Increase a firm's perceived subjective benefits of adoption by x% where x is the policy efficacy  np.clip keeps it bounded between 0 and 1
        agent.beliefs["perceivedBarriers"] = np.clip(agent.beliefs["perceivedBarriers"] * (1 - delta), 0.0, 1.0) # Reduce a frim's subjective perceived barriers to adoption by x% where x is the policy efficacy 
   

    ## Structural shocks - those which modify thresholds and rates

    # Policy champions
    def effective_organisationalReadiness_min(self, agent, exposure = 1.0):
        base = self.organisationalReadiness_min
        if "policyChampion" not in self.active_shocks:
            return base

        efficacy = self.shock_efficacy["policyChampion"]
        sensitivity = agent.shock_sensitivity["policyChampion"]["value"]
        delta = efficacy * exposure * sensitivity
        delta = max(0.0, min(1.0, delta))  # Making sure delta is bounded between 0 and 1

        return max(0.0, base * (1 - delta))
        
    # Subsidy
    def effective_resource_min(self, agent, exposure = 1.0):
        base = self.resource_min

        if "subsidy" not in self.active_shocks:
            return base # If susbidy isn't active use the og value

        efficacy = self.shock_efficacy["subsidy"]
        sensitivity = agent.shock_sensitivity["subsidy"]["value"]
        delta = efficacy * exposure * sensitivity
        delta = max(0.0, min(1.0, delta))  # Making sure delta is bounded between 0 and 1

        return max(0.0, base * (1 - delta))
    
    # Accreditation
    def effective_learning_rate(self, agent, exposure=1.0):
        base = self.learning_rate
        if "accreditationAward" not in self.active_shocks:
            return base
        efficacy = self.shock_efficacy["accreditationAward"]
        sensitivity = agent.shock_sensitivity["accreditationAward"]["value"]
        delta = max(0.0, min(1.0, efficacy * exposure * sensitivity))
        return max(0.0, min(1.0, base * (1 + delta)))
    

    def effective_competitor_inference_increment(self, agent, exposure=1.0):
        base = self.competitor_inference_increment
        if "accreditationAward" not in self.active_shocks:
            return base
        efficacy = self.shock_efficacy["accreditationAward"]
        sensitivity = agent.shock_sensitivity["accreditationAward"]["value"]
        delta = max(0.0, min(1.0, efficacy * exposure * sensitivity))
        return max(0.0, min(1.0, base * (1 + delta)))

    # Will add this if I have time... 
    # Learning shocks - those which modify decision rules
    #def businessContactRecommendedIt(self): # One competitor is treated as a peer for one tick 
    #def manyBusinessContactsRecommendedIt(self): # All competitors are treated as a peer for one tick 