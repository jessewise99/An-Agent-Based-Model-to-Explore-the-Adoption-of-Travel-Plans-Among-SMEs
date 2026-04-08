############################################
#     Model - Model                        #
#     Date: 2026-04-02                     #
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
    They learn different from strong ties (peers) and weak ties (competitors). I just assume that those in a network have strong ties, while you have weak ties with your neighbours.
    They re-assess what level of adoption is right for them, and change their status accordingly.
    We want to test whether this theory can explain observed adoption patterns.
        
    Mesa 3.4.1 notes:
    - Agents auto register with the model when instantiated
    - self.agents is an AgentSet
    - self.agents.shuffle_do("step") is supported

    Attributes:
        num_agents (int): The number of agents in the model.
        grid (NetworkGrid): The space in which agents act.
        running (bool): Whether the model should continue running.
        datacollector (DataCollector): Collects data for analysis.
    """
    grid: NetworkGrid
    G: nx.Graph
    resource_min: float
    organisationalReadiness_min: float
    knowledge_min: float
    publicTransport_min: float
    competitor_inference_increment: float
    learning_rate: float
    obj_net_benefit_max: int
    obj_net_benefit_min: int


    def __init__(self, num_agents:int, learning_rate:float, competitor_inference_increment:float,
                 realism_pull_constraints:float, organisationalReadiness_min:float,
                 publicTransport_min:float, knowledge_min:float, resource_min:float,
                 obj_net_benefit_min:int, obj_net_benefit_max:int, 
                 init_positive_shift: float,
                 collect_agent_data:bool, 
                 shock_parameters=None,seed=None, active_shocks=None, debug=True): #_init_ means the model is being intialised
        super().__init__(seed=seed)  # This *initialises* the parent Model class and sets the random seed

        self.num_agents = int(num_agents) # Makes sure its an integer
        self.running = True
        self.debug=bool(debug) # This helps me debug

        # Used for calibrating starting distributions of beleifs
        self.init_positive_shift = init_positive_shift
        self.init_barrier_shift = init_positive_shift

        # Exogenous shocks
        self.active_shocks = set(active_shocks) if active_shocks is not None else set() # Should be set in run.py
        self.shock_efficacy = dict(shock_parameters) if shock_parameters is not None else {} # Should be set in run.py

        # Set learning parameters
        self.competitor_inference_increment = competitor_inference_increment
        self.learning_rate = learning_rate
        self.realism_pull_sociallyInfluencedVars =    0.5 * realism_pull_constraints						# For benefits, costs, and knowledge, the realism pull is lower as these are more subjective likely to be swayed by social influence
        self.realism_pull_constraints = realism_pull_constraints
        
        # Set thresholds and bounds
        self.organisationalReadiness_min = organisationalReadiness_min
        self.publicTransport_min = publicTransport_min
        self.knowledge_min = knowledge_min
        self.resource_min = resource_min
        self.resource_min_base = resource_min
        self.organisationalReadiness_min_base = organisationalReadiness_min
        self.learning_rate_base = learning_rate

        # Set objective values
        self.obj_net_benefit_min = obj_net_benefit_min
        self.obj_net_benefit_max = obj_net_benefit_max

        # 1. Create agents first (they sample their own attributes inside FirmAgent which is my definintion of an agent)
        created_agents = [FirmAgent(self) for _ in range(self.num_agents)]

        # 2. Build networkX graph from these agents' attributes
        self.G = self.build_network_from_agents(created_agents)

        # 3. Create grid from the network. Mesa creates a network based “grid” wrapper around my networkX graph, it doesn't change it. This provides methods like get_neighbors and handles agent placement on nodes.
        self.grid = NetworkGrid(self.G)

        # 4. Place each agent on its own node id
        for a in created_agents:
            self.grid.place_agent(a, a.unique_id)

        if self.debug:
            print("Debug model init")
            print("Requested agents:", self.num_agents)
            print("Agents registered on model:", len(self.agents))
            print("Graph nodes:", self.G.number_of_nodes())
            print("Graph edges:", self.G.number_of_edges())
            print("Agents placed on grid:", len(self.grid.get_all_cell_contents()))

            # Sanity check a single agent
            a0 = created_agents[0]
            deg = len(list(self.G.neighbors(a0.unique_id)))
            print("Example agent id:", a0.unique_id, "pos:", a0.pos, "degree:", deg)

        # Data collection
        # self.datacollector = DataCollector(
        #     model_reporters={
        #         "Num_Considering": lambda m: m.count_adoption_stage("B. May consider"),
        #         "Num_Developers": lambda m: m.count_adoption_stage("C. Is developing a WTP"),
        #         "Num_Adopters": lambda m: (m.count_adoption_stage("D. Has a WTP")),
        #         "Prop_Aware": lambda m: sum(a.beliefs["awareness"] for a in m.agents) / m.num_agents,
        #     },
        #     agent_reporters={
        #         "Adoption Stage": "adoption_stage",
        #         "Adoption Probability": "prob_adoption",
        #         "Perceived Net Benefit": "perceived_net_benefit",
        #         "Awareness": lambda a: a.beliefs["awareness"],
        #     },
        # )
        agent_reporters = {
            "Adoption Stage": "adoption_stage",
            "Adoption Probability": "prob_adoption",
            "Perceived Net Benefit": "perceived_net_benefit",
            "Awareness": lambda a: a.beliefs["awareness"],
        } if collect_agent_data else None

        self.datacollector = DataCollector(
            model_reporters={
                "Num_Considering": lambda m: m.count_adoption_stage("B. May consider"),
                "Num_Developers": lambda m: m.count_adoption_stage("C. Is developing a WTP"),
                "Num_Adopters": lambda m: m.count_adoption_stage("D. Has a WTP"),
                "Prop_Aware": lambda m: sum(a.beliefs["awareness"] for a in m.agents) / m.num_agents,
            },
            agent_reporters=agent_reporters,
        )

        for agent in self.agents:
            agent.initialise_step() # This is determinstic pass

        self.datacollector.collect(self)

    def step(self): # This defines what happens each tick
        self.apply_active_shocks() # If you have any active shocks, this step applies them

        # Phase 1: snapshot the start of the tick for every agent
        agents = list(self.agents)
        self.random.shuffle(agents) # This keeps a random order of agents at each tick, but everyone gets a snapshot
        for agent in agents:
            agent.store_previous_state()
    
        # Phase 2: compute next state from snapshots only
        for agent in agents:
            agent.step()

        # Phase 3: commit next state for everyone
        for agent in agents:
            agent.advance()

        self.datacollector.collect(self) # Collect data

    def build_network_from_agents(self, agents):
        """
        Build graph from agents using the rule:
        - strong ties (peer edge) if agents are in the same None network
        - weak ties (comeptitor edge) if agents are in the same sector AND region
          otherwise no edge
        """
        G = nx.Graph() # Creates an empty undirected simple graph

        # Add nodes, storing the following attributes needed for edge creation
        for a in agents:
            G.add_node(
                a.unique_id,
                postcode=a.postcode,
                sector=a.sector,
                network=a.network,
                #size=a.size,
            )

        # Group by linkage keys. These are two dictionaries acting as grouping structures
        by_sector_postcode = {}
        by_network = {}

        for a in agents: # Filling the by_sector and by_network lists...
            # Group by combined sector + postcode for competitor links
            key = (a.sector, a.postcode)
            by_sector_postcode.setdefault(key, []).append(a.unique_id)

            # Group by network for peer links
            if a.network is not None: # If the network is not none
                by_network.setdefault(a.network, []).append(a.unique_id) # add the agent info to the network label

        def add_edges_within_group(id_list):
            for i, j in itertools.combinations(id_list, 2): # For each unique pair of nodes in that group, check whether they should be connected.
                p1, s1, n1 = G.nodes[i]["postcode"], G.nodes[i]["sector"], G.nodes[i]["network"]
                p2, s2, n2 = G.nodes[j]["postcode"], G.nodes[j]["sector"], G.nodes[j]["network"]
                
                # Define link types
                competitor_link = (s1 == s2 and p1 == p2) # If in the same sector and postcode, they are competitors
                network_link = (
                    n1 is not None and n2 is not None
                    and n1 == n2
                ) # Network link if they are in the same network
                
                if network_link:
                    edge_type ="peer"
                elif competitor_link:
                    edge_type = "competitor" # Peer dominates if both are true
                else:
                    continue # with no link at all

                # Because the same pair of firms (i & j) can be encountered in both the postcode and network grouping. 
                # I need a rule for what to do if an edge already exists when I encounter the pair the second time....
                if G.has_edge(i, j):
                    # Only ever upgrade to peer
                    if edge_type == "peer":
                        G.edges[i, j]["type"] = "peer"
                else:
                    G.add_edge(i, j, type=edge_type)

        # Add competitor edges within sector + postcode groups
        for ids in by_sector_postcode.values():
            if len(ids) > 1:
                add_edges_within_group(ids)

        #  Add competitor edges within sector + postcode groups
        for ids in by_network.values():
            if len(ids) > 1:
                add_edges_within_group(ids)

        return G


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