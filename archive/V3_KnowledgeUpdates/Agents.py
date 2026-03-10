############################################
#     Toy model prototype - Agents         #
#     Date: 2025-07-01                     #
#     Author: Jesse Wise                   #
#     Purpose: To learn how Mesa works     #
#     and to test agents in a simple model #
############################################

# Note: I have used Chat GPT to help me learn how to use Mesa, and to write this code. All errors are my own. 
import random  # Required for belief initialisation
import math    # Required for sigmoid/logit
from mesa import Agent  # Using Agent which is more flexible than FixedAgent. Correct base class for custom agents


class FirmAgent(Agent):
    """An agent representing a firm in the model.

    Attributes:
        unique_id (int): Unique identifier for the agent.
        model (AdoptionModel): Reference to the model this agent belongs to.
        sector, postcode, network, size: Static attributes.
        beliefs (dict): Subjective and objective beliefs.
        belief_types (dict): Used to set learning realism pull per belief type.
    """

    def __init__(self, model, sector, postcode, network_membership, size):
        super().__init__(model)  # Pass parameters to the parent class. Mesa should automatically generate unique_id and register the agent

        # Static attributes
        self.sector = sector
        self.postcode = postcode
        self.network = network_membership
        self.size = size

        # Beliefs dictionary - subject to social learning (initial values drawn from plausible uniform distribution)
        self.beliefs = {
            "benefits": self.model.random.uniform(0.4, 0.7), #The seed passed to the model controls all randomness.
            "costs": self.model.random.uniform(0.1, 0.3),
            "time": self.model.random.uniform(0.1, 0.9),
            "money": self.model.random.uniform(0.1, 0.9),
            "knowledge": self.model.random.uniform(0.1, 0.9),
        }

        # Store the initial beliefs to pull toward realism baseline
        self.beliefs_initial = self.beliefs.copy()

        # Metadata to distinguish belief types
        self.belief_types = {
            "benefits": "subjective",
            "costs": "subjective",
            "time": "objective",
            "money": "objective",
            "knowledge": "subjective"
        }

        # Dynamic attributes (not beliefs)
        self.adoption_stage = "A. No intention"
        self.feasible = False
        self.prob_adoption = 0

        # Store static thresholds from the model
        self.t_min = model.time_min
        self.m_min = model.money_min
        self.k_min = model.knowledge_min


    def step(self):
        #self.observe_peers()                   # Firms observe their peers to learn from them.
        #self.update_knowledge()                # Based on observations, firms update their knowledge and beliefs.
        self.observe_network()                  # Firms observe their network (strong and weak ties) to learn from them.
        self.update_knowledge_partially()         # Based on observations of weak ties (competitors), firms infer their knowledge based on partial information
        self.update_knowledge_fully()            # Based on observations of strong ties (peers), firms have full information.
        self.update_perceived_feasibility()     # Firms update their perceived feasibility of adopting a WTP.
        self.update_prob_adoption()             # Firms update their probability of adoption.
        self.update_adoption_status()           # Their adoption status is updated.

    def observe_peers(self):
        """Observe the beliefs of peers and store simple averages for each belief."""
        peers = []  # List of peer agents

        for agent in self.model.grid.get_neighbors(self.pos, include_center=False): # This converts neighbours (agents) back to their positions (node IDs).
            node_id = agent.pos  # get the node ID
            if self.model.G[self.pos][node_id]["type"] == "peer":
                peers.append(node_id)


        for node_id in self.model.grid.get_neighbors(self.pos, include_center=False): #Loop through neighbor node IDs
            node_id = agent.pos  # get the node ID
            if self.model.G[self.pos][node_id]["type"] == "peer":
                peers.append(node_id)

        if not peers:
            self.social_average = {b: self.beliefs[b] for b in self.beliefs}  # No peers? Fall back to own beliefs
            return

        # Initialise accumulators
        sums = {b: 0 for b in self.beliefs}
        for peer_id in peers:
            peer = self.model.grid.get_cell_list_contents([peer_id])[0]  # Get agent object
            for b in self.beliefs:
                sums[b] += peer.beliefs[b]

        self.social_average = {b: sums[b] / len(peers) for b in self.beliefs}




    def update_knowledge(self):
        # [Correction: Assumes you have a dict called social_average available in context]
        social_average = self.social_average

        for b in self.beliefs:
            if b not in social_average:
                continue  # skip if no social info for this belief

            personal = self.beliefs[b]
            baseline = self.beliefs_initial[b]
            social = social_average[b]

            realism_pull = (
                self.model.realism_pull_sociallyInfluencedVars
                if self.belief_types[b] == "subjective"
                else self.model.realism_pull_constraints
            )

            self.beliefs[b] += self.model.learning_rate * (social - personal) + realism_pull * (baseline - personal)


    def observe_network(self):
        """ Observe the beliefs of network members (peers and competitors) and store the information"""
        self.peer_ids = [] # initialise empty lists to store peer and competitor IDs
        self.competitor_ids = []
        self.peer_beliefs_raw = {b: [] for b in self.beliefs}  # Store each peer's beliefs

        for agent in self.model.grid.get_neighbors(self.pos, include_center=False): # Loop through all agents connected to self (this agent). For each neighbour check the edge's "type" in Graph G
            node_id = agent.pos
            link_type = self.model.G[self.pos][node_id]["type"]

            if link_type == "peer":                 # If the link type is "peer"    
                self.peer_ids.append(node_id)       # Add to peer_ids   
                peer = self.model.grid.get_cell_list_contents([node_id])[0] # Get the peer agent object
                for b in self.beliefs:
                    self.peer_beliefs_raw[b].append(peer.beliefs[b]) # Store the peer's beliefs in the raw list
                
            elif link_type == "competitor":         # If the link type is "competitor", add to competitor_ids
                self.competitor_ids.append(node_id) # Store the competitor's node ID

        # Store competitor adoption status
        self.competitor_adoptions = []              
        for comp_id in self.competitor_ids:
            comp = self.model.grid.get_cell_list_contents([comp_id])[0]
            self.competitor_adoptions.append(comp.adoption_stage)


    def update_knowledge_partially(self):
        for adoption_stage in self.competitor_adoptions:                    # Loop through the adoption stages of competitors
            if adoption_stage in ["A. No intention", "B. May consider"]:    # If competitors have not adopted
                self.beliefs["benefits"] -= self.model.competitor_inference_increment # decrease perceived benefits by the inference increment
            if adoption_stage in ["D. Has a low efficacy WTP", "E. Has an effective WTP"]:  # If competitors have adopted
                self.beliefs["benefits"] += self.model.competitor_inference_increment # increase perceived benefits by the inference increment

    def update_knowledge_fully(self):
        for b in self.beliefs:  # Loop through all belief dimensions like time, money, knowledge, benefits, costs
            peer_values = self.peer_beliefs_raw.get(b, []) # retrieve the list of peer beliefs for this dimension
            if not peer_values: #If the firm has no peers for this belief, skip
                continue  # No peers for this belief

            social_mean = sum(peer_values) / len(peer_values) # takes an average of peer values
            personal = self.beliefs[b] # agent's current belief
            baseline = self.beliefs_initial[b] # agent's initial belief

            realism_pull = (
                self.model.realism_pull_sociallyInfluencedVars if self.belief_types[b] == "subjective"
                else self.model.realism_pull_constraints
            ) # Use the correct realism pull based on belief type i.e., it is stronger for constraints (objective) than for subjective beliefs

            self.beliefs[b] += (                                    # new belief = old belief
                self.model.learning_rate * (social_mean - personal) #  + (learning rate × peer signal gap) nudges beliefs towards the social average
                + realism_pull * (baseline - personal)              # + (realism pull × gap from original belief)  pulls beliefs back toward where they started
            )


    def update_perceived_feasibility(self):
        self.feasible = (
            self.beliefs["time"] > self.t_min and
            self.beliefs["money"] > self.m_min and
            self.beliefs["knowledge"] > self.k_min
        ) # Feasible = True if all constraints (time, money, knowledge) are above the minimum thresholds, otherwise it is false

    def update_prob_adoption(self):
        if self.feasible:                                                            # If the WTP is perceived as feasible, then:
            perceived_net_benefit = self.beliefs["benefits"] - self.beliefs["costs"] # Calculate the perceived net benefit of adopting a WTP
            self.prob_adoption = 1 / (1 + math.exp(-perceived_net_benefit))          # The logit (sigmoidal) function converts the perceived net benefit into a probability of adoption
        else:
            self.prob_adoption = 0                                                   # If a WTP is not perceived as feasible, then the probability of adoption is 0

    def update_adoption_status(self):                                                # Update the adoption stage based on the probability of adoption
        p = self.prob_adoption
        if p < 0.2:
            self.adoption_stage = "A. No intention"
        elif p < 0.4:
            self.adoption_stage = "B. May consider"
        elif p < 0.6:
            self.adoption_stage = "C. Is developing a WTP"
        elif p < 0.8:
            self.adoption_stage = "D. Has a low efficacy WTP"
        else:
            self.adoption_stage = "E. Has an effective WTP"
