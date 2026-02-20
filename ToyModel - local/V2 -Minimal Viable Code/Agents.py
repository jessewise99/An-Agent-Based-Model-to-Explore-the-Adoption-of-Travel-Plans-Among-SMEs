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
            "benefits": self.model.random.uniform(0.1, 0.6), #The seed passed to the model controls all randomness.
            "costs": self.model.random.uniform(0.1, 0.6),
            "time": self.model.random.uniform(0.1, 0.6),
            "money": self.model.random.uniform(0.1, 0.6),
            "knowledge": self.model.random.uniform(0.1, 0.6),
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



    def step(self):
        self.observe_peers()               # Firms observe their peers to learn from them.
        self.update_knowledge()           # Based on observations, firms update their knowledge and beliefs.
        self.update_perceived_feasibility()  # Firms update their perceived feasibility of adopting a WTP.
        self.update_prob_adoption()       # Firms update their probability of adoption.
        self.update_adoption_status()     # Their adoption status is updated.

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
                self.model.realism_pull_subjective
                if self.belief_types[b] == "subjective"
                else self.model.realism_pull_constraints
            )

            self.beliefs[b] += self.model.learning_rate * (social - personal) + realism_pull * (baseline - personal)

    def update_perceived_feasibility(self):
        t_min = self.model.time_min
        m_min = self.model.money_min
        k_min = self.model.knowledge_min

        self.feasible = (
            self.beliefs["time"] > t_min and
            self.beliefs["money"] > m_min and
            self.beliefs["knowledge"] > k_min
        )

    def update_prob_adoption(self):
        if self.feasible:
            diff = self.beliefs["benefits"] - self.beliefs["costs"]
            self.prob_adoption = 1 / (1 + math.exp(-diff))  # Sigmoid/logit
        else:
            self.prob_adoption = 0

    def update_adoption_status(self):
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
