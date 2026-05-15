############################################
#     Model - Agents                       #
#     Date: 2026-04-02                     #
#     Author: Jesse Wise                   #
#     Purpose: Testing V12     #
############################################

# This agent file uses a k out of 4 decision making rule
# Think of this file as the decision-maker, given the world what do agents do?
# It should hold beliefs and sensitiviteis, observe neighbours, compute feasibility, probability of adoption and change adoption state.
# Note: I have used Chat GPT to help me learn how to use Mesa, and to write this code. All errors are my own. 
import random  # Required for belief initialisation
import math    # Required for sigmoid/logit
import numpy as np # Required for log function
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
    resource_min: float
    organisationalReadiness_min: float
    knowledge_min: float
    publicTransport_min: float
    competitor_inference_increment: float
    learning_rate: float

    def __init__(self, model, sector=None, postcode=None, network=None):
        super().__init__(model)  # Pass parameters to the parent class. Mesa should automatically generate unique_id and register the agent

        # Static attributes
        sector_cats = [
        "Agriculture, Forestry and Fishing",
        "Mining and Quarrying",
        "Manufacturing",
        "Electricity, Gas, Steam and Air conditioning supply",
        "Water supply",
        "Construction",
        "Wholesale and retail",
        "Transport and storage",
        "Accommodation and food service",
        "Information and communication",
        "Financial and insurance",
        "Real estate",
        "Professional, scientific and technical",
        "Administrative and support services",
        "Public administration and defence",
        "Education",
        "Human health and social work",
        "Arts, entertainment and recreation"]
        sector_weights=[0.0172, 0.0043, 0.0776, 0.0043, 0.0043, 0.056, 0.0862, 0.0431, 0.0388, 0.0647, 0.0603, 0.0086, 0.1034, 0.0345, 0.0043, 0.1422, 0.1336, 0.1166] # Because of rounding errors I have had to change the numbers slightly to sum to 1


        postcode_cats =["East Midlands", "East of England", "London", "North West", "Northern Ireland", "Scotland", "South East", "South West", 
                        "Wales", "West Midlands", "Yorkshire and The Humber"]
        postcode_weights=[0.056, 0.0905, 0.1853, 0.1078, 0.0259, 0.0603, 0.1853, 0.0819, 0.0431, 0.0862, 0.0777] # Same here

        network_cats=[None, "Chamber of Commerce", "FSB","British Assocation for Counselling & Psychotherapy", "Chartered Insurance Institute", "Cleaning & Hygiene Suppliers Assocaiation", "AgeUk", "ASPIRE", "National Care Association", "Road Haulage Association", "Other"] 
        network_weights=[0.71, 0.02, 0.02,  0.02,  0.02,  0.02, 0.02, 0.02, 0.02, 0.02, 0.11] # Approx 25% are in a network. Often more than one network at a time. For now I start with a simple model where firms are only members of 1 network. Again, to make it sum to 1, I've had to change the proportions slightly.

        # size_cats = ["10-19", "20-49","50-99", "100-249"]
        # size_weights=[0.16, 0.28, 0.20, 0.36]
        # self.SIZE_MIDPOINT = {"10-19": 14.5, "20-49": 34.5, "50-99": 74.5, "100-249": 174.5}

        # def draw_size_from_bin(bin_label: str) -> int:
        #     lo_str, hi_str = bin_label.split("-")
        #     lo, hi = int(lo_str), int(hi_str)
        #     # Inclusive on both ends
        #     return self.model.random.randrange(lo, hi + 1)

        # Static attributes (size, sector, postcode, network memebership draw from survey data distributions)
        self.time_in_stage = 0 # This is an agent level counter which will be used to add time lags to adoption
        #self.size_cat = self.model.random.choices(size_cats, weights=size_weights, k=1)[0] # Bin based sampling from your survey distribution
        #self.size = draw_size_from_bin(self.size_cat)
        self.sector = sector if sector is not None else self.model.random.choices(sector_cats, weights=sector_weights, k=1)[0]
        self.postcode = postcode if postcode is not None else self.model.random.choices(postcode_cats, weights=postcode_weights, k=1)[0]
        self.network = network if network is not None else self.model.random.choices(network_cats, weights=network_weights, k=1)[0]

        #### Shock sensititvity dictionary - These determine how sensitive the agent is to exogenous shocks.
        sensitivity_cat=["Not at all influential", "Slightly influential",	"Moderately influential",	"Very influential",	"Extremely influential"] # These are the labels that I used in my survey

        # Numeric mapping for calculations
        sensitivity_to_value = {
            "Not at all influential": 0.0,
            "Slightly influential": 0.25,
            "Moderately influential": 0.5,
            "Very influential": 0.75,
            "Extremely influential": 1.0,
        } # Here I assign values to them. These are not empirically validated

        # These are the distributions of responses to each policy intervention
        subsidy_weights=[0.03, 0.075, 0.203, 0.346, 0.346]
        caseStudy_weights=[0.053, 0.1818, 0.3333, 0.303, 0.1288]
        proofOfROI_weights=[0.015, 0.053, 0.173, 0.444, 0.316]
        accreditationAward_weights=[0.143, 0.188,0.263,0.301,0.105]
        policyChampion_weights=[0.145, 0.298, 0.366, 0.153, 0.038]
        wordOfMouth_weights=[0.145, 0.298, 0.366, 0.153, 0.038] # Need to come back and edit this

        def tilt_weights(weights, strength, direction="up"):
            idx = np.arange(len(weights))
            if direction == "up":
                multipliers = np.exp(strength * idx)
            elif direction == "down":
                multipliers = np.exp(-strength * idx)
            else:
                raise ValueError("direction must be 'up' or 'down'")

            tilted = np.array(weights, dtype=float) * multipliers
            tilted /= tilted.sum()
            return tilted.tolist()
        
        # Draw category then store both category and numeric sensitivity
        def draw_sensitivity(weights):
            cat = self.model.random.choices(sensitivity_cat, weights=weights, k=1)[0]
            return {"category": cat, "value": sensitivity_to_value[cat]}

        self.shock_sensitivity = {
            "subsidy": draw_sensitivity(subsidy_weights),
            "wordOfMouth": draw_sensitivity(wordOfMouth_weights),
            "caseStudy": draw_sensitivity(caseStudy_weights),
            "proofOfROI": draw_sensitivity(proofOfROI_weights),
            "accreditationAward": draw_sensitivity(accreditationAward_weights),
            "policyChampion": draw_sensitivity(policyChampion_weights),
        }

        levels_5 = [0, 0.25, 0.5, 0.75, 1]

        ### Beliefs dictionary - subject to social learning (initial values drawn from plausible uniform distribution (tilted during calibration), but end values should match my survey data)
        self.beliefs = {
            "motivations": self.model.random.choices(levels_5, 
                                                     weights=tilt_weights([0.2351, 0.2498, 0.3420, 0.0977, 0.0754], self.model.init_positive_shift, direction="up"))[0], # No logitudinal data so I am estimating it based on the other variables

            "perceivedBarriers": self.model.random.choices(levels_5, 
                                                           weights=tilt_weights([0.2351, 0.2498, 0.3420, 0.0977, 0.0754],self.model.init_barrier_shift, direction="down"))[0], # No logitudinal data so I am estimating it based on the other variables

            "knowledge": self.model.random.choices(levels_5, 
                                                   weights=tilt_weights([0.2351, 0.2498, 0.3420, 0.0977, 0.0754], self.model.init_positive_shift, direction="up"))[0], # No logitudinal data so I am estimating it based on the other variables

            "organisationalReadiness": self.model.random.choices(levels_5, 
                                                                 weights=tilt_weights([0.288, 0.178, 0.278, 0.118, 0.138], self.model.init_positive_shift, direction="up"))[0], # Coleman (2000) data used as a proxy

            "publicTransport": self.model.random.choices([0, 0.2, 0.4, 0.6, 0.8, 1],
                                                         weights=tilt_weights([0.05, 0.23, 0.38, 0.26, 0.07, 0.01], self.model.init_positive_shift, direction="up"))[0], # Coleman (2000) data used as a proxy

            "resources": self.model.random.choices(levels_5, 
                                                   weights=tilt_weights([0.295, 0.265, 0.31, 0.075, 0.055], self.model.init_positive_shift, direction="up"))[0], # Coleman (2000) data used as a proxy
            
            "awareness": self.model.random.choices([0, 1], weights=[0.64, 0.36])[0] # Coleman 2000
            }

        # Store the initial beliefs to pull toward realism baseline
        self.beliefs_initial = self.beliefs.copy()

        # Metadata to distinguish belief types
        self.belief_types = {
            "motivations": "subjective",
            "perceivedBarriers": "subjective",
            "knowledge": "subjective",
            "organisationalReadiness": "subjective",
            "resources": "objective",
            "publicTransport": "objective",
            "awareness": "subjective"
        }

        # Dynamic attributes (not beliefs). These are just place holders that get overwritten by agent.initialise_step() in model.py
        self.adoption_stage = "A. No intention"
        self.prev_adoption_stage = self.adoption_stage
        self.prob_adoption = 0
        self.perceived_net_benefit = None
        self.perceivedPeerAdoption = 0.0
        self.numberOfConstraintsMet=0

        # Snapshots from the previous tick
        self.prev_beliefs = self.beliefs.copy()
        self.prev_numberOfConstraintsMet = self.numberOfConstraintsMet
        self.prev_prob_adoption = self.prob_adoption
        self.prev_perceived_net_benefit = self.perceived_net_benefit
        self.prev_time_in_stage = self.time_in_stage

        # Buffers for the next tick
        self.next_beliefs = self.beliefs.copy()
        self.next_numberOfConstraintsMet = self.numberOfConstraintsMet
        self.next_prob_adoption = self.prob_adoption
        self.next_perceived_net_benefit = self.perceived_net_benefit
        self.next_adoption_stage = self.adoption_stage
        self.next_time_in_stage = self.time_in_stage
        self.next_perceivedPeerAdoption = self.perceivedPeerAdoption

        # Store static thresholds from the model
        self.r_min_eff = self.model.effective_resource_min(self) # calculate their resource min based on whether subsidies are active or not
        self.or_min_eff = self.model.effective_organisationalReadiness_min(self) # calculate their OR based on whether policy champions are active or not
        self.r_min = model.resource_min
        self.k_min = model.knowledge_min
        self.or_min = model.organisationalReadiness_min
        self.pt_min= model.publicTransport_min
        self.obj_net_benefit_min = model.obj_net_benefit_min
        self.obj_net_benefit_max = model.obj_net_benefit_max

        self.perceived_net_benefit = None # This is also to help me debug

    def initialise_step(self): # This is to derive the variables, just based on firms internal states rather than social learning
        self.update_perceived_feasibility()     # Firms update their perceived feasibility of adopting a WTP.
        self.update_prob_adoption()             # Firms update their probability of adoption.
        self.update_adoption_status_FIRSTTICKONLY()           # Their adoption status is updated.
        self.advance()
        self.store_previous_state()             # So I save the OG values but have calculate adoption status based on values

    def step(self):
        self.next_beliefs = self.prev_beliefs.copy() # Start from the previous state as a baseline
        self.next_time_in_stage = self.prev_time_in_stage + 1  # Time advances by one tick unless reset later by a stage change

        ### This is used for implementing exogenous shocks
        self.r_min_eff = self.model.effective_resource_min(self) # calculate their resource min based on whether subsidies are active or not
        self.or_min_eff = self.model.effective_organisationalReadiness_min(self) # calculate their OR based on whether policy champions are active or not
        self.learning_rate_eff = self.model.effective_learning_rate(self) # caluculate learning rate based on whether accridatiation is on or not
        self.competitor_inference_increment_eff = self.model.effective_competitor_inference_increment(self) # caluculate competitor inference increment based on whether accridatiation is on or not
        
        ###  Step
        self.observe_network()                  # Firms observe their network (strong and weak ties) to learn from them.
        self.update_knowledge_fully()            # Based on observations of strong ties (peers), firms have full information.
        self.update_knowledge_partially()         # Based on observations of weak ties (competitors), firms infer their knowledge based on partial information
        self.update_perceived_peer_adoption() # Firms update their perception of peer adoption based on their observations. This is just a latent variable to observe
        self.update_perceived_feasibility()     # Firms update their perceived feasibility of adopting a WTP.
        self.update_prob_adoption()             # Firms update their probability of adoption.
        self.update_adoption_status()           # Their adoption status is updated.

    def store_previous_state(self):
        """Snapshot the agent's state at the start of the tick."""
        self.prev_beliefs = self.beliefs.copy()
        self.prev_adoption_stage = self.adoption_stage
        self.prev_numberOfConstraintsMet = self.numberOfConstraintsMet
        self.prev_prob_adoption = self.prob_adoption
        self.prev_perceived_net_benefit = self.perceived_net_benefit
        self.prev_time_in_stage = self.time_in_stage


    def advance(self):
        """Commit the next state after all agents have computed."""
        self.beliefs = self.next_beliefs.copy()
        self.numberOfConstraintsMet = self.next_numberOfConstraintsMet
        self.prob_adoption = self.next_prob_adoption
        self.perceived_net_benefit = self.next_perceived_net_benefit
        self.adoption_stage = self.next_adoption_stage
        self.time_in_stage = self.next_time_in_stage
        self.perceivedPeerAdoption = self.next_perceivedPeerAdoption

    def observe_network(self):
        """ Observe the beliefs of network members (peers and competitors) and store the information. This makes sure all agents have information from the same timepoint so we don't get order effects"""
        self.peer_ids = [] # initialise empty lists to store peer and competitor IDs
        self.competitor_ids = []
        self.peer_beliefs_raw = {b: [] for b in self.beliefs}  # Store each peer's beliefs
        self.competitor_adoptions = [] # a dictionary of transition statuses

        for agent in self.model.grid.get_neighbors(self.pos, include_center=False): # Loop through all agents connected to self (this agent). For each neighbour check the edge's "type" in Graph G
            node_id = agent.pos
            link_type = self.model.G[self.pos][node_id]["type"]

            if link_type == "peer":                 # If the link type is "peer"    
                self.peer_ids.append(node_id)       # Add to peer_ids   
                for b in self.beliefs:
                    self.peer_beliefs_raw[b].append(agent.prev_beliefs[b]) # Store the peer's beliefs in the raw list
            
            elif link_type == "competitor":         # If they are a competitior...
                self.competitor_ids.append(node_id) # Store their node id
                self.competitor_adoptions.append(agent.prev_adoption_stage) # as well as their previous adoption status

    def update_knowledge_fully(self):
        for b in self.beliefs:  # Loop through all belief dimensions
            peer_values = self.peer_beliefs_raw.get(b, []) # retrieve the list of peer beliefs for this dimension
            if not peer_values: #If the firm has no peers, skip
                continue 

            # Awareness is a special case as it is binary (0 OR 1)
            if b == "awareness":
                if any(v == 1 for v in peer_values): # If any peer of theirs is aware, then they are also aware 
                    self.next_beliefs["awareness"] = 1
                continue  # Skip the continuous update for awareness

            # Continuous beliefs (0 -> 1)
            social_mean = sum(peer_values) / len(peer_values) # take an average of peer values
            personal = self.prev_beliefs[b] # agent's current belief

            self.next_beliefs[b] = np.clip(( personal # New belief = old belief + 
               + self.learning_rate_eff * (social_mean - personal) # (learning rate × peer signal gap) nudges beliefs towards the social average 
               ), 0.0, 1.0)

    def update_knowledge_partially(self): 
        """Per weak tie, while a weak tie has (or is developing) a plan, then they infer this as a signal that WTPs are a net benefit. Motivations increase and perceived barriers decrease.""" 
        adopted = {"C. Is developing a WTP","D. Has a WTP"}

        for competitor_stage_prev in self.competitor_adoptions: # Stacks competitor effects, but ONLY for the previous timestep i.e. a while loop, not if they have ever adopted in the past
            if competitor_stage_prev in adopted:
                self.next_beliefs["motivations"] = np.clip(
                    self.next_beliefs["motivations"] + self.competitor_inference_increment_eff, 0.0, 1.0)
                self.next_beliefs["perceivedBarriers"] = np.clip(
                    self.next_beliefs["perceivedBarriers"] - self.competitor_inference_increment_eff, 0.0, 1.0)
                self.next_beliefs["awareness"] = 1

                if self.prev_adoption_stage in {"B. May consider", "C. Is developing a WTP", "D. Has a WTP"}: #If a firm has at least reached the “may consider” stage, then competitor adoption motivates them to increase feasibility.
                    self.next_beliefs["knowledge"] = np.clip(
                        self.next_beliefs["knowledge"] + self.competitor_inference_increment_eff, 0.0, 1.0
                    )
                    self.next_beliefs["organisationalReadiness"] = np.clip(
                        self.next_beliefs["organisationalReadiness"] + self.competitor_inference_increment_eff, 0.0, 1.0
                    )
                    self.next_beliefs["resources"] = np.clip(
                        self.next_beliefs["resources"] + self.competitor_inference_increment_eff, 0.0, 1.0
                    )

    
    def update_perceived_peer_adoption(self):
        # Combine peer and competitor ids
        neighbour_ids = list(set(self.peer_ids + self.competitor_ids))
        total = len(neighbour_ids)

        if total == 0: # If they have no neighbours then perceived peer adoption is 0
            self.next_perceivedPeerAdoption = 0.0
            return

        adopted_stages = {"D. Has a WTP"}
        num_with_plan = 0

        for nid in neighbour_ids: # Counter for peer adoption
            contents = self.model.grid.get_cell_list_contents([nid])
            if not contents:
                continue  # No agent on that node
            neighbour = contents[0]
            if neighbour.prev_adoption_stage in adopted_stages:
                num_with_plan += 1

        self.next_perceivedPeerAdoption = num_with_plan / total # At the moment this is just a latent variable to observe.    

    def update_perceived_feasibility(self):
        "This calculates how many of the constraints are met and returns them"
        self.numberOfConstraintsMet = sum([
            self.next_beliefs["resources"] >= self.r_min_eff,
            self.next_beliefs["knowledge"] >= self.k_min,
            self.next_beliefs["organisationalReadiness"] >= self.or_min_eff,
            self.next_beliefs["publicTransport"] >= self.pt_min,
        ])
        return self.numberOfConstraintsMet
        

    def update_prob_adoption(self):
        # size_mid = self.SIZE_MIDPOINT.get(self.size_cat) # Get the midpoint of the size bin
        # if size_mid is None: # For debuggin
        #     raise ValueError(f"Unknown size category: {self.size}")
        
        # Calculate the perceived net benefit of adopting a WTP
        self.next_perceived_net_benefit = self.model.obj_net_benefit_min + (
            (self.model.obj_net_benefit_max - self.model.obj_net_benefit_min) * (
                (self.next_beliefs["motivations"] - (self.next_beliefs["perceivedBarriers"])))) # estimated net benefit is equal to the minimum plausible net benefit plus a range of plausible net benefit values that depends on the perception of costs and benefits of adoption (which are scaled between 0 and 1). 
                                                                 # If the WTP is perceived as feasible, then:
        self.next_prob_adoption = 1 / (1 + math.exp(-0.03*(self.next_perceived_net_benefit-126)))          # The logit (sigmoidal) function converts the perceived net benefit into a probability of adoption for a range of NB from 126 to 250. Which is what we want when size does not influence
                                                 # If a WTP is not perceived as feasible, then the probability of adoption is 0

    def probability_to_stage(self, p):
        """Map adoption probability to a candidate stage."""
        if p < 0.14:
            return "A. No intention"
        elif p < 0.58:
            if self.numberOfConstraintsMet >= 2:
                return "B. May consider"
            else:
                return "A. No intention"
        elif p < 0.79:
            if self.numberOfConstraintsMet >= 3:
                return "C. Is developing a WTP"
            elif self.numberOfConstraintsMet >= 2:
                return "B. May consider"
            else:
                return "A. No intention"
        else:
            if self.numberOfConstraintsMet >= 4:
                return "D. Has a WTP"
            elif self.numberOfConstraintsMet >= 3:
                return "C. Is developing a WTP"
            elif self.numberOfConstraintsMet >= 2:
                return "B. May consider"
            else:
                return "A. No intention"

    def update_adoption_status(self):
        """Update adoption stage using candidate stage, progression rules, and time locks."""
        old_stage = self.prev_adoption_stage
        candidate_stage = self.probability_to_stage(self.next_prob_adoption)

        STAGES = [
            "A. No intention",
            "B. May consider",
            "C. Is developing a WTP",
            "D. Has a WTP",]

        STAGE_TO_INDEX = {stage: i for i, stage in enumerate(STAGES)}

        old_idx = STAGE_TO_INDEX[old_stage]
        cand_idx = STAGE_TO_INDEX[candidate_stage]

        # Default: move toward candidate, but by at most one stage per tick
        if cand_idx > old_idx + 1:
            next_idx = old_idx + 1
        elif cand_idx < old_idx:
            next_idx = cand_idx
        else:
            next_idx = cand_idx

        next_stage = STAGES[next_idx]

        # Must spend at least 1 tick in B before moving to C
        if old_stage == "B. May consider" and candidate_stage in {"C. Is developing a WTP", "D. Has a WTP"}:
            if self.time_in_stage < 1:
                next_stage = "B. May consider"

        # Must spend at least 2 ticks in C before moving to D
        if old_stage == "C. Is developing a WTP" and candidate_stage == "D. Has a WTP":
            if self.time_in_stage < 2:
                next_stage = "C. Is developing a WTP"

        # Must spend at least 3 ticks in D before dropping out
        if old_stage == "D. Has a WTP" and candidate_stage != "D. Has a WTP":
            if self.time_in_stage < 3:
                next_stage = "D. Has a WTP"

        self.next_adoption_stage = next_stage

        if next_stage != old_stage:
            self.next_time_in_stage = 0
        else:
            self.next_time_in_stage = self.time_in_stage + 1

    def update_adoption_status_FIRSTTICKONLY(self):
        """On the first tick we're just updating adoption status based on their beliefs etc, so there should be no time lags."""
        self.next_adoption_stage = self.probability_to_stage(self.next_prob_adoption)
        self.next_time_in_stage = 0
