############################################
#     Toy model prototype - Agents         #
#     Date: 2026-02-04                     #
#     Author: Jesse Wise                   #
#     Purpose: Implementing Pseudocode V2  #
############################################

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

    def __init__(self, model, sector=None, postcode=None, network=None, size=None):
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
        sector_weights=[0.0172, 0.0043, 0.0776, 0.0043, 0.0043, 0.056, 0.0862, 0.0431, 0.0388, 0.0647, 0.0603, 0.0086, 0.1034, 0.0345, 0.0043, 0.1422, 0.1336, 0.1166] # This used to sum to .9998 so i have had to change the numbers slightly


        postcode_cats =["East Midlands", "East of England", "London", "North West", "Northern Ireland", "Scotland", "South East", "South West", 
                        "Wales", "West Midlands", "Yorkshire and The Humber"]
        postcode_weights=[0.056, 0.0905, 0.1853, 0.1078, 0.0259, 0.0603, 0.1853, 0.0819, 0.0431, 0.0862, 0.0777] # same here


        # network_cats=[None, "Chamber of Commerce", "FSB","BACP", "CII", "CHSA", "UK Finance", "ABC", "ARLA", "ATOL", "Age Care UK", "Aging Better UK",
        #               "Alzheimers Society", "American Bankers Association", "Aspire", "Association of Cycle Retailers", "Association of Professional Builders",
        #               "BAFTA", "BEN", "Bank of England Regulatory and Industry Forums", "Boots", "British Standards Institution", "Business Network International",
        #               "Chartered Banker Institute", "Clinical Board Meeting","Constructionline", "Deeside Decarbonisation Forum", "Entrepreneurs Circle", 
        #               "Essex Care Association", "European Arenas Association", "Federation of Master Builders","Gambica","General Teaching Council for Scotland",
        #               "HEP Hounslow Education Partnership", "IMAD", "IT Association", "Imperial Society of Teachers of Dancing", "Insurance Information Institute", 
        #               "International Air Transport Association", "International Dance Teachers Association", "Local Council", "Local Health Network", "Multiple / Miscellaneous",
        #               "National Association of Estate Agents",  "National Care Association", "National House Building Council", "Northern Ireland Hotels Federation",
        #               "Nurture UK", "Ofqual", "Paradigm", "Passivhaus Institut", "Passivhaus Trust", "Payment Systems Regulator", "PiXL", "Public Relations and Communications Association",
        #               "RNIC", "Recruitment & Employment Confederation", "Reigate Business Guild", "Risk Management Association","Road Haulage Association", 
        #               "Royal College of Anaesthetists", "Russo-British Chamber of Commerce", "Society of Authors","Southern Farmers Network", 
        #               "The Chartered Institute of Personnel and Development", "The Tile Association","Trade Unions & Collaboration of Schools", "UK Arts Network", 
        #               "UK Contact Centre Forum", "UK Proptech", "UK Weighing Federation", "Visit Belfast", "Wales Area Entertainment Complexes", "Welsh Arts Council",
        #               "West London Business Partnership"] # Commenting this out while i fix the problem of joint network memberships 
        network_cats=[None, "Chamber of Commerce", "FSB","BACP", "CII", "CHSA"] 
        #network_weights=[0.7586, 0.1202, 0.0689,  0.0345,  0.0345,  0.0345, 0.0345, 0.01724]
        network_weights=[0.7074, 0.1202, 0.0689,  0.0345,  0.0345,  0.0345]
        # Approx 25% are in a network. Often more than one network at a time. network_cats lists the cleaned names. This will be something I need to change

        size_cats = ["10-19", "20-49","50-99", "100-249"]
        size_weights=[0.16, 0.28, 0.20, 0.36]
        self.SIZE_MIDPOINT = {"10-19": 14.5, "20-49": 34.5, "50-99": 74.5, "100-249": 174.5}

        def draw_size_from_bin(bin_label: str) -> int:
            lo_str, hi_str = bin_label.split("-")
            lo, hi = int(lo_str), int(hi_str)
            # Inclusive on both ends
            return self.model.random.randrange(lo, hi + 1)

        # Static attributes (size, sector, postcode, network memebership draw from survey data distributions)
        self.time_in_stage = 0 # This is an agent level counter which will be used to add time lags to adoption
        self.size_cat = self.model.random.choices(size_cats, weights=size_weights, k=1)[0] # Bin based sampling from your survey distribution
        self.size = draw_size_from_bin(self.size_cat)
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
        
        # Draw category then store both category and numeric sensitivity
        def draw_sensitivity(weights):
            cat = self.model.random.choices(sensitivity_cat, weights=weights, k=1)[0]
            return {"category": cat, "value": sensitivity_to_value[cat]}

        self.shock_sensitivity = {
            "subsidy": draw_sensitivity(subsidy_weights),
            "wordOfMouth": draw_sensitivity(caseStudy_weights),
            "caseStudy": draw_sensitivity(caseStudy_weights),
            "proofOfROI": draw_sensitivity(proofOfROI_weights),
            "accreditationAward": draw_sensitivity(accreditationAward_weights),
            "policyChampion": draw_sensitivity(policyChampion_weights),
        }

        ### Beliefs dictionary - subject to social learning (initial values drawn from plausible uniform distribution, but end values should match my survey data)
        self.beliefs = {
            "motivations": self.model.random.uniform(0.1, 0.9), #The seed passed to the model controls all randomness.
            "perceivedBarriers": self.model.random.uniform(0.1, 0.9),
            "organisationalReadiness": self.model.random.uniform(0.1, 0.9),
            "publicTransport": self.model.random.uniform(0.1, 0.9),
            "resources": self.model.random.uniform(0.1, 0.9),
            "knowledge": self.model.random.uniform(0.1, 0.9),
            "awareness": self.model.random.choices([0, 1], weights=[0.64, 0.36])[0] # <- this distribution comes from Coleman 2000 [0.21,0.79] would give me a distribution of random 0s and 1s which match my survey
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

        # Dynamic attributes (not beliefs)
        self.adoption_stage = "A. No intention"
        self.prev_adoption_stage = self.adoption_stage # This will store the adoption stage to be used in observe_network for competitors
        self.feasible = False
        self.prob_adoption = 0

        # Store static thresholds from the model
        self.r_min = model.resource_min
        self.k_min = model.knowledge_min
        self.or_min = model.organisationalReadiness_min
        self.pt_min= model.publicTransport_min
        self.obj_net_benefit_min = model.obj_net_benefit_min
        self.obj_net_benefit_max = model.obj_net_benefit_max


    def step(self):
        self.prev_adoption_stage = self.adoption_stage # Store this adoption stage to be used in observe_network
        self.time_in_stage += 1                 # Increase the time step
        self.r_min_eff = self.model.effective_resource_min(self) # calculate their resource min based on whether subsidies are active or not
        self.or_min_eff = self.model.effective_organisationalReadiness_min(self) # calculate their OR based on whether policy champions are active or not
        self.learning_rate_eff = self.model.effective_learning_rate(self) # caluculate learning rate based on whether accridatiation is on or not
        self.competitor_inference_increment_eff = self.model.effective_competitor_inference_increment(self) # caluculate competitor inference increment based on whether accridatiation is on or not
        self.observe_network()                  # Firms observe their network (strong and weak ties) to learn from them.
        self.update_knowledge_fully()            # Based on observations of strong ties (peers), firms have full information.
        self.update_knowledge_partially()         # Based on observations of weak ties (competitors), firms infer their knowledge based on partial information
        self.update_perceived_peer_adoption() # Firms update their perception of peer adoption based on their observations. This is just a latent variable to observe
        self.update_perceived_feasibility()     # Firms update their perceived feasibility of adopting a WTP.
        self.update_prob_adoption()             # Firms update their probability of adoption.
        self.update_adoption_status()           # Their adoption status is updated.


    def observe_network(self):
        """ Observe the beliefs of network members (peers and competitors) and store the information"""
        self.peer_ids = [] # initialise empty lists to store peer and competitor IDs
        self.competitor_ids = []
        self.peer_beliefs_raw = {b: [] for b in self.beliefs}  # Store each peer's beliefs
        self.competitor_adoptions = [] # a dictionary of transition statuses

        for agent in self.model.grid.get_neighbors(self.pos, include_center=False): # Loop through all agents connected to self (this agent). For each neighbour check the edge's "type" in Graph G
            node_id = agent.pos
            link_type = self.model.G[self.pos][node_id]["type"]

            if link_type == "peer":                 # If the link type is "peer"    
                self.peer_ids.append(node_id)       # Add to peer_ids   
                peer = self.model.grid.get_cell_list_contents([node_id])[0] # Get the peer agent object
                for b in self.beliefs:
                    self.peer_beliefs_raw[b].append(peer.beliefs[b]) # Store the peer's beliefs in the raw list
            
            elif link_type == "competitor":         # If they are a competitior...
                self.competitor_ids.append(node_id) # Store their node id
                self.competitor_adoptions.append((agent.prev_adoption_stage, agent.adoption_stage)) # as well as their adoption status

    def update_perceived_peer_adoption(self):
        # Combine peer and competitor ids
        neighbour_ids = list(set(self.peer_ids + self.competitor_ids))

        total = len(neighbour_ids)
        if total == 0:
            self.perceivedPeerAdoption = 0.0
            return

        adopted_stages = {"D. Has a low efficacy WTP", "E. Has an effective WTP"}

        num_with_plan = 0
        for nid in neighbour_ids:
            contents = self.model.grid.get_cell_list_contents([nid])
            if not contents:
                continue  # No agent on that node
            neighbour = contents[0]
            if neighbour.adoption_stage in adopted_stages:
                num_with_plan += 1

        self.perceivedPeerAdoption = num_with_plan / total

    def update_knowledge_partially(self):
        non_adopted = {"A. No intention", "B. May consider"}
        adopted = {"C. Is developing a WTP","D. Has a low efficacy WTP", "E. Has an effective WTP"}

        for prev_stage, curr_stage in self.competitor_adoptions:
            if prev_stage in non_adopted and curr_stage in adopted:
                self.beliefs["motivations"] += self.competitor_inference_increment_eff
                self.beliefs["awareness"]=1

        for prev_stage, curr_stage in self.competitor_adoptions:
            if prev_stage in adopted and curr_stage in non_adopted:
                self.beliefs["motivations"] -= self.competitor_inference_increment_eff

    def update_knowledge_fully(self):
        for b in self.beliefs:  # Loop through all belief dimensions
            peer_values = self.peer_beliefs_raw.get(b, []) # retrieve the list of peer beliefs for this dimension
            if not peer_values: #If the firm has no peers, skip
                continue 

            # Awareness is a special case as it is binary (0 OR 1)
            if b == "awareness":
                if any(v == 1 for v in peer_values):
                    self.beliefs["awareness"] = 1
                continue  # Skip the continuous update for awareness

            # Continuous beliefs (0 -> 1)
            social_mean = sum(peer_values) / len(peer_values) # takes an average of peer values
            personal = self.beliefs[b] # agent's current belief
            baseline = self.beliefs_initial[b] # agent's initial belief

            realism_pull = (
                self.model.realism_pull_sociallyInfluencedVars if self.belief_types[b] == "subjective"
                else self.model.realism_pull_constraints
            ) # Use the correct realism pull based on belief type i.e., it is stronger for constraints (objective) than for subjective beliefs

            self.beliefs[b] += ( # New belief = old belief + 
                self.learning_rate_eff * (social_mean - personal) # (learning rate × peer signal gap) nudges beliefs towards the social average +
                + realism_pull * (baseline - personal) # (realism pull × gap from original belief)  pulls beliefs back toward where they started
                )
    
    def update_perceived_feasibility(self):
        self.feasible = (
            self.beliefs["resources"] > self.r_min_eff and
            self.beliefs["knowledge"] > self.k_min and
            self.beliefs["organisationalReadiness"] > self.or_min_eff and
            self.beliefs["publicTransport"] > self.pt_min and
            self.beliefs["awareness"] == 1
        ) # Feasible = True if all constraints  are above the minimum thresholds, otherwise it is false

    def update_prob_adoption(self):
        size_mid = self.SIZE_MIDPOINT.get(self.size_cat) # Get the midpoint of the size bin
        if size_mid is None: # For debuggin
            raise ValueError(f"Unknown size category: {self.size}")

        if self.feasible:                                                            # If the WTP is perceived as feasible, then:
            perceived_net_benefit = self.model.obj_net_benefit_min + (                  # Calculate the perceived net benefit of adopting a WTP
            (self.model.obj_net_benefit_max - self.model.obj_net_benefit_min) * (self.beliefs["motivations"] - ((1.5 +  math.log(size_mid, 0.01))*self.beliefs["perceivedBarriers"])))  # estimated net benefit is equal to the minimum plausible net benefit plus a range of plausible net benefit values that depends on the perception of costs and benefits of adoption (which are scaled between 0 and 1). 
            self.prob_adoption = 1 / (1 + math.exp(-perceived_net_benefit))          # The logit (sigmoidal) function converts the perceived net benefit into a probability of adoption
        else:
            self.prob_adoption = 0                                                   # If a WTP is not perceived as feasible, then the probability of adoption is 0

    def update_adoption_status(self):                                                # Update the adoption stage based on the probability of adoption
        # These are vairables local to this function only (i.e., they exist only during executation and can't be accessed from outside): old_stage, candidate_stage, allowed_stage
        old_stage = self.adoption_stage    # Record the adoption stage
        p = self.prob_adoption              # Based on probability of adoption update their new_stage... 
        #  1. First find the candidate stage implied by their probability of adoption
        # These thresholds will be based on survey data eventually
        if p < 0.14:
            candidate_stage  = "A. No intention"
        elif p < 0.58:
            candidate_stage  = "B. May consider"
        elif p < 0.79:
            candidate_stage  = "C. Is developing a WTP"
        elif p < 0.85:
            candidate_stage  = "D. Has a low efficacy WTP"
        else:
            candidate_stage  = "E. Has an effective WTP"
            # Agents are allowed to regress to lower stages of adoption

        # 2. Check time lag rules (i.e., block transitions until they have had time to transition)
        allowed_stage = candidate_stage 

        # DevelopersLag: Must spend at least 2 ticks in development before moving to adoption
        # if old_stage == "C. Is developing a WTP" and candidate_stage in {"D. Has a low efficacy WTP", "E. Has an effective WTP"}: # If they have been developing a WTP and their candidate stage suggests adoption 
        #     if self.time_in_stage < 2: # and if they have been in this stage for less than 2 ticks
        #         allowed_stage = old_stage # Then they are not allowed to progress
        # I think because of the sigmoidal fucntion, they just skipped stage c 

        if candidate_stage in {"D. Has a low efficacy WTP", "E. Has an effective WTP"} and old_stage in {"A. No intention", "B. May consider"}:
            if self.time_in_stage < 3: # and if they have been in this stage for less than 2 ticks
                allowed_stage = "C. Is developing a WTP"


        # ImprovementLag: Must spend at least 2 ticks in D before moving to E
        if old_stage == "D. Has a low efficacy WTP" and candidate_stage == "E. Has an effective WTP": # If they have had a bad plan and want to more to a better plan
            if self.time_in_stage < 3: # They must be in this stage for two ticks
                allowed_stage = old_stage # Then they are not allowed to progress

        # 3. Commit the stage
        self.adoption_stage = allowed_stage # The adoption stage is whatever is allowed, either the candidate stage or old stage depending on the lags

        # 4. Reset counter only if the stage actually changed
        if self.adoption_stage != old_stage: # If adoption stage is different to old stage
            self.time_in_stage = 0 # Reset the counter 