from Model_NoRealismPullCompInfInc4MotPBonly import AdoptionModel
import numpy as np


def make_model(active_shocks, shock_parameters):
    return AdoptionModel(
        learning_rate=0.65,
        competitor_inference_increment=0.04,
        init_positive_shift=0.1,
        B_min_time=2,
        C_min_time=2,
        D_min_time=4,
        B_constraints=2,
        D_constraints=3,
        logit_pivot=180,
        logit_steepness=0.04,
        cap_first_tick_at="D. Has a WTP",
        collect_agent_data=False,
        num_agents=500,
        organisationalReadiness_min=0.4367,
        publicTransport_min=0.5883,
        resource_min=0.5683,
        knowledge_min=0.4667,
        obj_net_benefit_min=188,
        obj_net_benefit_max=250,
        active_shocks=active_shocks,
        shock_parameters=shock_parameters,
        seed=1
    )


def summarise(values):
    return {
        "min": round(float(np.min(values)), 4),
        "mean": round(float(np.mean(values)), 4),
        "max": round(float(np.max(values)), 4),
    }


# 1. Policy champion should lower organisational readiness threshold
model = make_model(
    active_shocks={"policyChampion"},
    shock_parameters={"policyChampion": 0.15}
)

or_thresholds = [
    model.effective_organisationalReadiness_min(a)
    for a in model.agents
]

print("\nPolicy champion")
print("Base organisational readiness threshold:", model.organisationalReadiness_min)
print("Effective thresholds:", summarise(or_thresholds))
print("Number lower than base:", sum(x < model.organisationalReadiness_min for x in or_thresholds))


# 2. Subsidy should lower resource threshold
model = make_model(
    active_shocks={"subsidy"},
    shock_parameters={"subsidy": 0.15}
)

resource_thresholds = [
    model.effective_resource_min(a)
    for a in model.agents
]

print("\nSubsidy")
print("Base resource threshold:", model.resource_min)
print("Effective thresholds:", summarise(resource_thresholds))
print("Number lower than base:", sum(x < model.resource_min for x in resource_thresholds))


# 3. Accreditation should increase learning rate and competitor inference increment
model = make_model(
    active_shocks={"accreditationAward"},
    shock_parameters={"accreditationAward": 0.15}
)

learning_rates = [
    model.effective_learning_rate(a)
    for a in model.agents
]

competitor_increments = [
    model.effective_competitor_inference_increment(a)
    for a in model.agents
]

print("\nAccreditation award")
print("Base learning rate:", model.learning_rate)
print("Effective learning rates:", summarise(learning_rates))
print("Number higher than base:", sum(x > model.learning_rate for x in learning_rates))

print("Base competitor inference increment:", model.competitor_inference_increment)
print("Effective competitor increments:", summarise(competitor_increments))
print("Number higher than base:", sum(x > model.competitor_inference_increment for x in competitor_increments))


# 4. Infrastructure investment should lower public transport threshold
# This only works if your agents actually call this function in update_perceived_feasibility().
model = make_model(
    active_shocks={"infrastructureInvestment"},
    shock_parameters={"infrastructureInvestment": 0.15}
)

pt_thresholds = [
    model.infrastructureInvestment()
    for a in model.agents
]

print("\nInfrastructure investment")
print("Base public transport threshold:", model.publicTransport_min)
print("Effective thresholds:", summarise(pt_thresholds))
print("Number lower than base:", sum(x < model.publicTransport_min for x in pt_thresholds))