############################################
#     Model - Batch Runner                 #
#     Date: 2026-07-14                     #
#     Author: Jesse Wise                   #
#     Purpose: Scenario Analysis to 2050   #
############################################

from Model_NoRealismPullCompInfInc4MotPBonly import AdoptionModel
import pandas as pd
import pyreadr

T = 55
N = 500
N_RUNS = 100
POLICY_START_YEAR = 2030

policy_names = [
    "infrastructureInvestment",
    "caseStudy",
    "proofOfROI",
    "subsidy",
    "policyChampion",
    "accreditationAward",
]

efficacies = [0.05, 0.10, 0.15]

scenario_results = []


def create_model(seed):
    """Create a model using the baseline parameterisation."""

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
        num_agents=N,
        organisationalReadiness_min=0.4367,
        publicTransport_min=0.5883,
        resource_min=0.5683,
        knowledge_min=0.4667,
        obj_net_benefit_min=188,
        obj_net_benefit_max=250,
        seed=seed,
        active_shocks=set(),
        shock_parameters={},
    )


def run_model(model, policy=None, efficacy=0.0):
    """Run one model realisation and return its model-level data."""

    for _ in range(T):
        if (
            policy is not None
            and model.current_year == POLICY_START_YEAR
        ):
            model.activate_shock(policy, efficacy)

        model.step()

    return (
        model.datacollector
        .get_model_vars_dataframe()
        .reset_index()
    )


# Baseline: one run for each seed
for seed in range(N_RUNS):
    print(f"Running baseline, seed={seed + 1}/{N_RUNS}")

    model = create_model(seed)
    df = run_model(model)

    df["policy"] = "baseline"
    df["efficacy"] = 0.0
    df["Seed"] = seed
    df["policy_start_year"] = POLICY_START_YEAR

    scenario_results.append(df)


# Policy scenarios
for policy in policy_names:
    for efficacy in efficacies:
        for seed in range(N_RUNS):

            print(
                f"Running {policy}, efficacy={efficacy}, "
                f"seed={seed + 1}/{N_RUNS}"
            )

            model = create_model(seed)

            df = run_model(
                model=model,
                policy=policy,
                efficacy=efficacy,
            )

            df["policy"] = policy
            df["efficacy"] = efficacy
            df["Seed"] = seed
            df["policy_start_year"] = POLICY_START_YEAR

            scenario_results.append(df)


results_df = pd.concat(
    scenario_results,
    ignore_index=True,
)

print(f"The results have {len(results_df)} rows.")
print(f"The columns are {list(results_df.columns)}.")

print(
    results_df[
        ["policy", "efficacy", "Seed"]
    ]
    .drop_duplicates()
    .groupby(["policy", "efficacy"])
    .size()
)

pyreadr.write_rds(
    "batch_results_ScenarioAnalysis2050_100Seeds.rds",
    results_df,
)

print("Finished saving .rds file")