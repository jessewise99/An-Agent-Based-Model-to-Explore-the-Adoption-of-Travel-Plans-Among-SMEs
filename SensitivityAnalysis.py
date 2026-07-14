############################################
#     Model  - Batch Runner                #
#     Date: 2026-07-08                     #
#     Author: Jesse Wise                   #
#     Purpose: Local Sensitivity Analysis  #
############################################

from Model_NoRealismPullCompInfInc4MotPBonly import AdoptionModel
import pandas as pd
import pyreadr

T = 31
N = 500
N_RUNS = 100

# Baseline parameters
baseline_params = {
    "learning_rate": 0.65,
    "competitor_inference_increment": 0.04,
    "init_positive_shift": 0.1,
    "logit_pivot": 180.0,
    "logit_steepness": 0.04,
    "organisationalReadiness_min": 0.4367,
    "publicTransport_min": 0.5883,
    "resource_min": 0.5683,
    "knowledge_min": 0.4667,
    "obj_net_benefit_min": 188.0,
    "obj_net_benefit_max": 250.0,
}

# One-at-a-time local sensitivity values
sensitivity_values = {
    param: [value * 0.95, value, value * 1.05]
    for param, value in baseline_params.items()
}

sensitivity_results = []

for varied_param, values in sensitivity_values.items():
    for value in values:
        for seed in range(N_RUNS):

            params = baseline_params.copy()
            params[varied_param] = value

            print(
                f"Running {varied_param}={value}, "
                f"Run={seed + 1}/{N_RUNS}"
            )

            model = AdoptionModel(
                learning_rate=params["learning_rate"],
                competitor_inference_increment=params["competitor_inference_increment"],
                init_positive_shift=params["init_positive_shift"],
                B_min_time=2,
                C_min_time=2,
                D_min_time=4,
                B_constraints=2,
                D_constraints=3,
                logit_pivot=params["logit_pivot"],
                logit_steepness=params["logit_steepness"],
                cap_first_tick_at="D. Has a WTP",
                collect_agent_data=False,
                num_agents=N,
                organisationalReadiness_min=params["organisationalReadiness_min"],
                publicTransport_min=params["publicTransport_min"],
                resource_min=params["resource_min"],
                knowledge_min=params["knowledge_min"],
                obj_net_benefit_min=params["obj_net_benefit_min"],
                obj_net_benefit_max=params["obj_net_benefit_max"],
                seed=seed, # I forgot to do this for calibration, but I thought it was more important to add this for reproducibility. At 100 runs the coeff of var stabilises.=
                active_shocks=None,
                shock_parameters=None
            )

            for _ in range(T):
                model.step()

            df = model.datacollector.get_model_vars_dataframe().reset_index()

            df["Seed"] = seed
            df["varied_param"] = varied_param
            df["param_value"] = value

            for param_name, param_value in params.items():
                df[param_name] = param_value

            sensitivity_results.append(df)

results_df = pd.concat(sensitivity_results, ignore_index=True)

print(f"The results have {len(results_df)} rows.")
print(f"The columns of the data frame are {list(results_df.columns)}.")

run_check = (
    results_df[
        ["varied_param", "param_value", "Seed"]
    ]
    .drop_duplicates()
)

print(
    run_check
    .groupby(["varied_param", "param_value"])
    .size()
)

print(
    run_check.groupby(
        ["varied_param", "param_value"]
    )["Seed"].nunique()
)

pyreadr.write_rds("batch_results_LocalSensitivityAnalysis.rds", results_df)

print("Finished saving .rds file")