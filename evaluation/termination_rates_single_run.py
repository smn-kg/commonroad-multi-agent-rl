import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from cycler import cycler

import timeit

import wandb

def main():

    current_dir = Path(__file__).parent
    output_dir = current_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    entity = "safe-rl-commonroad"
    project = "safe-rl-autodrive"

    run_group_names = [
        "baseline",
        "masking/projection/baseline",
        "masking/distributional/baseline",
        "masking/distributional/simple_ent",
        "masking/ray/baseline",
        "masking/ray/no_observe_action_set",
    ]

    # run_group_name = run_group_names[0]
    run_group_name = run_group_names[3]  # Select the run group to plot

    metrics = [
        "test/goal_reaching_rate",
        "test/collision_rate",
        "test/max_s_position_rate",
        "test/off_road_rate",
        "test/no_safe_action",
        "test/no_driving_corridor",
        "test/initial_condition_invalid",
        "test/empty_relevant_action_set",
    ]

    map_run_group_name_to_string = {
        "baseline": "Baseline",
        "masking/projection/baseline": "Projection",
        "masking/distributional/baseline": "Distributional Mask (Actual Entropy)",
        "masking/distributional/simple_ent": "Distributional Mask",
        "masking/ray/baseline": "Ray Mask (Observe Action Set)",
        "masking/ray/no_observe_action_set": "Ray Mask"
    }

    map_metric_to_string = {
        "train/returns_stat/mean": "Mean Reward",
        "test/goal_reaching_rate": "Goal Reaching Rate",
        "test/collision_rate": "Collision Rate",
        "test/max_s_position_rate": "Missed Goal Rate",
        "test/off_road_rate": "Offroad Rate",
        "test/no_safe_action": "No Safe Action Rate",
        "test/no_driving_corridor": "No Driving Corridor Rate",
        "test/initial_condition_invalid": "Initial Condition Invalid Rate",
        "test/empty_relevant_action_set": "Empty Relevant Action Set Rate",
    }

    api = wandb.Api()

    
    collected_metric_data = {}
    for metric in metrics:
        print(f"Performing analysis for metric {metric}")
        run_group_data = process_run_group(api, entity, project, run_group_name, metric)

        collected_metric_data[metric] = run_group_data

    ibm_color_cycle = cycler(
        color=[
            "#648FFF",  # blue
            "#785EF0",  # purple
            "#DC267F",  # magenta
            "#FE6100",  # orange
            "#9467bd",  # purple
            "#FFB000",  # yellow
        ]
    )
    
    step_factor = 1e-4
    step_factor_name = "$10^{-4}$"

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({
        # "font.size": 8,
        # "axes.linewidth": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # "lines.linewidth": 1.2,
        "font.size": 10,
        # "axes.prop_cycle": ibm_color_cycle,
    })
    for metric, data in collected_metric_data.items():
        x = data.global_step.iloc[:, 0] * step_factor  * 400_000 / 409_000
        plt.plot(x, data[metric + "_mean"], label=f"{map_metric_to_string[metric]}")
        # plt.plot(x, data[metric], alpha=0.3, label=f"{metric} individual runs")
        plt.fill_between(
            x,
            data[metric + "_lower_95_ci"],
            data[metric + "_upper_95_ci"],
            alpha=0.3,
            # label="95% CI"
        )
    plt.xlabel(f"Step (in {step_factor_name})")
    plt.ylabel("Rate")
    plt.xlim(left=0)
    plt.xlim(right=400_000 * step_factor)
    plt.title(f"Termination Rates of {map_run_group_name_to_string[run_group_name]}", pad=15, fontsize=14)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),  # center below the axis
        ncol=len(collected_metric_data) / 2,  # number of columns = number of legend entries
        frameon=False
    )
    # plt.grid(True)
    plt.tight_layout(pad=0.5)
    plt.tick_params(width=0.8, length=4.5)
    plt.savefig(current_dir / "output" / f"termination_rates_{run_group_name.replace('/', '_')}.svg", bbox_inches="tight", dpi=300)
    plt.show()
    
    
    
def process_run_group(api, entity, project, run_group_name, metric):
    
    normalize_invalid_initial_condition = not metric in ["test/initial_condition_invalid", "train/returns_stat/mean"]
    
    runs = api.runs(
        f"{entity}/{project}",
        {
            "$and": [
                {"group": run_group_name},
            ]
        }
    )

    print(f"Found {len(runs)} runs")

    data = []
    for run in runs:
        print(f"Loading data of run {run.name}.  ", end="")

        # Normalize effect of invalid initial condition only for rate metrics

        start = timeit.default_timer()

             
        history = run.history(keys=[metric], samples=10_000_000, x_axis="global_step", pandas=True)


        if normalize_invalid_initial_condition:
            initial_condition_invalid_rates = run.history(keys=["test/initial_condition_invalid"], samples=10_000_000, x_axis="global_step", pandas=True)
            # Normalize the metric by the number of invalid initial conditions
            history[metric] = history[metric] / (1 - initial_condition_invalid_rates["test/initial_condition_invalid"] + 1e-6)
            history[metric] = history[metric].clip(0, 1)


        end = timeit.default_timer()
        print(f"Done. Loading data took {end - start:.1f} seconds")

        grouped = history.groupby("global_step")[metric].mean().reset_index()   # Take mean of values with same global_step
        grouped = grouped.ffill()
        data.append(grouped)
    
    # Concatenate all dataframes
    data = pd.concat(data, axis=1)

    data[metric + "_mean"] = data.loc[:, metric].mean(axis=1)

    upper_95_ci, lower_95_ci = bootstrap_ci(data[metric].to_numpy(), num_bootstrap=10000, ci=95)

    data[metric + "_upper_95_ci"] = upper_95_ci
    data[metric + "_lower_95_ci"] = lower_95_ci

    return data

    

def bootstrap_ci(data_matrix, num_bootstrap=10000, ci=95):
    """
    Compute bootstrap confidence intervals at each timestep.
    
    Parameters:
    - data_matrix: np.ndarray of shape (num_timesteps, num_runs)
    - num_bootstrap: number of bootstrap samples
    - ci: confidence interval percentage (e.g., 95)
    
    Returns:
    - lower_bounds: np.ndarray of shape (num_timesteps,)
    - upper_bounds: np.ndarray of shape (num_timesteps,)
    """
    num_timesteps, num_runs = data_matrix.shape
    lower_bounds = np.empty(num_timesteps)
    upper_bounds = np.empty(num_timesteps)

    for t in range(num_timesteps):
        samples = data_matrix[t, :]
        bootstrap_samples = np.random.choice(samples, size=(num_bootstrap, num_runs), replace=True)
        bootstrap_means = bootstrap_samples.mean(axis=1)
        lower_bounds[t] = np.percentile(bootstrap_means, (100 - ci) / 2)
        upper_bounds[t] = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)

    return upper_bounds, lower_bounds


if __name__ == "__main__":
    main()