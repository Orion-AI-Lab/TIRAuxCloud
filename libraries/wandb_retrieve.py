import wandb
import pandas as pd

def wandinit(paramsdict, group, 
             entity=None,
             project=None):
    if entity is None:
        entity=""
    if project is None:
        project=""
    wandbrun = wandb.init(
        entity=entity,
        project=project,
        group=group,
        config=paramsdict)
    return wandbrun

def get_filtered_wandb_runs(entity=None, project=None, filters=None):
    """
    Retrieves W&B runs from a specified project that match given filters.

    Args:
        entity (str): Your W&B username or team name.
        project (str): The name of your W&B project.
        filters (dict, optional): A dictionary of filters to apply.
                                  Keys can be "config.param_name" or "summary_metrics.metric_name".
                                  Values can be exact matches or use operators like "$gt", "$lt", "$in", etc.
                                  Defaults to None (no filters).

    Returns:
        pandas.DataFrame: A DataFrame containing the IDs, names, URLs, config parameters,
                          and summary metrics of the matching runs.
    """
    api = wandb.Api()
    run_data = []

    entity="" if entity is None else entity
    project="Cloud Segmentation with Thermal band" if project is None else project

    print(f"Searching for runs in project '{entity}/{project}' with filters: {filters}")

    try:
        # Fetch runs based on the provided path and filters
        runs = api.runs(
            path=f"{entity}/{project}",
            filters=filters
        )

        for run in runs:
            row = {
                #"Run ID": run.id,
                "Name": run.name,
                #"State": run.state,
                #"URL": run.url
                "Group": run.group
            }

            # Add all parameters from run.config
            for key, value in run.config.items():
                if not key.startswith('_'): # Exclude W&B internal config keys
                    row[f"config_{key}"] = value # Prefix with 'config_' for clarity

            run_data.append(row)

        if run_data:
            df = pd.DataFrame(run_data)
            print(f"Found {len(df)} runs matching the criteria.")
            return df
        else:
            print("No runs found matching the specified filters.")
            return pd.DataFrame()

    except wandb.errors.CommError as e:
        print(f"W&B API communication error: {e}. Check your entity, project, and API key.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()