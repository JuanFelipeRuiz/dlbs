"""
W&B Model Downloader
====================

Downloads model artifacts from Weights & Biases (W&B) based on
a provided CSV file containing a model overview.

It can:
- Iterate through a CSV of model runs.
- For each run, find the specified model artifact.
- Download the artifact to a local path defined in the CSV.

Usage:
    python -m dlbs.wandb_api.get_models --csv_path models/wandb_runs.csv
    python -m dlbs.wandb_api.get_models --csv_path models/wandb_runs.csv --run_ids run1 run2
"""

import os
import argparse
import pandas as pd
import wandb

# Default W&B settings matching the new repo (from configs/baseline_no_aug.yaml)
DEFAULT_ENTITY = "dlbs-jf"
DEFAULT_PROJECT = "runs_segment"


def download_all_models(api, df, entity=DEFAULT_ENTITY, project=DEFAULT_PROJECT):
    """Download all models from the DataFrame."""
    successful = 0
    failed = 0
    
    for _, row in df.iterrows():
        if pd.isna(row.get("model_artifact")):
            continue
        try:
            download_model(api, row, entity=entity, project=project)
            successful += 1
        except Exception as e:
            print(f"Failed to download model for run {row.get('run_id')}: {e}")
            failed += 1
    
    print(f"\nDownload complete: {successful} successful, {failed} failed")
    return successful, failed


def download_model(api, entry, entity=DEFAULT_ENTITY, project=DEFAULT_PROJECT):
    """
    Download a single model artifact based on the entry.
    
    Args:
        api: W&B API object
        entry: Row from the model overview DataFrame (dict-like)
        entity: W&B entity
        project: W&B project name
    """
    run_id = entry["run_id"]
    artifact_name = entry["model_artifact"]
    
    if pd.isna(artifact_name) or not artifact_name:
        print(f"No model artifact for run {run_id}, skipping...")
        return None
    
    run = api.run(f"{entity}/{project}/{run_id}")
    
    artifact = next(
        (a for a in run.logged_artifacts() if a.name == artifact_name),
        None
    )

    if artifact is None:
        print(f"Artifact '{artifact_name}' not found in run {run_id}")
        return None

    # Determine download path
    model_path = entry.get("model_path")
    if pd.isna(model_path) or not model_path:
        model_path = os.path.join("models/weights", artifact_name)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)
    
    # Download the artifact
    path = artifact.download(root=model_path)
    print(f"Downloaded artifact '{artifact_name}' from run {run_id} to {path}")
    return path


def download_models_from_csv(
    csv_path,
    entity=DEFAULT_ENTITY,
    project=DEFAULT_PROJECT,
    run_ids=None
):
    """
    Download models from a CSV file.
    
    Args:
        csv_path: Path to the model overview CSV
        entity: W&B entity
        project: W&B project name
        run_ids: Optional list of run_ids to filter downloads
        
    Returns:
        Tuple of (successful_count, failed_count)
    """
    api = wandb.Api()
    df = pd.read_csv(csv_path)
    
    if run_ids:
        df = df[df["run_id"].isin(run_ids)]
        print(f"Filtered to {len(df)} runs")
    
    return download_all_models(api, df, entity=entity, project=project)


def wandb_model_argparser():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description="Download model artifacts from W&B.")
    parser.add_argument("--csv_path", type=str, default="models/wandb_runs.csv", help="Path to model overview CSV")
    parser.add_argument("--entity", type=str, default=DEFAULT_ENTITY, help=f"W&B entity (default: {DEFAULT_ENTITY})")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT, help=f"W&B project (default: {DEFAULT_PROJECT})")
    parser.add_argument("--run_ids", type=str, nargs='*', default=None, help="Specific run IDs to download")
    return parser.parse_args()


if __name__ == "__main__":
    args = wandb_model_argparser()
    
    download_models_from_csv(
        csv_path=args.csv_path,
        entity=args.entity,
        project=args.project,
        run_ids=args.run_ids
    )
