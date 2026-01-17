#!/usr/bin/env python3
"""Initialize MLflow experiment for GravityViT."""

import mlflow


def setup_mlflow_experiment():
    """Set up the GravityViT MLflow experiment."""
    # Set experiment name (creates if doesn't exist)
    experiment_name = "gravityvit"
    mlflow.set_experiment(experiment_name)

    # Get experiment info
    experiment = mlflow.get_experiment_by_name(experiment_name)

    print(f"MLflow experiment '{experiment_name}' initialized!")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Artifact location: {experiment.artifact_location}")
    print("\nTo view experiments, run: mlflow ui")
    print("Then open http://localhost:5000 in your browser")


if __name__ == "__main__":
    setup_mlflow_experiment()
