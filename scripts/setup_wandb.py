#!/usr/bin/env python3
"""Initialize Weights & Biases project for GravityViT experiments."""

import wandb


def setup_wandb_project():
    """Set up the GravityViT W&B project."""
    # Initialize wandb (will prompt for login if not authenticated)
    wandb.login()

    # Create a test run to initialize the project
    run = wandb.init(
        project="gravityvit",
        name="project-init",
        notes="Initial project setup",
        tags=["setup"],
    )

    # Log project configuration
    run.config.update(
        {
            "project": "GravityViT",
            "description": "Vision Transformer for LIGO Gravitational Wave Glitch Classification",
            "dataset": "Gravity Spy",
            "num_classes": 22,
        }
    )

    print("W&B project 'gravityvit' initialized successfully!")
    print(f"Project URL: {run.get_project_url()}")

    run.finish()


if __name__ == "__main__":
    setup_wandb_project()
