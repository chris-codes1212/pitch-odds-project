import os
import re
import pickle
import joblib
import wandb


def load_production_model(
    entity, project, model_name="pitch-odds-model"
):
    """
    Load production model pipeline from W&B artifact.
    Returns: model
    """

    # Login to W&B (uses WANDB_API_KEY env variable)
    wandb.login(key=os.environ["WANDB_API_KEY"])

    api = wandb.Api()

    # Fetch the production artifact
    artifact = api.artifact(
        f"{entity}/{project}/{model_name}:production", type="model"
    )

    labels = artifact.metadata.get("labels", ['strike', 'ball', 'hit', 'ns foul'])

    # Download the artifact locally
    artifact_path = artifact.download()

    # Load Keras model
    model_file = f"{artifact_path}/model.pkl"
    model = joblib.load(model_file)

    return model, labels