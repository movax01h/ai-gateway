import json
import os
from pathlib import Path


def get_yaml_versions(directory):
    versions = []
    for f in Path(directory).glob("*"):
        if f.suffix in [".yml", ".yaml"]:
            versions.append(f.stem)
    return sorted(versions)


def process_model_directory(model_path):
    if model_path.is_dir():
        versions = get_yaml_versions(model_path)
        if versions:
            return versions
    return None


def process_feature_with_subfeatures(feature_path):
    feature_dict = {}
    for subfeature in os.listdir(feature_path):
        subfeature_path = feature_path / subfeature
        if not subfeature_path.is_dir():
            continue

        model_families = {}
        for model_dir in os.listdir(subfeature_path):
            model_path = subfeature_path / model_dir
            versions = process_model_directory(model_path)
            if versions:
                model_families[model_dir] = versions

        if model_families:
            feature_dict[subfeature] = model_families

    return feature_dict


def process_feature_without_subfeatures(feature_path):
    model_families = {}
    for model_dir in os.listdir(feature_path):
        model_path = feature_path / model_dir
        versions = process_model_directory(model_path)
        if versions:
            model_families[model_dir] = versions

    return model_families


def get_feature_models():
    base_path = "ai_gateway/prompts/definitions"
    feature_models = {}

    target_features_with_subfeatures = ["chat", "code_suggestions"]
    all_features = [f for f in os.listdir(base_path) if Path(base_path, f).is_dir()]
    target_features_without_subfeatures = [
        f for f in all_features if f not in target_features_with_subfeatures
    ]

    for feature in target_features_with_subfeatures:
        feature_path = Path(base_path) / feature
        if feature_path.is_dir():
            result = process_feature_with_subfeatures(feature_path)
            if result:
                feature_models[feature] = result

    for feature in target_features_without_subfeatures:
        feature_path = Path(base_path) / feature
        if feature_path.is_dir():
            result = process_feature_without_subfeatures(feature_path)
            if result:
                feature_models[feature] = result

    return feature_models


feature_models = get_feature_models()
with open("public/prompt_directory_structure.json", "w") as f:
    json.dump(feature_models, f, indent=2)
