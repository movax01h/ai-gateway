import json
import os
from pathlib import Path


def get_yaml_versions(directory):
    versions = []
    for file in Path(directory).glob("*"):
        if file.suffix in [".yml", ".yaml"]:
            versions.append(file.stem)
    return sorted(versions)


def process_model_directory(model_path):
    if model_path.is_dir():
        versions = get_yaml_versions(model_path)
        if versions:
            return versions
    return None


def process_feature_with_subfeatures(path):
    feature_dict = {}
    for subfeature in os.listdir(path):
        subfeature_path = path / subfeature
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


def process_feature_without_subfeatures(path):
    model_families = {}
    for model_dir in os.listdir(path):
        model_path = path / model_dir
        versions = process_model_directory(model_path)
        if versions:
            model_families[model_dir] = versions

    return model_families


if __name__ == "__main__":
    BASE_PATH = "ai_gateway/prompts/definitions"
    feature_models = {}

    target_features_with_subfeatures = ["chat", "code_suggestions"]
    all_features = [
        file for file in os.listdir(BASE_PATH) if Path(BASE_PATH, file).is_dir()
    ]
    target_features_without_subfeatures = [
        f for f in all_features if f not in target_features_with_subfeatures
    ]

    for feature in target_features_with_subfeatures:
        feature_path = Path(BASE_PATH) / feature
        if feature_path.is_dir():
            result = process_feature_with_subfeatures(feature_path)
            if result:
                feature_models[feature] = result

    for feature in target_features_without_subfeatures:
        feature_path = Path(BASE_PATH) / feature
        if feature_path.is_dir():
            result = process_feature_without_subfeatures(feature_path)
            if result:
                feature_models[feature] = result

    with open("public/prompt_directory_structure.json", "w") as f:
        json.dump(feature_models, f, indent=2)
