#!/usr/bin/env python3
"""Generate overrides for default models.

Usage: python models-override.py "claude_sonnet_4_6,claude_sonnet_4_5_20250929"
"""

import json
import sys
import yaml
from pathlib import Path

target = (
    sys.argv[1].split(",")
    if len(sys.argv) > 1
    else ["claude_sonnet_4_6", "claude_sonnet_4_5_20250929"]
)

with open(Path.cwd() / "ai_gateway/model_selection/unit_primitives.yml") as f:
    features = yaml.safe_load(f)["configurable_unit_primitives"]

result = {}
for feature in features:
    name = feature.get("feature_setting")
    defaults = feature.get("default_models", [])
    selectable = feature.get("selectable_models", [])

    # Skip if already using target or no defaults
    if not name or not defaults or any(m in target for m in defaults):
        continue

    valid = next((m for m in target if m in selectable), None)
    if not valid:
        continue

    result[name] = [valid]

print(json.dumps(result, separators=(",", ":")))
