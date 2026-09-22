# Foundational agents

Flow configs fetched from the AI Catalog by `fetch-foundational-agents` at image
build. Each agent is a regular feature directory,
`<flow_id>/config/<semver>.yml`, so flow-config discovery registers it like any
bundled feature. A fetched agent must not share its name with a bundled feature:
discovery fails at boot on a duplicate flow ID.

A `.gitkeep` keeps this directory in Git and in the image; fetched agents are
gitignored. See `docs/flow_registry/index.md`, "Syncing Flows from the AI Catalog".
