# Foundational agents

Flow configs fetched from the AI Catalog by `fetch-foundational-agents` at image
build. `config/ai-catalog-agents.yml` in this directory lists the agents to fetch: each entry
names the agent's `reference` and its `catalog_id` on GitLab.com. To bundle a
new agent, add an entry there.

Each fetched agent is a regular feature directory,
`<reference>/config/<semver>.yml`, so flow-config discovery registers it like any
bundled feature. A fetched agent must not share its name with a bundled feature:
discovery fails at boot on a duplicate flow ID.

A `.gitkeep` keeps this directory in Git and in the image; fetched agents are
gitignored. See `docs/flow_registry/index.md`, "Syncing Flows from the AI Catalog".
