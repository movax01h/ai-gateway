import json
from typing import Any, Type

from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


class GetGlqlSchemaInput(BaseModel):
    data_source: str = Field(
        default="all",
        description="""Which data source schema(s) to retrieve.

        Accepted values:
        - A single data source: "WorkItem", "MergeRequest", "Pipeline", "Job", or "Project"
        - Multiple comma-separated: "Pipeline,Job" (returns a dict keyed by source name)
        - "all" (default): returns every data source schema

        Examples:
        - "Pipeline" → returns the Pipeline schema (filters, display_fields, sort_fields)
        - "Pipeline,Job" → returns {"Pipeline": {...}, "Job": {...}}
        - "all" → returns all 5 data source schemas
        """,
    )


_SCHEMAS = {
    "WorkItem": {
        "type_values": [
            "Issue",
            "Epic",
            "Task",
            "Incident",
            "Ticket",
            "Objective",
            "KeyResult",
            "TestCase",
            "Requirement",
        ],
        "filters": [
            {"name": "project", "operators": ["="]},
            {"name": "group", "operators": ["="]},
            {"name": "assignee", "operators": ["=", "!=", "in"]},
            {"name": "author", "operators": ["=", "!=", "in"]},
            {"name": "cadence", "operators": ["=", "!=", "in"]},
            {"name": "closed", "operators": ["=", ">", "<", ">=", "<="]},
            {"name": "confidential", "operators": ["=", "!="]},
            {"name": "created", "operators": ["=", ">", "<", ">=", "<="]},
            {"name": 'customField("Name")', "operators": ["="]},
            {"name": "due", "operators": ["=", ">", "<", ">=", "<="]},
            {"name": "epic", "operators": ["=", "!=", "in"]},
            {
                "name": "health",
                "operators": ["=", "!="],
                "values": ["on track", "needs attention", "at risk"],
            },
            {"name": "id", "operators": ["=", "in"]},
            {"name": "iteration", "operators": ["=", "!=", "in"]},
            {"name": "label", "operators": ["=", "!=", "in"]},
            {"name": "milestone", "operators": ["=", "!=", "in"]},
            {"name": "myReaction", "operators": ["=", "!="]},
            {"name": "parent", "operators": ["=", "!=", "in"]},
            {
                "name": "state",
                "operators": ["="],
                "values": ["opened", "closed", "all"],
            },
            {"name": "status", "operators": ["="]},
            {"name": "subscribed", "operators": ["=", "!="]},
            {"name": "type", "operators": ["=", "in"]},
            {"name": "updated", "operators": ["=", ">", "<", ">=", "<="]},
            {"name": "weight", "operators": ["=", "!="]},
            {"name": "includeSubgroups", "operators": ["=", "!="]},
        ],
        "display_fields": [
            "title",
            "assignee",
            "author",
            "closed",
            "confidential",
            "created",
            "description",
            "due",
            "epic",
            "health",
            "id",
            "iteration",
            "labels",
            "lastComment",
            "milestone",
            "start (Epic only)",
            "state",
            "status",
            "subscribed",
            "type",
            "updated",
            "weight",
            "cadence",
            "parent",
            "project",
            "group",
            "progress",
            "color",
            "taskCompletionStatus",
            "timeEstimate",
            "totalTimeSpent",
            "popularity",
            "myReaction",
            "reference",
            "webUrl",
            "iid",
        ],
        "sort_fields": [
            "closed",
            "created",
            "due",
            "health",
            "milestone (not Epic)",
            "popularity",
            "start (Epic only)",
            "title",
            "updated",
            "weight (not Epic)",
        ],
    },
    "MergeRequest": {
        "type_values": ["MergeRequest"],
        "scope": ["project", "group"],
        "filters": [
            {"name": "project", "operators": ["="]},
            {"name": "group", "operators": ["="]},
            {"name": "approver", "operators": ["=", "!="]},
            {"name": "assignee", "operators": ["=", "!="]},
            {"name": "author", "operators": ["=", "!="]},
            {"name": "closed", "operators": ["=", ">", "<", ">=", "<="]},
            {"name": "created", "operators": ["=", ">", "<", ">=", "<="]},
            {"name": "deployed", "operators": ["=", ">", "<", ">=", "<="]},
            {"name": "draft", "operators": ["=", "!="]},
            {"name": "environment", "operators": ["="]},
            {"name": "id", "operators": ["=", "in"]},
            {"name": "includeSubgroups", "operators": ["=", "!="]},
            {"name": "label", "operators": ["=", "!="]},
            {"name": "merged", "operators": ["=", ">", "<", ">=", "<="]},
            {"name": "merger", "operators": ["="]},
            {"name": "milestone", "operators": ["=", "!="]},
            {"name": "myReaction", "operators": ["=", "!="]},
            {"name": "reviewer", "operators": ["=", "!="]},
            {"name": "sourceBranch", "operators": ["=", "in", "!="]},
            {
                "name": "state",
                "operators": ["="],
                "values": ["opened", "closed", "merged", "all"],
            },
            {"name": "subscribed", "operators": ["=", "!="]},
            {"name": "targetBranch", "operators": ["=", "in", "!="]},
            {"name": "updated", "operators": ["=", ">", "<", ">=", "<="]},
        ],
        "display_fields": [
            "title",
            "approver",
            "assignee",
            "author",
            "closed",
            "created",
            "deployed",
            "description",
            "draft",
            "id",
            "labels",
            "merged",
            "milestone",
            "reviewer",
            "sourceBranch",
            "sourceProject",
            "state",
            "subscribed",
            "targetBranch",
            "targetProject",
            "updated",
            "approved",
            "timeEstimate",
            "totalTimeSpent",
            "environment",
            "popularity",
            "myReaction",
            "project",
            "iid",
            "webUrl",
            "reference",
        ],
        "sort_fields": [
            "closed",
            "created",
            "merged",
            "milestone",
            "popularity",
            "title",
            "updated",
        ],
    },
    "Pipeline": {
        "type_values": ["Pipeline"],
        "filters": [
            {"name": "project", "operators": ["="], "required": "true"},
            {"name": "author", "operators": ["="]},
            {"name": "ref", "operators": ["="]},
            {
                "name": "scope",
                "operators": ["="],
                "values": ["branches", "tags", "finished", "pending", "running"],
            },
            {"name": "sha", "operators": ["="]},
            {"name": "source", "operators": ["="]},
            {
                "name": "status",
                "operators": ["="],
                "values": [
                    "canceled",
                    "canceling",
                    "created",
                    "failed",
                    "manual",
                    "pending",
                    "preparing",
                    "running",
                    "scheduled",
                    "skipped",
                    "success",
                    "waiting_for_callback",
                    "waiting_for_resource",
                ],
            },
            {"name": "updated", "operators": ["=", ">", "<", ">=", "<="]},
        ],
        "display_fields": [
            "id",
            "iid",
            "name",
            "status",
            "ref",
            "sha",
            "source",
            "duration",
            "coverage",
            "computeMinutes",
            "created",
            "started",
            "finished",
            "updated",
            "failedJobsCount",
            "totalJobs",
            "failureReason",
            "active",
            "complete",
            "latest",
            "retryable",
            "warnings",
            "path",
            "configSource",
            "committed",
            "yamlErrors",
            "yamlErrorMessages",
            "child",
            "stuck",
            "cancelable",
            "author",
            "project",
        ],
        "sort_fields": [],
    },
    "Job": {
        "type_values": ["Job"],
        "filters": [
            {"name": "project", "operators": ["="], "required": "true"},
            {"name": "kind", "operators": ["="], "values": ["bridge", "build"]},
            {"name": "pipeline", "operators": ["="], "note": "pipeline IID number"},
            {
                "name": "status",
                "operators": ["="],
                "values": [
                    "canceled",
                    "canceling",
                    "created",
                    "failed",
                    "manual",
                    "pending",
                    "preparing",
                    "running",
                    "scheduled",
                    "skipped",
                    "success",
                    "waiting_for_callback",
                    "waiting_for_resource",
                ],
            },
            {"name": "withArtifacts", "operators": ["=", "!="]},
        ],
        "display_fields": [
            "id",
            "name",
            "stage",
            "status",
            "kind",
            "duration",
            "coverage",
            "failureMessage",
            "created",
            "started",
            "finished",
            "queued",
            "refName",
            "shortSha",
            "source",
            "allowFailure",
            "retryable",
            "retried",
            "manualJob",
            "tags",
            "webPath",
            "project",
            "pipeline",
            "type",
            "active",
            "stuck",
            "cancelable",
            "playable",
            "scheduled",
            "triggered",
            "erased",
        ],
        "sort_fields": [],
    },
    "Project": {
        "type_values": ["Project"],
        "filters": [
            {
                "name": "namespace",
                "operators": ["="],
                "required": "true",
                "alias": "group",
            },
            {"name": "archivedOnly", "operators": ["=", "!="]},
            {"name": "hasCodeCoverage", "operators": ["=", "!="]},
            {"name": "hasVulnerabilities", "operators": ["=", "!="]},
            {"name": "includeArchived", "operators": ["=", "!="]},
            {"name": "includeSubgroups", "operators": ["=", "!="]},
            {"name": "issuesEnabled", "operators": ["=", "!="]},
            {"name": "mergeRequestsEnabled", "operators": ["=", "!="]},
        ],
        "display_fields": [
            "name",
            "fullPath",
            "path",
            "visibility",
            "id",
            "archived",
            "starCount",
            "forksCount",
            "openIssuesCount",
            "openMergeRequestsCount",
            "lastActivity",
            "issuesEnabled",
            "mergeRequestsEnabled",
            "webUrl",
            "duoFeaturesEnabled",
            "nameWithNamespace",
            "namespace",
            "group",
            "description",
            "forked",
            "secretPushProtectionEnabled",
            "hasVulnerabilities",
            "hasCodeCoverage",
        ],
        "sort_fields": ["fullPath", "lastActivity (desc only)", "path"],
    },
}


class GetGlqlSchema(DuoBaseTool):
    name: str = "get_glql_schema"
    description: str = """Get the GLQL schema for one or more data source types.

    Returns available filters (with valid operators and values), display_fields, and sort_fields.
    MUST be called before building any GLQL query to ensure only valid fields are used.

    Supports fetching a single source, multiple comma-separated sources, or "all" sources at once.
    Supported data sources: WorkItem, MergeRequest, Pipeline, Job, Project.
    """
    args_schema: Type[BaseModel] = GetGlqlSchemaInput

    async def _execute(self, data_source: str = "all", **_kwargs: Any) -> str:
        if data_source == "all":
            return json.dumps(_SCHEMAS)

        # Support comma-separated: "Pipeline,Job"
        sources = [s.strip() for s in data_source.split(",")]
        result = {}
        for src in sources:
            if src not in _SCHEMAS:
                return json.dumps(
                    {
                        "error": f"Unknown data source: {src}. Supported: {', '.join(_SCHEMAS.keys())}"
                    }
                )
            result[src] = _SCHEMAS[src]

        # Single source: return flat; multiple: return dict
        if len(result) == 1:
            return json.dumps(next(iter(result.values())))
        return json.dumps(result)

    def format_display_message(
        self, args: GetGlqlSchemaInput, _tool_response: Any = None
    ) -> str:
        return f"Looking up GLQL schema for {args.data_source}"
