import json
from typing import Any, Type

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from duo_workflow_service.gitlab.resource_resolver import resolve_identifier_to_path
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.merge_request import (
    MERGE_REQUEST_IDENTIFICATION_DESCRIPTION,
    MergeRequestResourceInput,
)
from duo_workflow_service.tools.queries.merge_requests import SET_REVIEWERS_MUTATION
from duo_workflow_service.tools.version_compatibility import (
    get_gitlab_version,
    supports_set_reviewers_mutation,
)


class AddMergeRequestReviewersInput(MergeRequestResourceInput):
    reviewer_usernames: list[str] = Field(
        description="Usernames of the reviewers to add. Reviewers already assigned to the merge request are "
        "kept. Pass an empty list when there is nobody to add.",
    )


# Appending reviewers is its own tool rather than an option on update_merge_request,
# for two reasons:
#
#   1. Tools are part of the stable v1 Flow Registry surface, so they carry the same
#      backwards-compatibility guarantee as the rest of it, and update_merge_request
#      is called by flows we don't own. Teaching it a second, GraphQL-backed code
#      path changes an existing tool's behaviour for every one of those callers.
#      A new tool is purely additive and cannot break them.
#   2. mergeRequestSetReviewers is an implementation detail we expect to replace —
#      suggested-reviewer records are the intended destination. Behind a tool whose
#      contract is only "add these reviewers, keep the ones already there", that
#      swap touches no flow config and no prompt.
class AddMergeRequestReviewers(DuoBaseTool):
    name: str = "add_merge_request_reviewers"
    # pylint: disable=line-too-long
    description: str = f"""Adds reviewers to a merge request, keeping the reviewers that are already assigned.

Use this rather than update_merge_request whenever you want to request more reviews: update_merge_request's
reviewer_ids replaces the whole reviewer list, so it removes reviewers somebody else assigned.

{MERGE_REQUEST_IDENTIFICATION_DESCRIPTION}

For example:
- Given project_id 13, merge_request_iid 9, and reviewer "jdoe", the tool call would be:
    add_merge_request_reviewers(project_id=13, merge_request_iid=9, reviewer_usernames=["jdoe"])
- Given the URL https://gitlab.com/namespace/project/-/merge_requests/103, the tool call would be:
    add_merge_request_reviewers(url="https://gitlab.com/namespace/project/-/merge_requests/103", reviewer_usernames=["jdoe"])
- Given there is nobody to add, the tool call would be:
    add_merge_request_reviewers(project_id=13, merge_request_iid=9, reviewer_usernames=[])
    """
    # pylint: enable=line-too-long
    args_schema: Type[BaseModel] = AddMergeRequestReviewersInput

    async def _execute(self, reviewer_usernames: list[str], **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        project_id = kwargs.pop("project_id", None)
        merge_request_iid = kwargs.pop("merge_request_iid", None)

        validation_result = self._validate_merge_request_url(
            url, project_id, merge_request_iid
        )

        if validation_result.errors:
            raise ToolException("; ".join(validation_result.errors))

        if not reviewer_usernames:
            return json.dumps(
                {"requested_reviewers": [], "message": "No reviewers to add."}
            )

        # Gated before the two network calls below: an unsupported instance rejects the
        # mutation as a top-level GraphQL error, which tells the model nothing about
        # what to do differently.
        if not supports_set_reviewers_mutation():
            raise ToolException(
                "Adding reviewers requires GitLab 19.2 or later. "
                f"This instance reports {get_gitlab_version()}."
            )

        # The mutation takes a namespace path (e.g. "group/subgroup/project"), so
        # resolve it from the project identifier validation already produced.
        # resolve_identifier_to_path reads path_with_namespace from the API for a
        # numeric id — the canonical namespace path, so relative URL roots on
        # self-managed instances can't skew it — and unquotes an already-path
        # identifier.
        project_path = await resolve_identifier_to_path(
            self.gitlab_client, str(validation_result.project_id), "project"
        )
        variables = {
            "input": {
                "projectPath": project_path,
                "iid": str(validation_result.merge_request_iid),
                "reviewerUsernames": reviewer_usernames,
                "operationMode": "APPEND",
            }
        }
        # operationMode APPEND unions the usernames with the merge request's live
        # reviewers server-side in one atomic call, so a reviewer somebody assigned
        # while this workflow was running is never dropped.
        response = await self.gitlab_client.graphql(SET_REVIEWERS_MUTATION, variables)

        payload = (response or {}).get("mergeRequestSetReviewers")
        if payload is None:
            raise ToolException(f"Failed to add reviewers: empty response {response}")
        if errors := payload.get("errors"):
            raise ToolException(f"Failed to add reviewers: {errors}")

        # "requested", not "added": the mutation drops usernames that don't resolve to
        # a visible user and still returns no errors, so empty errors is not proof the
        # assignment happened. Reporting back what was actually assigned needs
        # gitlab-org/gitlab#611232.
        return json.dumps({"requested_reviewers": reviewer_usernames})

    def format_display_message(
        self, args: AddMergeRequestReviewersInput, _tool_response: Any = None
    ) -> str:
        reviewers = ", ".join(args.reviewer_usernames) or "no reviewers"
        if args.url:
            return f"Add reviewers ({reviewers}) to merge request {args.url}"
        return (
            f"Add reviewers ({reviewers}) to merge request !{args.merge_request_iid} "
            f"in project {args.project_id}"
        )
