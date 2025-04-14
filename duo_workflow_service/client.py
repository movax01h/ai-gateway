import os

import grpc
from dotenv import load_dotenv
from gitlab_cloud_connector import (
    CloudConnectorUser,
    GitLabUnitPrimitive,
    TokenAuthority,
    UserClaims,
)

from contract import contract_pb2, contract_pb2_grpc


def generate_client_events():
    yield contract_pb2.ClientEvent(
        startRequest=contract_pb2.StartWorkflowRequest(
            clientVersion="1", workflowDefinition="test", goal="test"
        )
    )


def test_generate_token():
    port = int(os.environ.get("PORT", "50052"))
    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = contract_pb2_grpc.DuoWorkflowStub(channel)

    # Generate a token by the following steps:
    # 1. gdk start
    # 1. gdk rails console
    # ```
    # ::Gitlab::CloudConnector::SelfIssuedToken.new(
    #   audience: "gitlab-duo-workflow-service",
    #   subject: Gitlab::CurrentSettings.uuid,
    #   scopes: ["duo_workflow_execute_workflow", "duo_workflow_get_user_token"]).encoded
    # ```
    token = (
        os.environ.get("DUO_WORKFLOW_CLIENT_GDK_GL_TOKEN")
        or "<paste-your-local-gdk-token-here>"
    )

    # To get your gitlab_instance_id, run in gdk rails console:
    # ```
    # puts Gitlab::CurrentSettings.uuid
    # ```
    gitlab_instance_id = (
        os.environ.get("DUO_WORKFLOW_CLIENT_GDK_GL_INSTANCE_ID")
        or "<paste-your-local-gdk-instance-id-here>"
    )

    duo_seat_count = "111"

    metadata = [
        ("authorization", f"Bearer {token}"),
        ("x-gitlab-authentication-type", "oidc"),
        ("x-gitlab-realm", "saas"),
        ("x-gitlab-instance-id", gitlab_instance_id),
        ("x-gitlab-duo-seat-count", duo_seat_count),
        # ("x-gitlab-global-user-id", global_user_id), <-- we don't need user id while we test with realm == "saas"
    ]

    request = contract_pb2.GenerateTokenRequest()
    response = stub.GenerateToken(request, metadata=metadata)
    print("Generated token:")
    print(f"Token: {response.token}")
    print(f"Expires at: {response.expiresAt}")


def test_execute_workflow():
    port = int(os.environ.get("PORT", "50052"))
    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = contract_pb2_grpc.DuoWorkflowStub(channel)

    # To test GitLab authority, set DUO_WORKFLOW_CLIENT_TOKEN from the token generated in GitLab
    # token = os.environ.get("DUO_WORKFLOW_CLIENT_TOKEN", "your_bearer_token_here")
    # End test GitLab authority

    # To test local authority:
    local_signing_key = os.environ.get("DUO_WORKFLOW_SELF_SIGNED_JWT__SIGNING_KEY")
    ta = TokenAuthority(local_signing_key)
    global_user_id_and_subject = "777"
    gitlab_instance_id = (
        os.environ.get("DUO_WORKFLOW_CLIENT_GDK_GL_INSTANCE_ID")
        or "<paste-your-local-gdk-instance-id-here>"
    )
    duo_seat_count = "111"
    gitlab_realm = "self-managed"
    current_user = CloudConnectorUser(
        authenticated=True, claims=UserClaims(duo_seat_count=duo_seat_count)
    )
    token, _ = ta.encode(
        global_user_id_and_subject,
        gitlab_realm,
        current_user,
        gitlab_instance_id,
        [GitLabUnitPrimitive.DUO_WORKFLOW_EXECUTE_WORKFLOW],
    )
    # End test local authority

    metadata = [
        ("authorization", f"Bearer {token}"),
        ("x-gitlab-authentication-type", "oidc"),
        ("x-gitlab-realm", "self-managed"),
        ("x-gitlab-global-user-id", global_user_id_and_subject),
        ("x-gitlab-duo-seat-count", duo_seat_count),
    ]

    responses = stub.ExecuteWorkflow(generate_client_events(), metadata=metadata)
    for response in responses:
        print("Received response:")
        print(response)


if __name__ == "__main__":
    load_dotenv()

    test_generate_token()
    test_execute_workflow()
