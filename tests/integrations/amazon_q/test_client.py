from typing import Any, Optional
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from botocore.exceptions import ClientError
from fastapi import HTTPException
from pydantic import BaseModel

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.auth.glgo import GlgoAuthority
from ai_gateway.integrations.amazon_q.client import AmazonQClient, AmazonQClientFactory
from ai_gateway.integrations.amazon_q.errors import AWSException


# Create a custom ClientError subclass with the name "AccessDeniedException"
class AccessDeniedException(ClientError):
    def __init__(self):
        super().__init__(
            error_response={
                "Error": {"Code": "AccessDeniedException", "Message": "Access denied"}
            },
            operation_name="SendEvent",
        )


class TestAmazonQClientFactory:
    @pytest.fixture
    def mock_glgo_authority(self):
        return MagicMock(spec=GlgoAuthority)

    @pytest.fixture
    def mock_sts_client(self):
        mock_client = MagicMock()
        return mock_client

    @pytest.fixture
    def mock_boto3(self, mock_sts_client):
        with patch("ai_gateway.integrations.amazon_q.client.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_sts_client
            yield mock_boto3

    @pytest.fixture
    def amazon_q_client_factory(self, mock_glgo_authority, mock_boto3):
        return AmazonQClientFactory(
            glgo_authority=mock_glgo_authority,
            endpoint_url="https://mock.endpoint",
            region="us-east-1",
        )

    @pytest.fixture
    def mock_user(self):
        user = MagicMock(spec=StarletteUser)
        user.global_user_id = "test-user-id"
        user.cloud_connector_token = "mock-cloud-connector-token"
        user.claims = MagicMock(subject="test-session")
        return user

    def test_get_glgo_token(
        self, amazon_q_client_factory, mock_user, mock_glgo_authority
    ):
        mock_glgo_authority.token.return_value = "mock-token"
        token = amazon_q_client_factory._get_glgo_token(mock_user)

        mock_glgo_authority.token.assert_called_once_with(
            user_id="test-user-id", cloud_connector_token="mock-cloud-connector-token"
        )
        assert token == "mock-token"

    def test_missing_user_id_for_glgo_token(
        self, amazon_q_client_factory, mock_user, mock_glgo_authority
    ):
        mock_user.global_user_id = None

        with pytest.raises(HTTPException) as exc:
            amazon_q_client_factory._get_glgo_token(mock_user)
        assert exc.value.status_code == 400
        assert exc.value.detail == "User Id is missing"

    def test_glgo_token_raises_error(
        self, amazon_q_client_factory, mock_user, mock_glgo_authority
    ):
        mock_glgo_authority.token.side_effect = KeyError()

        with pytest.raises(HTTPException) as exc:
            amazon_q_client_factory._get_glgo_token(mock_user)
        assert exc.value.status_code == 500
        assert exc.value.detail == "Cannot obtain OIDC token"

    def test_get_aws_credentials(
        self, amazon_q_client_factory, mock_user, mock_sts_client
    ):
        mock_sts_client.assume_role_with_web_identity.return_value = {
            "Credentials": {
                "AccessKeyId": "mock-key",
                "SecretAccessKey": "mock-secret",
                "SessionToken": "mock-token",
            }
        }

        credentials = amazon_q_client_factory._get_aws_credentials(
            mock_user, token="mock-web-identity-token", role_arn="mock-role-arn"
        )

        mock_sts_client.assume_role_with_web_identity.assert_called_once_with(
            RoleArn="mock-role-arn",
            RoleSessionName="test-session",
            WebIdentityToken="mock-web-identity-token",
            DurationSeconds=43200,
        )
        assert credentials == {
            "AccessKeyId": "mock-key",
            "SecretAccessKey": "mock-secret",
            "SessionToken": "mock-token",
        }

    def test_get_aws_credentials_no_claims(
        self, amazon_q_client_factory, mock_user, mock_sts_client
    ):
        mock_user.claims = None
        mock_sts_client.assume_role_with_web_identity.return_value = {
            "Credentials": {
                "AccessKeyId": "mock-key",
                "SecretAccessKey": "mock-secret",
                "SessionToken": "mock-token",
            }
        }

        credentials = amazon_q_client_factory._get_aws_credentials(
            mock_user, token="mock-web-identity-token", role_arn="mock-role-arn"
        )

        mock_sts_client.assume_role_with_web_identity.assert_called_once_with(
            RoleArn="mock-role-arn",
            RoleSessionName="placeholder",
            WebIdentityToken="mock-web-identity-token",
            DurationSeconds=43200,
        )

        assert credentials == {
            "AccessKeyId": "mock-key",
            "SecretAccessKey": "mock-secret",
            "SessionToken": "mock-token",
        }

    def test_get_client(
        self, amazon_q_client_factory, mock_user, mock_glgo_authority, mock_sts_client
    ):
        with patch(
            "ai_gateway.integrations.amazon_q.client.AmazonQClient"
        ) as mock_q_client_class:
            mock_q_client_instance = MagicMock()
            mock_q_client_class.return_value = mock_q_client_instance

            credentials = {
                "AccessKeyId": "mock-key",
                "SecretAccessKey": "mock-secret",
                "SessionToken": "mock-token",
            }

            mock_glgo_authority.token.return_value = "mock-token"
            mock_sts_client.assume_role_with_web_identity.return_value = {
                "Credentials": credentials
            }

            client = amazon_q_client_factory.get_client(
                current_user=mock_user,
                role_arn="mock-role-arn",
            )

            mock_glgo_authority.token.assert_called_once_with(
                user_id="test-user-id",
                cloud_connector_token="mock-cloud-connector-token",
            )

            mock_sts_client.assume_role_with_web_identity.assert_called_once_with(
                RoleArn="mock-role-arn",
                RoleSessionName="test-session",
                WebIdentityToken="mock-token",
                DurationSeconds=43200,
            )

            mock_q_client_class.assert_called_once_with(
                url="https://mock.endpoint", region="us-east-1", credentials=credentials
            )

            assert client == mock_q_client_instance


class TestAmazonQClient:
    @pytest.fixture
    def send_message_params(self):
        return {
            "message": {"content": "test message"},
            "history": [
                {"userInputMessage": {"content": "response"}},
                {"assistantResponseMessage": {"content": "response"}},
            ],
            "conversationId": "conversation_id",
        }

    @pytest.fixture
    def mock_credentials(self):
        return {
            "AccessKeyId": "test-access-key",
            "SecretAccessKey": "test-secret-key",
            "SessionToken": "test-session-token",
        }

    @pytest.fixture
    def mock_application_request(self):
        class ApplicationRequest:
            client_id = "test-client-id"
            client_secret = "test-secret"
            instance_url = "https://test.example.com"
            redirect_url = "https://test.example.com/callback"

        return ApplicationRequest()

    @pytest.fixture
    def mock_event_request(self) -> Any:
        class Payload(BaseModel):
            first_field: str = "test field"
            second_field: int = 1
            third_field: Optional[str] = None

        class EventRequest:
            payload = Payload()

        return EventRequest()

    @pytest.fixture
    def mock_q_client(self):
        with patch(
            "ai_gateway.integrations.amazon_q.client.q_boto3.client"
        ) as mock_client:
            yield mock_client.return_value

    @pytest.fixture
    def q_client(self, mock_credentials, mock_q_client):
        return AmazonQClient(
            url="https://q-api.example.com",
            region="us-west-2",
            credentials=mock_credentials,
        )

    @pytest.fixture
    def params(self):
        return dict(
            clientId="test-client-id",
            clientSecret="test-secret",
            instanceUrl="https://test.example.com",
            redirectUrl="https://test.example.com/callback",
        )

    def test_init_creates_client_with_correct_params(self, mock_credentials):
        with patch(
            "ai_gateway.integrations.amazon_q.client.q_boto3.client"
        ) as mock_client:
            AmazonQClient(
                url="https://q-api.example.com",
                region="us-west-2",
                credentials=mock_credentials,
            )

            mock_client.assert_called_once_with(
                "q",
                region_name="us-west-2",
                endpoint_url="https://q-api.example.com",
                aws_access_key_id="test-access-key",
                aws_secret_access_key="test-secret-key",
                aws_session_token="test-session-token",
            )

    def test_create_auth_application_success(
        self, q_client, mock_q_client, mock_application_request, params
    ):
        q_client.create_or_update_auth_application(mock_application_request)
        mock_q_client.create_o_auth_app_connection.assert_called_once_with(**params)

        assert not mock_q_client.delete_o_auth_app_connection.called

    def test_update_auth_application_on_conflict(
        self, q_client, mock_q_client, mock_application_request, params
    ):
        error_response = {
            "Error": {"Code": "ConflictException", "Message": "A conflict occurred"}
        }
        mock_q_client.create_o_auth_app_connection.side_effect = [
            ClientError(error_response, "create_o_auth_app_connection"),
            None,
        ]

        q_client.create_or_update_auth_application(mock_application_request)

        mock_q_client.create_o_auth_app_connection.assert_has_calls(
            [call(**params), call(**params)]
        )
        mock_q_client.delete_o_auth_app_connection.assert_called_once()

    def test_raises_non_conflict_aws_errors(
        self, q_client, mock_q_client, mock_application_request
    ):
        error_response = {
            "Error": {"Code": "ValidationException", "Message": "invalid message"}
        }
        mock_q_client.create_o_auth_app_connection.side_effect = ClientError(
            error_response, "create_o_auth_app_connection"
        )

        with pytest.raises(AWSException):
            q_client.create_or_update_auth_application(mock_application_request)

        mock_q_client.create_o_auth_app_connection.assert_called_once()
        assert not mock_q_client.delete_o_auth_app_connection.called

    @pytest.mark.parametrize(
        "event_id,payload,client_error,expected_exception",
        [
            # Happy path - successful event sending
            ("Quick Action", '{"test": "data"}', None, None),
            # Test AccessDeniedException with retry
            (
                "Quick Action",
                '{"test": "data"}',
                AccessDeniedException(),
                None,
            ),
            # Test other ClientError
            (
                "Quick Action",
                '{"test": "data"}',
                ClientError(
                    error_response={
                        "Error": {"Code": "OtherError", "Message": "Error"}
                    },
                    operation_name="SendEvent",
                ),
                AWSException,
            ),
        ],
    )
    def test_send_event(
        self, q_client, event_id, payload, client_error, expected_exception
    ):
        """Tests event sending with various scenarios."""
        # Setup mock request
        mock_request = Mock()
        mock_request.payload.model_dump_json.return_value = payload
        mock_request.event_id = event_id
        mock_request.code = "test_code"

        q_client._retry_send_event = Mock(return_value={"Success": True})

        if client_error:
            # Configure mock to raise exception on first call
            q_client._send_event = Mock(side_effect=[client_error, {"Success": True}])
        else:
            # Configure mock to return successfully
            q_client._send_event = Mock(return_value={"Success": True})

        if expected_exception:
            with pytest.raises(expected_exception):
                q_client.send_event(mock_request)
        else:
            # Should not raise any exception
            q_client.send_event(mock_request)

            if (
                client_error
                and isinstance(client_error, ClientError)
                and client_error.response["Error"]["Code"] == "AccessDeniedException"
            ):
                # Verify _send_event was called first and raised the exception
                q_client._send_event.assert_called_with(event_id, payload)
                # Verify retry was called with correct parameters
                q_client._retry_send_event.assert_called_once_with(
                    client_error, mock_request.code, payload, event_id
                )
            else:
                # Verify normal _send_event was called
                q_client._send_event.assert_called_once_with(event_id, payload)

    def test_generate_code_recommendations(
        self, q_client, mock_q_client, mock_event_request
    ):
        q_client.generate_code_recommendations(
            {"fileContext": {"context": "content"}, "maxResults": 1}
        )
        mock_q_client.generate_code_recommendations.assert_called_once_with(
            fileContext={"context": "content"},
            maxResults=1,
        )

    def test_delete_o_auth_app_connection_success(self, q_client, mock_q_client):
        q_client.delete_o_auth_app_connection()
        mock_q_client.delete_o_auth_app_connection.assert_called_once_with()

    def test_delete_o_auth_app_connection_on_conflict(
        self, q_client, mock_q_client, mock_application_request, params
    ):
        error_response = {
            "Error": {"Code": "ConflictException", "Message": "A conflict occurred"}
        }
        mock_q_client.delete_o_auth_app_connection.side_effect = ClientError(
            error_response, "delete_o_auth_app_connection"
        )

        q_client.delete_o_auth_app_connection()

        mock_q_client.delete_o_auth_app_connection.assert_called_once_with()

    def test_delete_o_auth_app_connection_raises_non_conflict_aws_errors(
        self, q_client, mock_q_client, mock_application_request
    ):
        error_response = {
            "Error": {"Code": "ValidationException", "Message": "invalid message"}
        }
        mock_q_client.delete_o_auth_app_connection.side_effect = ClientError(
            error_response, "delete_o_auth_app_connection"
        )

        with pytest.raises(AWSException):
            q_client.delete_o_auth_app_connection()

        mock_q_client.delete_o_auth_app_connection.assert_called_once()

    def test_send_message_success(self, q_client, mock_q_client, send_message_params):
        mock_q_client.send_message.return_value = {"Success": True}

        response = q_client.send_message(
            send_message_params["message"], send_message_params["history"]
        )

        mock_q_client.send_message.assert_called_once_with(**send_message_params)

        assert response == {"Success": True}

    def test_send_message_client_error(
        self, q_client, mock_q_client, send_message_params
    ):
        error_response = {
            "Error": {"Code": "InternalServerError", "Message": "Internal error"}
        }

        mock_q_client.send_message.side_effect = ClientError(
            error_response, "send_message"
        )

        with pytest.raises(AWSException):
            q_client.send_message(
                send_message_params["message"], send_message_params["history"]
            )

        mock_q_client.send_message.assert_called_once_with(**send_message_params)

    def test_send_message_access_denied(
        self, q_client, mock_q_client, send_message_params
    ):
        mock_q_client.send_message.side_effect = AccessDeniedException()

        with pytest.raises(AWSException):
            q_client.send_message(
                send_message_params["message"], send_message_params["history"]
            )

        mock_q_client.send_message.assert_called_once_with(**send_message_params)

    @pytest.mark.parametrize(
        "client_error,expected_result,expected_exception",
        [
            # Happy path - successful verification
            (
                None,
                {
                    "response": {
                        "GITLAB_INSTANCE_REACHABILITY": {"status": "PASSED"},
                        "GITLAB_CREDENTIAL_VALIDITY": {"status": "PASSED"},
                    }
                },
                None,
            ),
            # Test AccessDeniedException with retry
            (
                ClientError(
                    {
                        "Error": {
                            "Code": "AccessDeniedException",
                            "Message": "Access Denied",
                        }
                    },
                    "verify_o_auth_app_connection",
                ),
                {"Status": "Retried"},
                None,
            ),
            # Test other ClientError
            (
                ClientError(
                    {
                        "Error": {
                            "Code": "InternalServerError",
                            "Message": "Internal Server Error",
                        }
                    },
                    "verify_o_auth_app_connection",
                ),
                None,
                AWSException,
            ),
        ],
    )
    def test_verify_oauth_connection(
        self, q_client, mock_q_client, client_error, expected_result, expected_exception
    ):
        """Tests OAuth connection verification with various scenarios."""
        # Setup mock request
        mock_health_request = Mock()
        mock_health_request.code = "test_code"

        # Setup retry mock
        q_client._retry_verify_oauth_connection = Mock(
            return_value={"Status": "Retried"}
        )

        if client_error:
            # Configure mock to raise exception
            q_client._verify_oauth_connection = Mock(side_effect=client_error)
        else:
            # Configure mock to return successfully
            q_client._verify_oauth_connection = Mock(return_value=expected_result)

        if expected_exception:
            with pytest.raises(expected_exception):
                q_client.verify_oauth_connection(mock_health_request)
        else:
            result = q_client.verify_oauth_connection(mock_health_request)

            if (
                client_error
                and client_error.response["Error"]["Code"] == "AccessDeniedException"
            ):
                # Verify retry was called with correct parameters
                q_client._retry_verify_oauth_connection.assert_called_once_with(
                    client_error, mock_health_request.code
                )
                assert result == {"Status": "Retried"}
            else:
                # Verify normal verification was successful
                assert result == expected_result
