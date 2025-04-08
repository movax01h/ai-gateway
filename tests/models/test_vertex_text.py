from typing import Sequence, Type, Union
from unittest.mock import AsyncMock, Mock, patch

import pytest
from google.api_core.exceptions import (
    DuplicateCredentialArgs,
    GoogleAPICallError,
    GoogleAPIError,
    InternalServerError,
    InvalidArgument,
    PermissionDenied,
    RetryError,
)
from google.cloud.aiplatform.gapic import PredictionServiceAsyncClient, PredictResponse
from google.protobuf import json_format, struct_pb2
from google.rpc.error_details_pb2 import ErrorInfo

from ai_gateway.models.base_text import TextGenModelOutput
from ai_gateway.models.vertex_text import (
    CodeBisonModelInput,
    CodeGeckoModelInput,
    KindVertexTextModel,
    PalmCodeBisonModel,
    PalmCodeGeckoModel,
    PalmTextBisonModel,
    TextBisonModelInput,
    VertexAPIConnectionError,
    VertexAPIStatusError,
)
from ai_gateway.safety_attributes import SafetyAttributes


class TestVertexAPIConnectionError:
    @pytest.mark.parametrize(
        ("original_error", "expected_error_string"),
        [
            (
                RetryError(message="retrying", cause=RuntimeError),
                "Vertex Model API error: RetryError retrying",
            ),
            (
                DuplicateCredentialArgs(),
                "Vertex Model API error: DuplicateCredentialArgs",
            ),
        ],
    )
    def test_from_exception(
        self, original_error: GoogleAPIError, expected_error_string: str
    ):
        wrapped_error = VertexAPIConnectionError.from_exception(original_error)

        assert str(wrapped_error) == expected_error_string


class TestVertexAPIStatusError:
    @pytest.mark.parametrize(
        ("original_error", "expected_error_string"),
        [
            (
                PermissionDenied(
                    message="Permission denied on resource project abc",
                    details=[
                        ErrorInfo(
                            reason="CONSUMER_INVALID",
                            metadata={
                                "consumer": "projects/uknown-project-id",
                                "service": "aiplatform.googleapis.com",
                            },
                        )
                    ],
                ),
                "403 Vertex Model API error: PermissionDenied Permission denied on resource project abc "
                '[reason: "CONSUMER_INVALID"\n'
                'metadata {\n  key: "service"\n  value: "aiplatform.googleapis.com"\n}\n'
                'metadata {\n  key: "consumer"\n  value: "projects/uknown-project-id"\n}\n]',
            ),
            (
                InternalServerError(message="Something went wrong"),
                "500 Vertex Model API error: InternalServerError Something went wrong",
            ),
        ],
    )
    def test_from_exception(
        self, original_error: GoogleAPICallError, expected_error_string: str
    ):
        wrapped_error = VertexAPIStatusError.from_exception(original_error)

        assert str(wrapped_error) == expected_error_string


TEST_PREFIX = "random prompt"
TEST_SUFFIX = "some suffix"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,prefix,suffix,expected_output,expected_generate_args",
    [
        (
            PalmTextBisonModel,
            TEST_PREFIX,
            TEST_SUFFIX,
            "some output",
            [TextBisonModelInput(TEST_PREFIX), 0.2, 32, 0.95, 40, 1, None],
        ),
        (
            PalmTextBisonModel,
            "",
            TEST_SUFFIX,
            "",
            [TextBisonModelInput(""), 0.2, 32, 0.95, 40, 1, None],
        ),
        (
            PalmCodeBisonModel,
            TEST_PREFIX,
            TEST_SUFFIX,
            "some output",
            [CodeBisonModelInput(TEST_PREFIX), 0.2, 2048, 0.95, 40, 1, None],
        ),
        (
            PalmCodeBisonModel,
            "",
            TEST_SUFFIX,
            "",
            [CodeBisonModelInput(""), 0.2, 2048, 0.95, 40, 1, None],
        ),
        (
            PalmCodeGeckoModel,
            TEST_PREFIX,
            TEST_SUFFIX,
            "some output",
            [
                CodeGeckoModelInput(TEST_PREFIX, TEST_SUFFIX),
                0.2,
                64,
                0.95,
                40,
                1,
                ["\n\n"],
                None,
            ],
        ),
        (
            PalmCodeGeckoModel,
            "",
            TEST_SUFFIX,
            "",
            [
                CodeGeckoModelInput("", TEST_SUFFIX),
                0.2,
                64,
                0.95,
                40,
                1,
                ["\n\n"],
                None,
            ],
        ),
    ],
)
async def test_palm_model_generate(
    model,
    prefix,
    suffix,
    expected_output,
    expected_generate_args,
):
    palm_model = model(Mock(), "test", "some location")
    palm_model._generate = AsyncMock(
        side_effect=lambda *_: TextGenModelOutput(
            text=expected_output, score=0, safety_attributes=SafetyAttributes()
        )
    )

    result = await palm_model.generate(prefix, suffix)

    assert result == TextGenModelOutput(
        text=expected_output, score=0, safety_attributes=SafetyAttributes()
    )

    palm_model._generate.assert_called_with(*expected_generate_args)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "model_name"),
    [
        (PalmCodeGeckoModel, KindVertexTextModel.CODE_GECKO_002),
        (PalmCodeBisonModel, KindVertexTextModel.CODE_BISON_002),
        (PalmTextBisonModel, KindVertexTextModel.TEXT_BISON_002),
    ],
)
async def test_palm_model_generate_instrumented(
    model: Type[Union[PalmTextBisonModel, PalmCodeBisonModel, PalmCodeGeckoModel]],
    model_name: KindVertexTextModel,
):
    mock_client = Mock()
    mock_client.predict = AsyncMock(return_value=PredictResponse())
    palm_model = model.from_model_name(model_name, mock_client, "test", "some location")
    with patch(
        "ai_gateway.instrumentators.model_requests.ModelRequestInstrumentator.watch"
    ) as mock_watch:
        await palm_model.generate(TEST_PREFIX, TEST_SUFFIX)
        mock_watch.assert_called()


@pytest.mark.parametrize(
    "model_input,is_valid,output_dict",
    [
        (CodeBisonModelInput(TEST_PREFIX), True, {"prefix": TEST_PREFIX}),
        (CodeBisonModelInput(""), False, None),
        (TextBisonModelInput(TEST_PREFIX), True, {"content": TEST_PREFIX}),
        (TextBisonModelInput(""), False, None),
        (
            CodeGeckoModelInput(TEST_PREFIX, TEST_SUFFIX),
            True,
            {"prefix": TEST_PREFIX, "suffix": TEST_SUFFIX},
        ),
        (CodeGeckoModelInput("", ""), False, None),
        (CodeGeckoModelInput("", TEST_SUFFIX), False, None),
    ],
)
def test_palm_model_inputs(model_input, is_valid, output_dict):
    assert model_input.is_valid() is is_valid
    assert output_dict is None or model_input.dict() == output_dict


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,client_exception,expected_exception",
    [
        (
            PalmCodeGeckoModel(
                Mock(spec=PredictionServiceAsyncClient),
                "random_project",
                "random_location",
            ),
            RetryError("Retry.", "Lost connection."),
            VertexAPIConnectionError,
        ),
        (
            PalmCodeGeckoModel(
                Mock(spec=PredictionServiceAsyncClient),
                "random_project",
                "random_location",
            ),
            InvalidArgument("Bad argument."),
            VertexAPIStatusError,
        ),
    ],
)
async def test_palm_model_api_error(model, client_exception, expected_exception):
    def _client_predict(*args, **kwargs):
        raise client_exception

    model.client.predict = AsyncMock(side_effect=_client_predict)

    with pytest.raises(expected_exception):
        result = await model.generate("random_prefix", "random_suffix")
        assert result == TextGenModelOutput(
            text="", score=0, safety_attributes=SafetyAttributes()
        )


@pytest.mark.parametrize(
    ("model", "model_name", "expected_metadata_name"),
    [
        (
            PalmTextBisonModel,
            KindVertexTextModel.TEXT_BISON_002,
            KindVertexTextModel.TEXT_BISON_002.value,
        ),
        (
            PalmCodeBisonModel,
            KindVertexTextModel.CODE_BISON_002,
            KindVertexTextModel.CODE_BISON_002.value,
        ),
        (
            PalmCodeGeckoModel,
            KindVertexTextModel.CODE_GECKO_002,
            KindVertexTextModel.CODE_GECKO_002.value,
        ),
    ],
)
def test_palm_model_from_name(
    model: Type[Union[PalmTextBisonModel, PalmCodeBisonModel, PalmCodeGeckoModel]],
    model_name: KindVertexTextModel,
    expected_metadata_name: str,
):
    model = model.from_model_name(model_name, Mock(), "project", "location")

    # mypy [attr-defined] exclusion due to a known issue https://github.com/python/mypy/issues/8085
    assert isinstance(model.metadata.name, str)  # type: ignore[attr-defined]

    assert model.metadata.name == expected_metadata_name  # type: ignore[attr-defined]
    assert model.metadata.engine == "vertex-ai"  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "stop_sequences", "expected_stop_sequences"),
    [
        (
            PalmTextBisonModel,
            None,
            None,
        ),
        (
            PalmCodeBisonModel,
            None,
            None,
        ),
        (
            PalmCodeGeckoModel,
            None,
            ["\n\n"],  # we set this sequence by default
        ),
        (
            PalmCodeGeckoModel,
            ["\n\n"],
            ["\n\n"],
        ),
        (
            PalmCodeGeckoModel,
            ["random stop sequence"],
            ["random stop sequence"],
        ),
    ],
)
async def test_palm_model_stop_sequences(
    model: Type[Union[PalmTextBisonModel, PalmCodeBisonModel, PalmCodeGeckoModel]],
    stop_sequences: Sequence[str],
    expected_stop_sequences: Sequence[str],
):
    client = Mock()
    client.predict = AsyncMock(return_value=PredictResponse())
    palm_model = model(client, "test", "some location")

    await palm_model.generate("foo", "", stop_sequences=stop_sequences)

    client.predict.assert_called_once()

    parameters = client.predict.call_args[1]["parameters"]
    params_dict = json_format.MessageToDict(parameters)
    assert params_dict.get("stopSequences", None) == expected_stop_sequences


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "prediction", "expected_safety_attributes"),
    [
        (
            PalmTextBisonModel,
            {
                "safetyAttributes": {
                    "categories": ["Violent"],
                    "blocked": True,
                    "scores": [1.0],
                },
                "content": "",
            },
            SafetyAttributes(categories=["Violent"], blocked=True),
        ),
        (
            PalmCodeBisonModel,
            {
                "safetyAttributes": {
                    "categories": ["Violent"],
                    "blocked": True,
                    "scores": [1.0],
                },
                "content": "",
            },
            SafetyAttributes(categories=["Violent"], blocked=True),
        ),
        (
            PalmCodeGeckoModel,
            {
                "safetyAttributes": {
                    "categories": ["Violent"],
                    "blocked": True,
                    "scores": [1.0],
                },
                "content": "",
            },
            SafetyAttributes(categories=["Violent"], blocked=True),
        ),
        (
            PalmCodeGeckoModel,
            {
                "safetyAttributes": {
                    "errors": [234],
                    "blocked": True,
                },
                "content": "",
            },
            SafetyAttributes(errors=[234], blocked=True),
        ),
        (
            PalmCodeGeckoModel,
            {
                "safetyAttributes": {
                    "categories": [],
                    "blocked": False,
                    "scores": [],
                },
                "content": "def awesome_func",
            },
            SafetyAttributes(categories=[], blocked=False),
        ),
        (
            PalmCodeGeckoModel,
            {
                "content": "def awesome_func",
            },
            SafetyAttributes(categories=[], blocked=False),
        ),
    ],
)
async def test_palm_model_safety_attributes(
    model: Type[Union[PalmTextBisonModel, PalmCodeBisonModel, PalmCodeGeckoModel]],
    prediction: dict,
    expected_safety_attributes: SafetyAttributes,
):
    client = Mock()
    predict_response = PredictResponse()
    prediction_value = json_format.ParseDict(prediction, struct_pb2.Value())
    predict_response.predictions.append(prediction_value)

    client.predict = AsyncMock(return_value=predict_response)
    palm_model = model(client, "test", "some location")

    model_output = await palm_model.generate("# bomberman", "")

    assert isinstance(model_output, list)
    assert isinstance(model_output[0], TextGenModelOutput)
    assert model_output[0].safety_attributes == expected_safety_attributes
