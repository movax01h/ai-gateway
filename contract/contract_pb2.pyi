from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ClientEvent(_message.Message):
    __slots__ = ("startRequest", "actionResponse")
    STARTREQUEST_FIELD_NUMBER: _ClassVar[int]
    ACTIONRESPONSE_FIELD_NUMBER: _ClassVar[int]
    startRequest: StartWorkflowRequest
    actionResponse: ActionResponse
    def __init__(self, startRequest: _Optional[_Union[StartWorkflowRequest, _Mapping]] = ..., actionResponse: _Optional[_Union[ActionResponse, _Mapping]] = ...) -> None: ...

class StartWorkflowRequest(_message.Message):
    __slots__ = ("clientVersion", "workflowID", "workflowDefinition", "goal", "workflowMetadata", "clientCapabilities", "mcpTools", "additional_context", "approval")
    CLIENTVERSION_FIELD_NUMBER: _ClassVar[int]
    WORKFLOWID_FIELD_NUMBER: _ClassVar[int]
    WORKFLOWDEFINITION_FIELD_NUMBER: _ClassVar[int]
    GOAL_FIELD_NUMBER: _ClassVar[int]
    WORKFLOWMETADATA_FIELD_NUMBER: _ClassVar[int]
    CLIENTCAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    MCPTOOLS_FIELD_NUMBER: _ClassVar[int]
    ADDITIONAL_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    APPROVAL_FIELD_NUMBER: _ClassVar[int]
    clientVersion: str
    workflowID: str
    workflowDefinition: str
    goal: str
    workflowMetadata: str
    clientCapabilities: _containers.RepeatedScalarFieldContainer[str]
    mcpTools: _containers.RepeatedCompositeFieldContainer[McpTool]
    additional_context: _containers.RepeatedCompositeFieldContainer[AdditionalContext]
    approval: Approval
    def __init__(self, clientVersion: _Optional[str] = ..., workflowID: _Optional[str] = ..., workflowDefinition: _Optional[str] = ..., goal: _Optional[str] = ..., workflowMetadata: _Optional[str] = ..., clientCapabilities: _Optional[_Iterable[str]] = ..., mcpTools: _Optional[_Iterable[_Union[McpTool, _Mapping]]] = ..., additional_context: _Optional[_Iterable[_Union[AdditionalContext, _Mapping]]] = ..., approval: _Optional[_Union[Approval, _Mapping]] = ...) -> None: ...

class ActionResponse(_message.Message):
    __slots__ = ("requestID", "response", "plainTextResponse", "httpResponse")
    REQUESTID_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    PLAINTEXTRESPONSE_FIELD_NUMBER: _ClassVar[int]
    HTTPRESPONSE_FIELD_NUMBER: _ClassVar[int]
    requestID: str
    response: str
    plainTextResponse: PlainTextResponse
    httpResponse: HttpResponse
    def __init__(self, requestID: _Optional[str] = ..., response: _Optional[str] = ..., plainTextResponse: _Optional[_Union[PlainTextResponse, _Mapping]] = ..., httpResponse: _Optional[_Union[HttpResponse, _Mapping]] = ...) -> None: ...

class PlainTextResponse(_message.Message):
    __slots__ = ("response", "error")
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    response: str
    error: str
    def __init__(self, response: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

class HttpResponse(_message.Message):
    __slots__ = ("headers", "statusCode", "body", "error")
    class HeadersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    HEADERS_FIELD_NUMBER: _ClassVar[int]
    STATUSCODE_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    headers: _containers.ScalarMap[str, str]
    statusCode: int
    body: str
    error: str
    def __init__(self, headers: _Optional[_Mapping[str, str]] = ..., statusCode: _Optional[int] = ..., body: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

class Action(_message.Message):
    __slots__ = ("requestID", "runCommand", "runHTTPRequest", "runReadFile", "runWriteFile", "runGitCommand", "runEditFile", "newCheckpoint", "listDirectory", "grep", "findFiles", "runMCPTool", "mkdir")
    REQUESTID_FIELD_NUMBER: _ClassVar[int]
    RUNCOMMAND_FIELD_NUMBER: _ClassVar[int]
    RUNHTTPREQUEST_FIELD_NUMBER: _ClassVar[int]
    RUNREADFILE_FIELD_NUMBER: _ClassVar[int]
    RUNWRITEFILE_FIELD_NUMBER: _ClassVar[int]
    RUNGITCOMMAND_FIELD_NUMBER: _ClassVar[int]
    RUNEDITFILE_FIELD_NUMBER: _ClassVar[int]
    NEWCHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    LISTDIRECTORY_FIELD_NUMBER: _ClassVar[int]
    GREP_FIELD_NUMBER: _ClassVar[int]
    FINDFILES_FIELD_NUMBER: _ClassVar[int]
    RUNMCPTOOL_FIELD_NUMBER: _ClassVar[int]
    MKDIR_FIELD_NUMBER: _ClassVar[int]
    requestID: str
    runCommand: RunCommandAction
    runHTTPRequest: RunHTTPRequest
    runReadFile: ReadFile
    runWriteFile: WriteFile
    runGitCommand: RunGitCommand
    runEditFile: EditFile
    newCheckpoint: NewCheckpoint
    listDirectory: ListDirectory
    grep: Grep
    findFiles: FindFiles
    runMCPTool: RunMCPTool
    mkdir: Mkdir
    def __init__(self, requestID: _Optional[str] = ..., runCommand: _Optional[_Union[RunCommandAction, _Mapping]] = ..., runHTTPRequest: _Optional[_Union[RunHTTPRequest, _Mapping]] = ..., runReadFile: _Optional[_Union[ReadFile, _Mapping]] = ..., runWriteFile: _Optional[_Union[WriteFile, _Mapping]] = ..., runGitCommand: _Optional[_Union[RunGitCommand, _Mapping]] = ..., runEditFile: _Optional[_Union[EditFile, _Mapping]] = ..., newCheckpoint: _Optional[_Union[NewCheckpoint, _Mapping]] = ..., listDirectory: _Optional[_Union[ListDirectory, _Mapping]] = ..., grep: _Optional[_Union[Grep, _Mapping]] = ..., findFiles: _Optional[_Union[FindFiles, _Mapping]] = ..., runMCPTool: _Optional[_Union[RunMCPTool, _Mapping]] = ..., mkdir: _Optional[_Union[Mkdir, _Mapping]] = ...) -> None: ...

class RunCommandAction(_message.Message):
    __slots__ = ("program", "arguments", "flags")
    PROGRAM_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    program: str
    arguments: _containers.RepeatedScalarFieldContainer[str]
    flags: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, program: _Optional[str] = ..., arguments: _Optional[_Iterable[str]] = ..., flags: _Optional[_Iterable[str]] = ...) -> None: ...

class ReadFile(_message.Message):
    __slots__ = ("filepath",)
    FILEPATH_FIELD_NUMBER: _ClassVar[int]
    filepath: str
    def __init__(self, filepath: _Optional[str] = ...) -> None: ...

class WriteFile(_message.Message):
    __slots__ = ("filepath", "contents")
    FILEPATH_FIELD_NUMBER: _ClassVar[int]
    CONTENTS_FIELD_NUMBER: _ClassVar[int]
    filepath: str
    contents: str
    def __init__(self, filepath: _Optional[str] = ..., contents: _Optional[str] = ...) -> None: ...

class EditFile(_message.Message):
    __slots__ = ("filepath", "oldString", "newString")
    FILEPATH_FIELD_NUMBER: _ClassVar[int]
    OLDSTRING_FIELD_NUMBER: _ClassVar[int]
    NEWSTRING_FIELD_NUMBER: _ClassVar[int]
    filepath: str
    oldString: str
    newString: str
    def __init__(self, filepath: _Optional[str] = ..., oldString: _Optional[str] = ..., newString: _Optional[str] = ...) -> None: ...

class RunHTTPRequest(_message.Message):
    __slots__ = ("method", "path", "body")
    METHOD_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    method: str
    path: str
    body: str
    def __init__(self, method: _Optional[str] = ..., path: _Optional[str] = ..., body: _Optional[str] = ...) -> None: ...

class RunGitCommand(_message.Message):
    __slots__ = ("command", "arguments", "repository_url")
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    REPOSITORY_URL_FIELD_NUMBER: _ClassVar[int]
    command: str
    arguments: str
    repository_url: str
    def __init__(self, command: _Optional[str] = ..., arguments: _Optional[str] = ..., repository_url: _Optional[str] = ...) -> None: ...

class GenerateTokenRequest(_message.Message):
    __slots__ = ("workflowDefinition",)
    WORKFLOWDEFINITION_FIELD_NUMBER: _ClassVar[int]
    workflowDefinition: str
    def __init__(self, workflowDefinition: _Optional[str] = ...) -> None: ...

class GenerateTokenResponse(_message.Message):
    __slots__ = ("token", "expiresAt")
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    EXPIRESAT_FIELD_NUMBER: _ClassVar[int]
    token: str
    expiresAt: int
    def __init__(self, token: _Optional[str] = ..., expiresAt: _Optional[int] = ...) -> None: ...

class NewCheckpoint(_message.Message):
    __slots__ = ("status", "checkpoint", "goal", "errors")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    GOAL_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    status: str
    checkpoint: str
    goal: str
    errors: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, status: _Optional[str] = ..., checkpoint: _Optional[str] = ..., goal: _Optional[str] = ..., errors: _Optional[_Iterable[str]] = ...) -> None: ...

class ListDirectory(_message.Message):
    __slots__ = ("directory",)
    DIRECTORY_FIELD_NUMBER: _ClassVar[int]
    directory: str
    def __init__(self, directory: _Optional[str] = ...) -> None: ...

class Grep(_message.Message):
    __slots__ = ("search_directory", "pattern", "case_insensitive")
    SEARCH_DIRECTORY_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    CASE_INSENSITIVE_FIELD_NUMBER: _ClassVar[int]
    search_directory: str
    pattern: str
    case_insensitive: bool
    def __init__(self, search_directory: _Optional[str] = ..., pattern: _Optional[str] = ..., case_insensitive: bool = ...) -> None: ...

class FindFiles(_message.Message):
    __slots__ = ("name_pattern",)
    NAME_PATTERN_FIELD_NUMBER: _ClassVar[int]
    name_pattern: str
    def __init__(self, name_pattern: _Optional[str] = ...) -> None: ...

class McpTool(_message.Message):
    __slots__ = ("name", "description", "inputSchema")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    INPUTSCHEMA_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    inputSchema: str
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., inputSchema: _Optional[str] = ...) -> None: ...

class RunMCPTool(_message.Message):
    __slots__ = ("name", "args")
    NAME_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    name: str
    args: str
    def __init__(self, name: _Optional[str] = ..., args: _Optional[str] = ...) -> None: ...

class AdditionalContext(_message.Message):
    __slots__ = ("category", "id", "content", "metadata")
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    category: str
    id: str
    content: str
    metadata: str
    def __init__(self, category: _Optional[str] = ..., id: _Optional[str] = ..., content: _Optional[str] = ..., metadata: _Optional[str] = ...) -> None: ...

class Approval(_message.Message):
    __slots__ = ("approval", "rejection")
    class Approved(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class Rejected(_message.Message):
        __slots__ = ("message",)
        MESSAGE_FIELD_NUMBER: _ClassVar[int]
        message: str
        def __init__(self, message: _Optional[str] = ...) -> None: ...
    APPROVAL_FIELD_NUMBER: _ClassVar[int]
    REJECTION_FIELD_NUMBER: _ClassVar[int]
    approval: Approval.Approved
    rejection: Approval.Rejected
    def __init__(self, approval: _Optional[_Union[Approval.Approved, _Mapping]] = ..., rejection: _Optional[_Union[Approval.Rejected, _Mapping]] = ...) -> None: ...

class Mkdir(_message.Message):
    __slots__ = ("directory_path",)
    DIRECTORY_PATH_FIELD_NUMBER: _ClassVar[int]
    directory_path: str
    def __init__(self, directory_path: _Optional[str] = ...) -> None: ...
