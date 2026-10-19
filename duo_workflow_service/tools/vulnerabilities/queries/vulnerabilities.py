_TRACKED_REF_SELECTION = """            trackedRef {
                name
                refType
            }"""

# ``trackedRef`` is omitted when unsupported because unknown GraphQL fields fail
# the entire request.
_GET_VULNERABILITY_DETAILS_QUERY_TEMPLATE = """
    fragment Url on VulnerabilityDetailUrl {
        type: __typename
        name
        href
    }

    fragment Diff on VulnerabilityDetailDiff {
        type: __typename
        name
        before
        after
    }

    fragment Code on VulnerabilityDetailCode {
        type: __typename
        name
        value
    }

    fragment FileLocation on VulnerabilityDetailFileLocation {
        type: __typename
        name
        fileName
        lineStart
        lineEnd
    }

    fragment ModuleLocation on VulnerabilityDetailModuleLocation {
        type: __typename
        name
        moduleName
        offset
    }

    fragment Commit on VulnerabilityDetailCommit {
        type: __typename
        name
        value
    }

    fragment Text on VulnerabilityDetailText {
        type: __typename
        name
        value
    }

    fragment Markdown on VulnerabilityDetailMarkdown {
        type: __typename
        name
        value
    }

    fragment Boolean on VulnerabilityDetailBoolean {
        type: __typename
        name
        value
    }

    fragment Int on VulnerabilityDetailInt {
        type: __typename
        name
        value
    }

    fragment NonNestedReportTypes on VulnerabilityDetail {
        ...FileLocation
        ...Url
        ...Diff
        ...Code
        ...Commit
        ...Markdown
        ...Text
        ...Int
        ...Boolean
        ...ModuleLocation
    }

    fragment ListFields on VulnerabilityDetailList {
        type: __typename
        name
    }

    fragment List on VulnerabilityDetailList {
        ...ListFields
        items {
            ...NonNestedReportTypes
            ... on VulnerabilityDetailList {
                ...ListFields
                items {
                    ...NonNestedReportTypes
                }
            }
        }
    }

    fragment NamedList on VulnerabilityDetailNamedList {
        type: __typename
        name
        items {
            name
            fieldName
            value {
                ...NonNestedReportTypes
                ...Table
                ... on VulnerabilityDetailList {
                    ...ListFields
                    items {
                        ...NonNestedReportTypes
                    }
                }
            }
        }
    }

    fragment CodeFlowNode on VulnerabilityDetailCodeFlowNode {
        type: __typename
        nodeType
        fileLocation {
            ...FileLocation
        }
    }

    fragment CodeFlows on VulnerabilityDetailCodeFlows {
        type: __typename
        name
        items {
            ... {
                ...CodeFlowNode
            }
        }
    }

    fragment TableFields on VulnerabilityDetailTable {
        type: __typename
        name
        headers {
            ...NonNestedReportTypes
        }
        rows {
            row {
                ...NonNestedReportTypes
            }
        }
    }

    fragment Table on VulnerabilityDetailTable {
        type: __typename
        name
        headers {
            ...NonNestedReportTypes
        }
        rows {
            row {
                ...NonNestedReportTypes
                ...TableFields
            }
        }
    }

    query GetVulnerabilityDetails($vulnerabilityId: VulnerabilityID!) {
        vulnerability(id: $vulnerabilityId) {
            id
            project {
                id
                fullPath
            }
            title
            state
            webUrl
            identifiers {
                name
            }
            description
            reportType
            reachability
            cveEnrichment {
                cve
                epssScore
                isKnownExploit
            }
            detectedAt
            dismissedAt
            initialDetectedPipeline {
                id
                name
                createdAt
            }
            latestFlag {
                id
                status
                confidenceScore
                origin
                description
                createdAt
                updatedAt
            }
__TRACKED_REF_FRAGMENT__
            location {
                __typename
                ... on VulnerabilityLocationClusterImageScanning {
                    image
                    operatingSystem
                    kubernetesResource {
                        agent {
                            id
                            name
                            webPath
                        }
                    }
                    dependency {
                        version
                        package {
                            name
                        }
                    }
                }
                ... on VulnerabilityLocationContainerScanning {
                    image
                    containerRepositoryUrl
                    dependency {
                        version
                        package {
                            name
                        }
                    }
                }
                ... on VulnerabilityLocationCoverageFuzzing {
                    blobPath
                    crashAddress
                    crashType
                    endLine
                    file
                    stacktraceSnippet
                    startLine
                    vulnerableClass
                    vulnerableMethod
                }
                ... on VulnerabilityLocationDast {
                    path
                }
                ... on VulnerabilityLocationDependencyScanning {
                    blobPath
                    file
                    dependency {
                        version
                        package {
                            name
                        }
                    }
                }
                ... on VulnerabilityLocationGeneric {
                    description
                }
                ... on VulnerabilityLocationSast {
                    blobPath
                    file
                    startLine
                }
                ... on VulnerabilityLocationSecretDetection {
                    blobPath
                    file
                    startLine
                }
            }
            details {
                __typename
                ...List
                ...Table
                ...NamedList
                ...CodeFlows
                ...NonNestedReportTypes
            }
        }
    }
    """

LIST_VULNERABILITIES_QUERY = """query($projectFullPath: ID!, $first: Int, $after: String, $severity: [VulnerabilitySeverity!], $reportType: [VulnerabilityReportType!]) {
    project(fullPath: $projectFullPath) {
        vulnerabilities(first: $first, after: $after, severity: $severity, reportType: $reportType) {
            pageInfo {
                hasNextPage
                endCursor
            }
            nodes {
                id
                title
                reportType
                severity
                state
                location{
                    ... on VulnerabilityLocationSast {
                        file
                        startLine
                    }
                    ... on VulnerabilityLocationDependencyScanning {
                        file
                        dependency {
                            package {
                                name
                            }
                            version
                        }
                    }
                    ... on VulnerabilityLocationContainerScanning {
                        image
                        operatingSystem
                        dependency {
                            package {
                                name
                            }
                            version
                        }
                    }
                    ... on VulnerabilityLocationSecretDetection {
                        file
                        startLine
                    }
                }
                latestFlag {
                    id
                    status
                    confidenceScore
                    origin
                    description
                    createdAt
                    updatedAt
                }
            }
        }
    }
}
"""

_TRACKED_REF_PLACEHOLDER = "\n__TRACKED_REF_FRAGMENT__"


def build_vulnerability_details_query(include_tracked_ref: bool) -> str:
    """Build the vulnerability details GraphQL query.

    Args:
        include_tracked_ref: Whether to select ``trackedRef { name refType }``.
            Only true when the target GitLab instance is >= 19.5, since older
            schemas reject the field and pre-19.5 backends return it as null for
            the AI workflow token.

    Returns:
        The GraphQL query string.
    """
    replacement = "\n" + _TRACKED_REF_SELECTION if include_tracked_ref else ""
    return _GET_VULNERABILITY_DETAILS_QUERY_TEMPLATE.replace(
        _TRACKED_REF_PLACEHOLDER, replacement
    )
