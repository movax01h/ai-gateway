GET_VULNERABILITY_DETAILS_QUERY = """
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
            location {
                __typename
                ... on VulnerabilityLocationClusterImageScanning {
                    image
                    kubernetesResource {
                        agent {
                            id
                            name
                            webPath
                        }
                    }
                }
                ... on VulnerabilityLocationContainerScanning {
                    image
                    containerRepositoryUrl
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
