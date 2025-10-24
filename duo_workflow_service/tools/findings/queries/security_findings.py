# editorconfig-checker-disable
GET_SECURITY_FINDING_DETAILS_QUERY = """
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

query GetSecurityReportFinding($projectFullPath: ID!, $pipelineId: CiPipelineID!, $findingUuid: String!) {
  project(fullPath: $projectFullPath) {
    id
    webUrl
    nameWithNamespace
    pipeline(id: $pipelineId) {
      id
      iid
      sha
      ref
      status
      createdAt
      securityReportFinding(uuid: $findingUuid) {
        uuid
        title
        description
        descriptionHtml
        state
        severity
        solution
        solutionHtml
        reportType
        falsePositive
        dismissalReason
        aiResolutionEnabled
        aiResolutionAvailable
        remediations {
          diff
          summary
        }
        scanner {
          id
          name
        }
        assets {
          name
          url
        }
        evidence {
          summary
          request {
            body
            headers {
              name
              value
            }
            method
            url
          }
          response {
            body
            reasonPhrase
            statusCode
            headers {
              name
              value
            }
          }
          supportingMessages {
            name
            response {
              body
              reasonPhrase
              statusCode
              headers {
                name
                value
              }
            }
          }
          source {
            name
          }
        }
        location {
          ... on VulnerabilityLocationSast {
            startLine
            endLine
            file
            blobPath
          }
          ... on VulnerabilityLocationSecretDetection {
            startLine
            endLine
            file
            blobPath
          }
          ... on VulnerabilityLocationDependencyScanning {
            blobPath
            file
          }
          ... on VulnerabilityLocationContainerScanning {
            image
            operatingSystem
          }
          ... on VulnerabilityLocationCoverageFuzzing {
            startLine
            endLine
            file
            blobPath
            crashAddress
            crashType
            stacktraceSnippet
            vulnerableMethod
            vulnerableClass
          }
          ... on VulnerabilityLocationDast {
            hostname
            path
          }
        }
        links {
          name
          url
        }
        identifiers {
          name
          url
          externalType
          externalId
        }
        issueLinks {
          nodes {
            id
            linkType
            issue {
              id
              iid
              createdAt
              webUrl
            }
          }
        }
        details {
          __typename
          ...List
          ...Table
          ...NamedList
          ...NonNestedReportTypes
        }
        dismissedAt
        dismissedBy {
          id
          name
          username
          webUrl
        }
        stateComment
        vulnerability {
          id
          presentOnDefaultBranch
        }
        userPermissions {
          adminVulnerability
          createIssue
        }
      }
    }
  }
}
"""

SEARCH_RECENT_PIPELINES_QUERY = """
query SearchRecentPipelines($projectPath: ID!, $first: Int!) {
  project(fullPath: $projectPath) {
    pipelines(first: $first) {
      nodes {
        id
        iid
        sha
        ref
        status
        createdAt
        securityReportSummary {
          sast {
            vulnerabilitiesCount
          }
          containerScanning {
            vulnerabilitiesCount
          }
          dependencyScanning {
            vulnerabilitiesCount
          }
          apiFuzzing {
            vulnerabilitiesCount
          }
          coverageFuzzing {
            vulnerabilitiesCount
          }
          clusterImageScanning {
            vulnerabilitiesCount
          }
          secretDetection {
            vulnerabilitiesCount
          }
        }
      }
    }
  }
}
"""


LIST_SECURITY_FINDINGS_QUERY = """
query PipelineFindings(
    $fullPath: ID!
    $pipelineId: CiPipelineID!
    $first: Int
    $after: String
    $severity: [String!]
    $reportType: [String!]
    $scanner: [String!]
    $state: [VulnerabilityState!]
) {
    project(fullPath: $fullPath) {
        id
        pipeline(id: $pipelineId) {
            id
            iid
            sha
            ref
            status
            securityReportFindings(
                after: $after
                first: $first
                severity: $severity
                reportType: $reportType
                scanner: $scanner
                state: $state
            ) {
                nodes {
                    uuid
                    title
                    description
                    severity
                    state
                    reportType
                    dismissalReason
                    falsePositive
                    aiResolutionAvailable
                    aiResolutionEnabled

                    identifiers {
                        externalType
                        name
                    }

                    scanner {
                        id
                        name
                        vendor
                    }

                    location {
                        ... on VulnerabilityLocationSast {
                            file
                            startLine
                            endLine
                        }
                        ... on VulnerabilityLocationSecretDetection {
                            file
                            startLine
                        }
                        ... on VulnerabilityLocationDependencyScanning {
                            file
                        }
                        ... on VulnerabilityLocationContainerScanning {
                            image
                            operatingSystem
                        }
                    }

                    vulnerability {
                        id
                        state
                    }
                }
                pageInfo {
                    hasNextPage
                    endCursor
                }
            }
        }
    }
}
"""

# Alternative query that still uses IID if needed
LIST_SECURITY_FINDINGS_QUERY_WITH_IID = """
query PipelineFindings(
    $fullPath: ID!
    $pipelineIid: ID!
    $first: Int
    $after: String
    $severity: [String!]
    $reportType: [String!]
    $scanner: [String!]
    $state: [VulnerabilityState!]
) {
    project(fullPath: $fullPath) {
        id
        pipeline(iid: $pipelineIid) {
            id
            iid
            sha
            ref
            status
            securityReportFindings(
                after: $after
                first: $first
                severity: $severity
                reportType: $reportType
                scanner: $scanner
                state: $state
            ) {
                nodes {
                    uuid
                    title
                    description
                    severity
                    state
                    reportType
                    dismissalReason
                    falsePositive
                    aiResolutionAvailable
                    aiResolutionEnabled

                    identifiers {
                        externalType
                        name
                    }

                    scanner {
                        id
                        name
                        vendor
                    }

                    location {
                        ... on VulnerabilityLocationSast {
                            file
                            startLine
                            endLine
                        }
                        ... on VulnerabilityLocationSecretDetection {
                            file
                            startLine
                        }
                        ... on VulnerabilityLocationDependencyScanning {
                            file
                        }
                        ... on VulnerabilityLocationContainerScanning {
                            image
                            operatingSystem
                        }
                    }

                    vulnerability {
                        id
                        state
                    }
                }
                pageInfo {
                    hasNextPage
                    endCursor
                }
            }
        }
    }
}
"""
# editorconfig-checker-enable
