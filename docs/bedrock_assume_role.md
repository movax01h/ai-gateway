# Amazon Bedrock: STS AssumeRole

AI Gateway can call Amazon Bedrock models using temporary credentials from
`sts:AssumeRole` for a role named in the request.

## Request model metadata

```json
{
  "provider": "gitlab",
  "name": "claude_sonnet_4_6_bedrock",
  "iam_role": "arn:aws:iam::123456789012:role/my-bedrock-role",
}
```

`iam_role` is optional and only applies to Bedrock models (`bedrock` and
`bedrock_converse` LiteLLM providers). It must be an IAM role ARN
(`arn:<partition>:iam::<12-digit account>:role/<path/name>`); a malformed value
is rejected when the metadata is parsed.

The gateway sets the `ExternalId` of the `sts:AssumeRole` call itself, from the instance
identity in the verified JWT, in this order:

1. the `gitlab_instance_uid` claim: assigned by CustomersDot, and copied into the
   tokens this gateway mints for direct access;
1. the `sub` claim of a token the GitLab instance signed itself (self-hosted
   models), which is the instance UUID. A token this gateway issued is never read
   this way, because its `sub` is a user.

The `gitlab_instance_id` claim is not used: CustomersDot documents it as reported by
the instance, so it can be spoofed. Request headers are not used either, as they are
not signed. The request body cannot choose the ID: an `iam_role_external_id` field
in `model_metadata` is ignored. A request with `iam_role` but no verified instance
ID is rejected rather than assumed without an ExternalId.

The ExternalId is not a secret. It protects against the confused deputy problem:
configure the role's trust policy to require it, so that another tenant of the
same AI Gateway cannot make it assume your role by guessing the ARN.

```json
{
  "Effect": "Allow",
  "Principal": { "AWS": "arn:aws:iam::<gateway-account>:role/<gateway-role>" },
  "Action": "sts:AssumeRole",
  "Condition": {
    "StringEquals": { "sts:ExternalId": "<GitLab instance ID>" }
  }
}
```

## GitLab.com is out of scope

Request-supplied `iam_role` is supported for self-managed and GitLab Dedicated gateways,
where one gateway serves one GitLab instance. It is not supported on GitLab.com, and that
gap is intentional.

The ExternalId identifies a GitLab *instance*. On GitLab.com the instance is shared by
every customer, so every customer's request would carry the same ExternalId. A role's
trust policy could then only tell GitLab.com from anyone else, not one customer from
another, and the confused deputy protection described above would not hold: any
GitLab.com customer who learns another customer's role ARN could make the gateway assume
it.

Closing the gap needs a per-customer identity in the ExternalId, such as the root
namespace ID that SaaS tokens carry in `gitlab_root_namespace_id`. That is not
implemented, and nothing here derives an ExternalId from it.

The gateway enforces this: a request with `iam_role` whose token has the `saas` realm
is rejected with `UnsupportedRealm` and nothing is assumed.

## Errors

A request with `iam_role` is rejected with HTTP 422 when custom models are not enabled
on the gateway (`AIGW_CUSTOM_MODELS__ENABLED`), so IAM roles are never assumed on the
GitLab-run cloud gateway.

If `sts:AssumeRole` fails (for example `AccessDenied`) or no source credentials
are available, the request fails with HTTP 422 and does not fall back to the
default model or credentials. `detail` reads like
`Failed to assume the requested IAM role: AccessDenied`. The role ARN is not
returned; it and the STS error code are logged as a warning. No credentials are
included.

## Source credentials

The identity that calls `sts:AssumeRole` is whatever boto3's default credential
chain resolves. AI Gateway sets none of this up itself; configure one of the
following and the role named in `iam_role` is assumed on top of it. Its identity
must be allowed to call `sts:AssumeRole` on the target, and the target's trust
policy must trust it.

- **Environment keys:** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and
  optionally `AWS_SESSION_TOKEN`.
- **Shared config or profile:** `AWS_PROFILE` with `~/.aws/credentials` or
  `~/.aws/config`.
- **Web identity (for example EKS IRSA):** `AWS_ROLE_ARN` and
  `AWS_WEB_IDENTITY_TOKEN_FILE`, which boto3 reads itself.
- **Container or instance role:** ECS task roles and EC2 instance profiles.

If no source credentials are available, or the assume call is denied, the request
fails with the error described above. AI Gateway does not fall back to other
credentials. A session name is not configurable; the call uses `gitlab-ai-gateway`.

A request cannot carry both `iam_role` and `api_key`, and `iam_role` is rejected
for models that are not served by Bedrock. Metadata that breaks these rules is not
applied.

## Caching and refresh

Credentials are cached per `(role, session name, external ID)` using botocore's
`AssumeRoleCredentialFetcher` with refreshable credentials, so STS is called once
per role and again only when the credentials near expiry. The cache holds at
most 256 roles. Credentials are resolved when the request's model parameters are
built, and a refresh is a blocking call to STS.
