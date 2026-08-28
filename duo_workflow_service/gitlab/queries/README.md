# GitLab GraphQL queries

The `versioned/` directory holds GraphQL queries DWS sends to GitLab, one file per milestone
(e.g. `19_3_0.graphql`).

## Adding a field

Add the field to the newest file, tagged with `@gl_introduced(version: "X.Y.0")`. Do not
create a new versioned file for an additive change.

```graphql
currentThread @gl_introduced(version: "19.4.0")
```

Client code must treat `null` the same as an absent field, and keep prior behavior.

| Backend version vs. tag | Field present in schema | Result |
|---|---|---|
| Older than the tagged version | n/a | `null` (field stripped) |
| At or newer than the tagged version | Yes | value returned |
| At or newer than the tagged version | No | `null` |

The last row covers GitLab.com `-pre` builds and stale CI branches; it requires GitLab 19.4
([gitlab!251821](https://gitlab.com/gitlab-org/gitlab/-/merge_requests/251821)).

## When to add a new versioned file

Only for a non-additive change: removing a field, renaming a field, or restructuring the
query. `18_2_0.graphql` is the floor; older GitLab rejects unknown directives.
