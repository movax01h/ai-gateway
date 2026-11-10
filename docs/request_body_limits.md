# Request body limits

AI Gateway answers HTTP 413 when a request body is larger than a limit. The gateway checks the size before it
buffers the body.

## Limits

- The code suggestion routes (`/v2/completions`, `/v2/code/`, `/v3/code/`, and `/v4/code/`) have a fixed limit of
  600,000 bytes.
- All other HTTP routes, except `/v1/proxy/`, use the limit from `AIGW_REQUEST_BODY_LIMIT__MAX_BYTES`. The default
  is `0`, which turns the limit off. GitLab.com sets it to 16 MiB (`16777216`).

The proxy routes have no limit, because legitimate proxy bodies can be larger than 16 MiB. AI Gateway does not parse
`model_metadata` on these routes.

## Response

The client gets HTTP 413 with a `Connection: close` header and a JSON body, for example:

```json
{"detail": "Request body exceeds 16777216 bytes"}
```

Before the 413, the gateway reads the rest of the body and drops it, up to 64 MiB. Some clients send the whole body
before they read the response, for example Ruby's `Net::HTTP`, which GitLab Rails uses. Without this drain, these
clients get a connection reset instead of the 413.

## Logs

A rejected request writes one or two log lines with the logger `request_body_limit`. Both lines have the
`correlation_id` of the request, so you can match them with the access log line that has `status_code` 413.

To find these lines in Cloud Logging, use the filter `jsonPayload.logger="request_body_limit"`.

### `Request body exceeds limit`

The gateway writes this warning when it rejects the body, before it drains the rest of the body.

| Field        | Description |
|--------------|-------------|
| `path`       | The request path. |
| `body_bytes` | The size of the body. The value depends on `declared`. |
| `declared`   | `true` if `body_bytes` is the `Content-Length` header. `false` if the request has no `Content-Length`. Then `body_bytes` is the byte count at the chunk that crossed the limit. |
| `max_bytes`  | The limit that rejected the body: `600000` on the code suggestion routes, or the value of `AIGW_REQUEST_BODY_LIMIT__MAX_BYTES`. |

### `Rejected request body drained`

The gateway writes this line after the drain. The line is not there if the chunk that crossed the limit was the last
chunk, because then there is nothing to drain. The line is also not there if the drain did not end, for example
because the request was cancelled.

| Field           | Description |
|-----------------|-------------|
| `path`          | The request path. |
| `drained_bytes` | The number of bytes that the gateway read and dropped. |
| `outcome`       | How the drain ended. See the next table. |
| `duration_s`    | The time that the drain took, in seconds. |

| `outcome`             | Meaning |
|-----------------------|---------|
| `body_complete`       | The client sent the whole body. Then the gateway sent the 413. |
| `client_disconnected` | The client closed the connection before it sent the whole body. |
| `drain_limit`         | The gateway stopped after 64 MiB and sent the 413 while the client was still sending. The client can get a connection reset. |

## Troubleshooting

### A client gets a connection reset instead of a 413

Find the `Request body exceeds limit` warning for the request, then the drain line with the same `correlation_id`:

- `body_complete`: the gateway read the whole body and sent the 413. Look for the reset between the client and the
  gateway, for example at the load balancer.
- `drain_limit`: the body was larger than 64 MiB.
- `client_disconnected`: the client closed the connection first.
- No drain line: the chunk that crossed the limit was the last chunk, or the drain did not end.

### Legitimate requests get a 413

Look at the `path` and `max_bytes` of the `Request body exceeds limit` warnings. If the limit is too low for a route,
increase `AIGW_REQUEST_BODY_LIMIT__MAX_BYTES`. The 600,000-byte limit on the code suggestion routes is set in the
code, in `ai_gateway/api/server.py`.
