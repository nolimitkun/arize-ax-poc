# Webhooks

Webhooks notify your systems when pipeline events occur.

## Creating a webhook

**Settings -> Webhooks -> New**. Provide an HTTPS URL and select the events to
subscribe to. HTTP URLs are rejected; the endpoint must present a valid
certificate.

## Events

| Event               | Fires when                                       |
|---------------------|--------------------------------------------------|
| `run.started`       | A pipeline run begins                            |
| `run.succeeded`     | A run completes without error                    |
| `run.failed`        | A run fails after exhausting retries             |
| `pipeline.paused`   | A pipeline is paused, manually or by limits      |
| `connection.broken` | A connection's credentials stop validating       |
| `quota.warning`     | Workspace crosses 80% of its monthly row quota   |

## Payload

```json
{
  "id": "evt_01HX...",
  "type": "run.failed",
  "created_at": "2026-03-14T09:21:04Z",
  "workspace": "acme-production",
  "data": {
    "pipeline_id": "pl_8fj2",
    "run_id": "run_91kd",
    "error": "connection_timeout",
    "rows_written": 0
  }
}
```

## Verifying deliveries

Each request carries `X-Nimbus-Signature`, an HMAC-SHA256 of the raw body using
your webhook's signing secret. Compute over the **raw bytes** -- re-serializing
the JSON changes the bytes and the signature will not match.

The secret is shown once at creation and can be rotated from the webhook's
detail page. During rotation both secrets are valid for 24 hours.

## Retries

Non-2xx responses are retried 5 times over roughly 1 hour with exponential
backoff. Redirects (3xx) are **not** followed and count as a failure. After the
final failure the event is dropped and a `delivery.failed` entry appears in the
webhook's log.

A webhook returning non-2xx continuously for 24 hours is automatically disabled.
Re-enable it from the detail page once your endpoint recovers.

## Timeouts

Your endpoint must respond within 10 seconds. Do the minimum work needed to
acknowledge -- queue the payload and return 200 immediately.
