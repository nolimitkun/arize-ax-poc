# Rate Limits

## API rate limits

Limits are applied per workspace, not per API key.

| Plan     | Requests/min | Concurrent runs |
|----------|--------------|-----------------|
| Free     | 60           | 1               |
| Team     | 600          | 5               |
| Business | 3,000        | 25              |
| Enterprise | Custom     | Custom          |

Every response carries the current state:

```
X-RateLimit-Limit: 600
X-RateLimit-Remaining: 412
X-RateLimit-Reset: 1735689600
```

## When you exceed a limit

Requests over the limit return `429 rate_limited` with a `Retry-After` header in
seconds. The response body includes the limit that was hit:

```json
{"error": "rate_limited", "limit": "requests_per_minute", "retry_after": 23}
```

We recommend exponential backoff with jitter, starting at 1 second and capping
at 60 seconds.

## Concurrent run limits

If a scheduled run would exceed your concurrent-run limit, it is **queued**, not
dropped. Queued runs wait up to 30 minutes; beyond that they are marked
`skipped` and the next scheduled occurrence proceeds normally.

Manual runs triggered from the UI or via the `run` scope jump the queue ahead of
scheduled runs.

## Burst allowance

Team and Business plans have a burst allowance of 2x the per-minute limit,
usable for up to 10 seconds, replenished over the following minute. This is
intended to absorb deploy-time spikes, not sustained load.

## Requesting a raise

Business and Enterprise customers can request a limit raise from
**Settings -> Limits -> Request increase**. Include your expected sustained and
peak request rate. Most requests are reviewed within two business days.

Free and Team plans cannot have limits raised; upgrade instead.
