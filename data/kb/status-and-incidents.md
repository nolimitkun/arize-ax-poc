# Status and Incidents

## Status page

Live status is at `status.nimbus.example`. Subscribe there for email or Slack
notifications. The page reports per-region status for the API, the scheduler,
the transform compute layer, and the web UI independently.

## Severity levels

| Level | Meaning                                        | Update cadence |
|-------|------------------------------------------------|----------------|
| SEV-1 | Total outage of a component in a region        | Every 30 min   |
| SEV-2 | Major degradation; most requests affected      | Every 60 min   |
| SEV-3 | Partial degradation; a subset affected         | Every 2 hours  |
| SEV-4 | Minor issue, no meaningful customer impact     | At resolution  |

## What happens to your pipelines during an incident

Scheduled runs are **not lost**. If the scheduler is degraded, runs are queued
and executed once it recovers. A run that would have started during the outage
executes late rather than being skipped, unless it queues for more than 30
minutes (see [rate limits](rate-limits.md)).

Runs interrupted mid-execution are retried from the beginning. For incremental
pipelines the cursor is only advanced on successful completion, so an
interrupted run does not skip data.

## Post-incident reports

SEV-1 and SEV-2 incidents get a public post-incident report within 5 business
days, published on the status page.

## SLA

| Plan       | Uptime SLA | Credits                              |
|------------|------------|--------------------------------------|
| Free       | None       | -                                    |
| Team       | None       | -                                    |
| Business   | 99.9%      | 10% of monthly fee per 0.1% below    |
| Enterprise | 99.95%     | Negotiated in contract               |

SLA credits are applied to a future invoice. To claim, open a support ticket
within 30 days of the incident referencing the incident ID from the status page.
Credits are the sole remedy under the SLA.

## Scheduled maintenance

Announced at least 7 days ahead on the status page. Maintenance windows are
Sundays 02:00-06:00 in the workspace's region. Pipelines continue running during
maintenance; only the UI and API may be briefly unavailable.
