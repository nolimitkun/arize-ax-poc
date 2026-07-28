# Troubleshooting Pipeline Runs

## `connection_timeout`

The source did not respond within 30 seconds.

- Confirm the host is reachable from Nimbus egress IPs (see
  [network access](network-access.md))
- For databases behind a firewall, allowlist our egress ranges
- Large initial syncs on slow sources: raise the connection's read timeout under
  **Connection -> Advanced -> Read timeout** (max 300 seconds)

## `auth_failed`

Credentials were rejected by the source. Most often a rotated database password
or an expired cloud key. Re-enter credentials on the connection and click
**Validate**. The pipeline resumes on its next scheduled run.

## `schema_drift`

A column present in a previous run is missing, or its type changed
incompatibly. Nimbus halts rather than silently dropping data.

Resolve by opening the pipeline's **Schema** tab and accepting the new schema,
or by pinning the affected column. Accepting a schema change applies from the
next run forward; it does not backfill.

## `destination_write_failed`

The destination rejected the write. Check the run log for the underlying error.
Common causes: destination out of storage, a table locked by another process, or
a permissions change on the destination credential.

## `row_limit_exceeded`

The workspace hit its monthly row allowance. On Free plans pipelines stop until
the next period. On Team and Business they continue and bill as overage -- if
the pipeline stopped instead, the workspace has a spend cap configured under
**Settings -> Billing -> Spend cap**.

## Runs stuck in `queued`

You are at your concurrent-run limit. See [rate limits](rate-limits.md). Queued
runs wait 30 minutes, then are skipped.

## Runs succeed but write zero rows

Almost always an incremental cursor that has advanced past your data. Check the
pipeline's **Cursor** value under the Schema tab. To re-read from the beginning,
use **Run -> Full resync**, which ignores the cursor for one run.

## Getting help

If a run fails with an error not listed here, open the run, click
**Copy diagnostics**, and include that payload when you contact support. It
carries the run ID, connector versions, and a redacted error trace.
