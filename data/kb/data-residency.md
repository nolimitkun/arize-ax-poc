# Data Residency and Regions

## Choosing a region

A workspace's region is chosen at creation and **cannot be changed afterwards**.
To move regions, create a new workspace in the target region and re-import your
pipeline definitions (see [data export](data-export.md)).

Available regions: `us-east`, `us-west`, `eu-central`, `ap-southeast`.

## What stays in region

All pipeline metadata, run history, schema samples, audit logs, and transform
execution stay within the workspace's region. Compute for transforms runs in the
same region.

## What is global

Three things are global regardless of workspace region:

- Account identity (email, password hash, MFA enrollment)
- Billing records and invoices
- Status-page and incident notifications

Enterprise customers with strict requirements can request regional identity
isolation; contact your account team.

## EU workspaces

`eu-central` workspaces are hosted in Frankfurt. Sub-processors for EU
workspaces are listed in the DPA. A signed DPA is available from
**Settings -> Legal -> Data processing agreement** on Business and above.

## Latency

Choose the region closest to your **sources**, not your team. Pipeline
throughput is dominated by source read latency; the UI is served from a CDN and
is equally fast everywhere.

Cross-region pipelines (source in one region, Nimbus workspace in another) work
but typically run 2-4x slower on initial syncs.

## Retention

| Data                 | Retention                              |
|----------------------|----------------------------------------|
| Run history          | 400 days                               |
| Schema samples       | 7 days, rolling                        |
| Audit log            | 400 days (Business), 7 years (Enterprise) |
| Deleted workspace    | 30 days, then purged                   |
