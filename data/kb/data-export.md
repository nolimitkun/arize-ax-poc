# Data Export

## Exporting pipeline definitions

**Settings -> Export -> Pipeline definitions** produces a ZIP of JSON files, one
per pipeline, including transforms and schedules. Connection **credentials are
excluded** -- the export references connections by ID only.

These files are the input to the Nimbus Terraform provider, so the usual path
for moving a workspace between environments is export, commit, then apply.

## Exporting run history

**Settings -> Export -> Run history** produces CSV covering up to the last 400
days. Columns: run ID, pipeline, started, finished, status, rows read, rows
written, error code.

Exports over 100 MB are generated asynchronously; you receive an email with a
download link valid for 24 hours.

## Exporting your data

Nimbus is a pipeline, not a warehouse -- your rows land in your destination and
Nimbus retains only metadata plus a rolling 7-day sample used for schema
inference and previews. There is no bulk export of row data because Nimbus does
not hold it.

The 7-day sample can be purged immediately from
**Settings -> Privacy -> Purge samples**.

## Account deletion export

Before deleting a workspace, use **Settings -> Export -> Full archive**. This
bundles pipeline definitions, run history, member list, and audit log into a
single ZIP. Generation takes up to 1 hour for large workspaces.

Request the archive **before** starting deletion -- once deletion begins the
archive can no longer be generated.

## API

Everything above is available programmatically:

```
POST /v1/exports          {"type": "pipelines" | "runs" | "archive"}
GET  /v1/exports/{id}     -> {"status": "pending"|"ready", "url": "..."}
```

Export jobs require the `admin` scope.
