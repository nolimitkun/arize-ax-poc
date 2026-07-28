# Getting Started with Nimbus

Nimbus is a managed data pipeline platform. This guide takes you from signup to
your first successful pipeline run.

## 1. Create a workspace

After signing up, you are prompted to create a workspace. A workspace is the
top-level container for pipelines, connections, and team members. Most
organizations use one workspace per environment (`production`, `staging`).

Workspace names must be lowercase alphanumeric with hyphens, 3-32 characters.
The name appears in your API endpoints and **cannot be changed after creation**.

## 2. Connect a source

From **Connections -> New**, pick a source type. Nimbus ships connectors for
PostgreSQL, MySQL, S3, Google Cloud Storage, Snowflake, and generic REST.

Nimbus performs a validation read during setup. If validation fails, the
connection is saved in a `draft` state and will not be scheduled.

## 3. Build a pipeline

A pipeline is a source, an optional transform, and a destination. From
**Pipelines -> New**, select your connection as the source, then choose a
destination. Transforms are written in SQL and run on the Nimbus compute layer.

## 4. Run and schedule

Click **Run now** for a one-off execution. To schedule, open the **Schedule**
tab and choose an interval (minimum 5 minutes on Team, 1 minute on Business).

The first run of a pipeline performs a full sync; subsequent runs are
incremental where the source supports it.

## 5. Monitor

The **Runs** tab shows every execution with row counts, duration, and errors.
Failed runs are retried automatically up to 3 times with exponential backoff
before the pipeline is marked `failing`.

## Next steps

- Set up [API keys](api-keys.md) for programmatic access
- Invite teammates under [team permissions](team-permissions.md)
- Configure [webhooks](webhooks.md) for run notifications
