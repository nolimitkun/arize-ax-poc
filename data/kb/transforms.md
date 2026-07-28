# Transforms

A transform is SQL that runs between source and destination.

## Writing a transform

Open a pipeline and go to **Transform**. The source is exposed as a table named
`source`. Return the rows you want written:

```sql
SELECT
  id,
  lower(trim(email))      AS email,
  created_at::timestamptz AS created_at
FROM source
WHERE deleted_at IS NULL
```

The dialect is DuckDB SQL. Most Postgres syntax works unchanged; the notable
differences are that `SELECT INTO` is unsupported and window functions require
an explicit frame in some cases.

## Referencing other pipelines

A transform can read another pipeline's output in the same workspace using
`pipeline('name')`:

```sql
SELECT s.*, u.plan
FROM source s
LEFT JOIN pipeline('users') u ON u.id = s.user_id
```

Nimbus resolves the dependency graph and runs upstream pipelines first. Circular
references are rejected at save time.

## Testing

**Preview** runs the transform against the rolling 7-day sample and shows the
first 100 result rows, plus the inferred output schema. Preview never writes to
the destination.

Preview uses sampled data, so `COUNT(*)` in preview is not your true row count.

## Limits

| Plan     | Transform timeout | Memory  |
|----------|-------------------|---------|
| Free     | 60s               | 1 GB    |
| Team     | 300s              | 4 GB    |
| Business | 1800s             | 32 GB   |

A transform exceeding its timeout fails the run with `transform_timeout`. The
usual fixes are filtering earlier, avoiding cross joins, and splitting one
pipeline into two.

## Incremental transforms

For incremental pipelines, `source` contains only the new rows for that run, not
the full table. Aggregations across all history therefore need a full resync or
a destination-side view.

Use `{{ cursor }}` in the transform to reference the current cursor value if you
need it explicitly.

## Version history

Every save creates a version. **Transform -> History** shows a diff of each and
allows rollback. Rolling back does not re-run past executions.
