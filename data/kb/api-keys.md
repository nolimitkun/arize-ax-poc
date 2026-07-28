# API Keys

## Creating a key

Go to **Settings -> API Keys -> Create key**. Choose a scope and an expiry, then
copy the key. The secret is shown **once only** and cannot be retrieved later --
if you lose it, revoke the key and create a new one.

Key format is `nmb_live_` followed by 32 characters. Staging workspaces issue
`nmb_test_` keys, which are rejected by production endpoints.

## Scopes

| Scope            | Grants                                              |
|------------------|-----------------------------------------------------|
| `read`           | List and read pipelines, runs, connections           |
| `write`          | Everything in `read`, plus create/update/delete      |
| `run`            | Trigger pipeline runs; cannot modify definitions     |
| `admin`          | Everything, including billing and member management  |

Scopes are additive and set at creation time. A key's scope cannot be changed
after creation.

## Using a key

Pass the key as a bearer token:

```
curl https://api.nimbus.example/v1/pipelines \
  -H "Authorization: Bearer nmb_live_..."
```

Keys sent in query parameters are rejected with `400 key_in_query`.

## Expiry

Keys can be created with an expiry of 30, 90, or 365 days, or as non-expiring.
Non-expiring keys require the `admin` scope to create and are disallowed
entirely in workspaces with the "Require key rotation" policy enabled.

You receive email warnings 14 days and 1 day before a key expires.

## Revoking

Revoke from **Settings -> API Keys**. Revocation is immediate and irreversible.
In-flight requests using the key complete; new requests fail with
`401 key_revoked`.

## Rotating without downtime

Create the new key first, deploy it, confirm traffic has moved using the
**Last used** column, then revoke the old key. Nimbus permits up to 20 active
keys per workspace to make overlap-based rotation straightforward.
