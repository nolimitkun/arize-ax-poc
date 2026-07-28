# Team Permissions

## Roles

| Role      | Pipelines        | Connections   | Members | Billing |
|-----------|------------------|---------------|---------|---------|
| Viewer    | read             | read          | read    | -       |
| Editor    | create/edit/run  | create/edit   | read    | -       |
| Admin     | full             | full          | full    | read    |
| Owner     | full             | full          | full    | full    |

Every workspace has exactly one Owner. Ownership can be transferred from
**Settings -> Members -> Transfer ownership**; the current Owner becomes an
Admin once the transfer is accepted.

## Inviting members

Admins and Owners can invite from **Settings -> Members -> Invite**. Invites are
sent by email and expire after 7 days. Unaccepted invites do not count against
your seat limit.

Seat limits: Free 3, Team 10, Business 50, Enterprise unlimited.

## Per-pipeline access

Business and Enterprise workspaces can restrict individual pipelines to named
members or groups. Open a pipeline, then **Settings -> Access**. A member
without access does not see the pipeline in listings at all.

This is layered *on top of* the workspace role -- a Viewer granted pipeline
access still cannot edit it.

## Groups

Business and Enterprise workspaces support groups. Create under
**Settings -> Groups**, then assign a workspace role to the group and add
members. A member's effective permissions are the union of their direct role and
all group roles.

## Removing a member

Removal is immediate. Pipelines they created are **not** deleted; ownership of
those pipelines transfers to the workspace Owner. API keys they created remain
active -- revoke these separately if needed, from **Settings -> API Keys**.

## Service accounts

For automation, prefer a service account over a personal member. Create under
**Settings -> Members -> Add service account**. Service accounts have a role but
no email, cannot log in to the UI, and are billed as a seat only on Free and
Team plans.
