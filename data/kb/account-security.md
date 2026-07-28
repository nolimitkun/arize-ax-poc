# Account Security

## Multi-factor authentication

Enable from **Account -> Security -> Two-factor authentication**. Supported:
TOTP authenticator apps and hardware security keys (WebAuthn/FIDO2). SMS is not
supported.

You are shown 10 single-use recovery codes at enrollment. These are displayed
once. Store them in a password manager.

## Requiring MFA workspace-wide

Admins can require MFA under **Settings -> Authentication -> Require 2FA**.
Members without MFA are prompted to enrol at next sign-in and cannot access the
workspace until they do.

If SSO enforcement is on, MFA is delegated to your identity provider and this
setting has no effect.

## Lost MFA device

Use a recovery code. Out of recovery codes: another workspace Admin can reset
your MFA from **Settings -> Members -> Reset 2FA**. If you are the sole Owner
with no recovery codes, contact support -- identity verification takes 3-5
business days and requires access to the billing email and payment method on
file.

## Sessions

Sessions last 30 days, or 12 hours in workspaces with **Short sessions**
enabled. See and revoke active sessions under **Account -> Security -> Sessions**.
Revoking a session is immediate.

Changing your password revokes every other session automatically.

## Audit log

Business and Enterprise workspaces have an audit log at
**Settings -> Audit log**, recording sign-ins, permission changes, key
creation/revocation, pipeline edits, and billing changes. Each entry carries
actor, IP, user agent, and timestamp.

Export as CSV, or stream to S3 (Enterprise).

## Reporting a vulnerability

Email `security@nimbus.example` with reproduction steps. We acknowledge within
one business day. Please do not test against workspaces you do not own.

## Compromised key

Revoke it immediately from **Settings -> API Keys** -- revocation takes effect
at once. Then check the audit log for actions taken with that key, and rotate
any source or destination credentials the key could have read.
