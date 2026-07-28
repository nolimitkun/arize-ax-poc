# SSO Setup

Single sign-on is available on Business and Enterprise plans.

## Supported providers

SAML 2.0 (Okta, Azure AD/Entra, OneLogin, Google Workspace, generic) and OIDC.
SCIM provisioning is Enterprise only.

## Configuring SAML

1. **Settings -> Authentication -> Enable SSO**, choose SAML.
2. Copy the ACS URL and Entity ID shown into your identity provider.
3. Paste your IdP's metadata URL, or upload the metadata XML, back into Nimbus.
4. Map attributes. Nimbus requires `email`; `firstName`, `lastName`, and
   `groups` are optional.
5. Click **Test connection**. This runs a full round trip using your own
   session and reports the exact assertion received.

Do not enable enforcement until the test passes.

## Enforcement

With **Require SSO** on, password login is disabled for everyone whose email
matches a verified domain. Verify domains under
**Settings -> Authentication -> Domains** by adding a DNS TXT record.

The Owner account retains a password-login fallback that cannot be disabled.
This is deliberate -- it is the recovery path if your IdP becomes unreachable.

## Group mapping

Map IdP groups to Nimbus roles under **Settings -> Authentication -> Group
mapping**. Unmapped groups grant no access. Users in no mapped group are
assigned the default role, configurable and defaulting to Viewer.

Mappings are evaluated at each login; changes take effect on the user's next
sign-in, not immediately.

## Just-in-time provisioning

On by default: a user authenticating successfully who has no Nimbus account gets
one created at the default role. Turn this off to require explicit invites, in
which case unknown users see `sso_user_not_provisioned`.

## Common failures

| Symptom                         | Cause                                        |
|---------------------------------|----------------------------------------------|
| `saml_invalid_signature`        | IdP certificate rotated; re-upload metadata  |
| `saml_audience_mismatch`        | Entity ID in IdP does not match Nimbus       |
| `sso_email_missing`             | `email` attribute not mapped                 |
| Redirect loop after login       | Clock skew > 5 min between IdP and Nimbus    |
