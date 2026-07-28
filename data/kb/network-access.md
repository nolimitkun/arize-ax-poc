# Network Access

## Egress IPs

Nimbus connects to your sources and destinations from a fixed set of IPs.
Allowlist these on your firewall or database security group.

| Region        | Egress IPs                                   |
|---------------|----------------------------------------------|
| `us-east`     | 52.14.88.0/29, 52.14.88.16/29                |
| `us-west`     | 54.202.11.0/29                               |
| `eu-central`  | 18.196.44.0/29, 18.196.44.16/29              |
| `ap-southeast`| 13.212.7.0/29                                |

Egress IPs are stable. We give 90 days' notice before changing them, announced
on the status page and by email to workspace Admins.

## Private connectivity

Business and Enterprise workspaces can use AWS PrivateLink or GCP Private
Service Connect instead of public egress. Request under
**Settings -> Network -> Private connectivity**; setup takes 2-3 business days
and requires your VPC endpoint service name.

## SSH tunnels

For sources not reachable directly, Nimbus supports SSH tunneling. On the
connection, choose **Connect via SSH bastion** and provide the bastion host,
port, and username. Nimbus generates a keypair; add the public key to the
bastion's `authorized_keys`.

The bastion must permit TCP forwarding. Password authentication to the bastion
is not supported.

## Static outbound for webhooks

Webhook deliveries originate from the same egress IPs as the table above, so a
single allowlist covers both directions.

## IP allowlisting for the API

Business and Enterprise workspaces can restrict which IPs may call the Nimbus
API, under **Settings -> Network -> API allowlist**. The allowlist applies to
API keys only, never to browser sessions -- locking yourself out of the UI this
way is not possible.

An empty allowlist means "allow all", not "deny all".
