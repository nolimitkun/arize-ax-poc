# Billing and Plans

## Plans

| Plan     | Monthly | Pipelines | Rows/month | Min schedule | Support   |
|----------|---------|-----------|------------|--------------|-----------|
| Free     | $0      | 2         | 500K       | 60 min       | Community |
| Team     | $99     | 25        | 25M        | 5 min        | Email     |
| Business | $499    | 200       | 250M       | 1 min        | Priority  |
| Enterprise | Custom | Unlimited | Custom    | 1 min        | Dedicated |

Row counts are measured as rows written to destinations, summed across all
pipelines in the billing period.

## Billing period

Billing runs monthly from the day you subscribe. Invoices are issued on the
first day of each period and charged to the card on file. You can view and
download past invoices under **Settings -> Billing -> Invoices**.

## Upgrading

Upgrades take effect immediately. You are charged a prorated amount for the
remainder of the current period, calculated daily.

## Downgrading

Downgrades take effect at the **end** of the current billing period. Your
current plan's limits stay in force until then. If your usage exceeds the
target plan's limits at the switchover date, pipelines above the limit are
paused in reverse order of creation until you are back within limits.

## Overages

Rows written beyond your plan's monthly allowance are billed at $2 per
additional million rows on Team and $1 per additional million on Business.
Free-plan workspaces are hard-capped: pipelines stop once the 500K limit is
hit, and resume at the start of the next period.

## Cancelling

Cancel from **Settings -> Billing -> Cancel plan**. Cancellation stops future
charges and moves the workspace to the Free plan at the end of the current
billing period. Your data and pipeline definitions are retained for 30 days
after cancellation, after which the workspace is deleted.

## Payment methods

Nimbus accepts Visa, Mastercard, and American Express. Enterprise customers may
pay by invoice with NET-30 terms. We do not currently support PayPal or ACH on
self-serve plans.

## Failed payments

If a charge fails, we retry on days 3, 5, and 7. After the fourth failure the
workspace is downgraded to Free and pipelines above the Free limits are paused.
