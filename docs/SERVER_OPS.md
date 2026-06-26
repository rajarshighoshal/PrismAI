# PrismAI server ops notes

## Production Fugu policy

Keep `ENABLE_FUGU=false` in production until Sakana has an explicit EU/GDPR-supported
path for this account.

The previous `FUGU_BASE_URL=http://172.18.0.1:8080/v1` setup depended on a laptop
SSH reverse tunnel into the EU server. It was useful for testing, but it is fragile
and should not be a production dependency:

- if the laptop, network, or SSH session drops, Fugu drops;
- the listener is owned by `sshd`, not by a supervised service;
- the relay may be interpreted as avoiding the provider's regional controls.

If Fugu is re-enabled later, prefer an official Sakana EU/GDPR endpoint. If a relay
is used for testing, make it temporary and keep PrismAI's honesty verifier enabled
on every Fugu response.

## R2 backups

`backup-r2.sh` is committed because it contains no secrets. It reads credentials from
the ignored `/opt/owui-hybrid-router/r2-backup.env` file.

Cron:

```cron
15 3 * * * /opt/owui-hybrid-router/backup-r2.sh >> /var/log/r2-backup.log 2>&1
```

What it backs up:

- consistent `webui.db` snapshot from the `open-webui` container;
- consistent `router_mem.db` snapshot from the `owui-tool-server` container;
- the OpenWebUI `uploads/` directory.

R2 layout:

```text
prismai/owui/webui.db
prismai/owui/router_mem.db
prismai/owui/uploads/
prismai/snapshots/YYYY-MM-DD/webui.db
prismai/snapshots/YYYY-MM-DD/router_mem.db
```

The dated DB snapshots are retained for `SNAPSHOT_RETENTION_DAYS` days, default 14.

## Pricing updater

Monthly cron should run:

```cron
0 2 1 * * docker exec owui-tool-server python3 /app/update_prices.py >> /var/log/pricing-update.log 2>&1
```

`update_prices.py` writes `/app/backend/data/usage_prices.json`. The usage panel
reloads that file on mtime change, so the container does not need a restart after
the cron refresh.
