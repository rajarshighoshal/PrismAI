# PrismAI server ops notes

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
