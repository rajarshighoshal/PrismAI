#!/bin/bash
# Disk space alert — emails if / exceeds ALERT_PCT (default 80).
# Set ALERT_EMAIL env var to override the alert recipient.
set -euo pipefail

ALERT_PCT="${ALERT_PCT:-80}"
ALERT_EMAIL="${ALERT_EMAIL:-root}"

USED=$(df / | tail -1 | awk '{print $5}' | tr -d '%')
if [ "$USED" -gt "$ALERT_PCT" ]; then
  echo "Disk ${USED}% full on $(hostname) ($(date))" | mail -s "DISK ALERT" "$ALERT_EMAIL" 2>/dev/null
fi
