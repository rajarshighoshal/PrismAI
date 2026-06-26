#!/usr/bin/env python3
"""Tiny TLS relay: laptop localhost:8080 -> api.sakana.ai:443.

TESTING ONLY. Do not use this to avoid provider regional or contractual
restrictions. Production PrismAI should keep ENABLE_FUGU=false until Sakana
provides an explicit EU/GDPR-supported endpoint for this account.

For short-lived manual Fugu plumbing tests from an EU server through a laptop.
Keeps the TLS handshake between laptop and Sakana — the server
connects via plain HTTP through an SSH reverse tunnel.

Usage:
  python3 fugu_relay.py          # listen on :8080
  # Then from the EU server:
  #   ssh -R 8080:localhost:8080 user@laptop-ip
  #   FUGU_BASE_URL=http://localhost:8080/v1
"""
import asyncio
import ssl
import sys


SAKANA_HOST = "api.sakana.ai"
SAKANA_PORT = 443
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 8080


async def relay(src: asyncio.StreamReader, dst: asyncio.StreamWriter):
    """Copy bytes one direction."""
    try:
        while not src.at_eof():
            data = await src.read(65536)
            if not data:
                break
            dst.write(data)
            await dst.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
        pass
    finally:
        try:
            dst.close()
        except Exception:
            pass


async def handle(client_reader, client_writer):
    """Handle one incoming HTTP connection."""
    try:
        # TLS upstream to Sakana
        ctx = ssl.create_default_context()
        upstream_reader, upstream_writer = await asyncio.open_connection(
            SAKANA_HOST, SAKANA_PORT, ssl=ctx
        )
    except Exception as e:
        print(f"[relay] upstream failed: {e}", file=sys.stderr)
        try:
            client_writer.close()
        except Exception:
            pass
        return

    # Bidirectional relay
    c2u = asyncio.create_task(relay(client_reader, upstream_writer), name="c2u")
    u2c = asyncio.create_task(relay(upstream_reader, client_writer), name="u2c")
    try:
        await asyncio.wait([c2u, u2c], return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in [c2u, u2c]:
            if not t.done():
                t.cancel()
        try:
            client_writer.close()
        except Exception:
            pass


async def main():
    server = await asyncio.start_server(handle, LISTEN_HOST, LISTEN_PORT)
    print(f"[relay] listening on {LISTEN_HOST}:{LISTEN_PORT} -> {SAKANA_HOST}:{SAKANA_PORT}")
    print(f"[relay] ready — run on EU server:")
    print(f"        ssh -R 8080:localhost:8080 root@YOUR_LAPTOP_IP")
    print(f"        then set FUGU_BASE_URL=http://localhost:8080/v1")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
