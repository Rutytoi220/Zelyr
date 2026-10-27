#!/usr/bin/env python3
"""Zelyr Headless Swarm Node — Zero-Trust Tailscale Launcher.

Automatically detects the host's Tailscale mesh IP and binds the FastAPI
server exclusively to it, making the node invisible to the public internet.
Falls back to 0.0.0.0 (all interfaces) if Tailscale is unavailable.
"""
import argparse
import subprocess
import sys

import uvicorn

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  ASCII Banner                                                            #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
BANNER = r"""
  ╔═══════════════════════════════════════════════════════════════╗
  ║                                                               ║
  ║                     Z  E  L  Y  R                            ║
  ║                                                               ║
  ║           S W A R M   N O D E   v 1 1 . 0                    ║
  ║                                                               ║
  ╚═══════════════════════════════════════════════════════════════╝
"""

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"
MAG    = "\033[35m"
BG_GREEN = "\033[42m"
BG_RED   = "\033[41m"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Tailscale Auto-Discovery                                                #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
def get_tailscale_ip() -> str | None:
    """Attempt to discover this host's Tailscale IPv4 address.

    Returns the 100.x.x.x IP if Tailscale is installed and connected,
    otherwise returns None so the caller can decide on a fallback.
    """
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            ip = result.stdout.strip().splitlines()[0].strip()
            if ip.startswith("100."):
                return ip
    except FileNotFoundError:
        pass  # tailscale binary not on PATH
    except subprocess.TimeoutExpired:
        pass  # tailscale hung — skip
    except Exception:
        pass
    return None


def get_tailscale_hostname() -> str | None:
    """Return the Tailscale MagicDNS hostname (e.g. 'zelyr-node')."""
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            self_node = data.get("Self", {})
            dns_name = self_node.get("DNSName", "")
            if dns_name:
                return dns_name.rstrip(".")
    except Exception:
        pass
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Main                                                                    #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
def main():
    parser = argparse.ArgumentParser(description="Zelyr Headless Swarm Node")
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--bind", type=str, default=None,
        help="Explicit bind address. Overrides auto-detection.",
    )
    parser.add_argument(
        "--no-tailscale", action="store_true",
        help="Skip Tailscale auto-detection; bind to 0.0.0.0.",
    )
    args = parser.parse_args()

    print(f"{CYAN}{BANNER}{RESET}")

    # ── Network detection ─────────────────────────────────────────── #
    ts_ip = None
    ts_hostname = None
    bind_addr = "0.0.0.0"

    if args.bind:
        bind_addr = args.bind
        print(f"  {BOLD}[🔧 MANUAL BIND]{RESET}  Using explicit address: {BOLD}{bind_addr}{RESET}")
    elif args.no_tailscale:
        bind_addr = "0.0.0.0"
        print(f"  {BOLD}[⚡ LAN MODE]{RESET}     Binding to all interfaces ({bind_addr})")
    else:
        ts_ip = get_tailscale_ip()
        ts_hostname = get_tailscale_hostname()
        if ts_ip:
            bind_addr = ts_ip
        else:
            bind_addr = "0.0.0.0"

    print()

    # ── Status banner ─────────────────────────────────────────────── #
    if ts_ip:
        print(f"  {BG_GREEN}{BOLD} 🛡️  ZERO-TRUST NODE ONLINE {RESET}")
        print()
        print(f"  {GREEN}[🌐]{RESET} Tailscale Network Detected.")
        if ts_hostname:
            print(f"  {GREEN}[📛]{RESET} MagicDNS Hostname: {BOLD}{ts_hostname}{RESET}")
        print(f"  {GREEN}[🔒]{RESET} Bound exclusively to Tailscale mesh — invisible to public internet.")
        print()
        print(f"  {CYAN}╭──────────────────────────────────────────────────╮{RESET}")
        print(f"  {CYAN}│{RESET}  {BOLD}Enter this into your Zelyr Desktop App:{RESET}       {CYAN}│{RESET}")
        print(f"  {CYAN}│{RESET}                                                  {CYAN}│{RESET}")
        print(f"  {CYAN}│{RESET}     {GREEN}{BOLD}{ts_ip}:{args.port}{RESET}{''.ljust(37 - len(f'{ts_ip}:{args.port}'))}{CYAN}│{RESET}")
        if ts_hostname:
            magic_addr = f"{ts_hostname}:{args.port}"
            print(f"  {CYAN}│{RESET}     {DIM}or: {magic_addr}{RESET}{''.ljust(33 - len(magic_addr))}{CYAN}│{RESET}")
        print(f"  {CYAN}│{RESET}                                                  {CYAN}│{RESET}")
        print(f"  {CYAN}╰──────────────────────────────────────────────────╯{RESET}")
    else:
        print(f"  {YELLOW}{BOLD}[⚠️  LOCAL / LAN MODE]{RESET}")
        print()
        print(f"  {YELLOW}[!]{RESET} Tailscale not detected — binding to {BOLD}{bind_addr}{RESET}")
        print(f"  {DIM}    Install Tailscale for zero-trust encrypted mesh networking.{RESET}")
        print(f"  {DIM}    https://tailscale.com/download{RESET}")
        print()
        print(f"  {CYAN}╭──────────────────────────────────────────────────╮{RESET}")
        print(f"  {CYAN}│{RESET}  {BOLD}Connect from your Zelyr Desktop on this LAN:{RESET}   {CYAN}│{RESET}")
        print(f"  {CYAN}│{RESET}                                                  {CYAN}│{RESET}")
        print(f"  {CYAN}│{RESET}     {YELLOW}{BOLD}{bind_addr}:{args.port}{RESET}{''.ljust(37 - len(f'{bind_addr}:{args.port}'))}{CYAN}│{RESET}")
        print(f"  {CYAN}│{RESET}                                                  {CYAN}│{RESET}")
        print(f"  {CYAN}╰──────────────────────────────────────────────────╯{RESET}")

    print()
    print(f"  {DIM}───────────────────────────────────────────────────────{RESET}")
    print(f"  {DIM}Starting Uvicorn on {bind_addr}:{args.port} ...{RESET}")
    print(f"  {DIM}Press Ctrl+C to shut down.{RESET}")
    print(f"  {DIM}───────────────────────────────────────────────────────{RESET}")
    print()

    uvicorn.run(
        "zelyr.server.node_api:app",
        host=bind_addr,
        port=args.port,
        ws="auto",
        log_level="info",
    )


if __name__ == "__main__":
    main()
