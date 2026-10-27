"""Network security validation and SSRF protection for Zelyr."""

from __future__ import annotations

import ipaddress
import logging
import socket
import urllib.parse
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


# Additional reserved / private subnets not flagged by is_private or is_reserved in stdlib
DISALLOWED_NETWORKS = (
    ipaddress.ip_network("100.64.0.0/10"),  # RFC 6598 Carrier-Grade NAT / Tailscale private mesh
    ipaddress.ip_network("0.0.0.0/8"),      # RFC 1122 Current network / local host
    ipaddress.ip_network("2001:db8::/32"),  # RFC 3849 IPv6 Documentation
)


def validate_safe_url(url: str) -> str:
    """Validate that a URL is safe to fetch, enforcing strict SSRF protections.

    Rejects:
    - Schemes other than http and https
    - Missing or empty hostnames
    - Loopback IPs (127.0.0.0/8, ::1)
    - Private network subnets (RFC 1918: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
    - Carrier-Grade NAT / Tailscale private mesh IPs (RFC 6598: 100.64.0.0/10)
    - Link-local and cloud metadata IPs (169.254.0.0/16, fe80::/10)
    - Reserved, multicast, unspecified (0.0.0.0), and IPv4-mapped IPv6 private ranges

    Returns the validated URL string on success, or raises ValueError on violation.
    """
    if not url or not isinstance(url, str):
        raise ValueError("SSRF blocked: URL must be a non-empty string")

    parsed = urllib.parse.urlparse(url.strip())
    scheme = (parsed.scheme or "").lower()

    if scheme not in ("http", "https"):
        raise ValueError(
            f"SSRF blocked: Scheme '{parsed.scheme}' disallowed. Only http and https are allowed."
        )

    raw_hostname = parsed.hostname
    if not raw_hostname:
        raise ValueError("SSRF blocked: Missing or invalid hostname in URL")

    hostname = raw_hostname.strip("[]").strip()
    if not hostname:
        raise ValueError("SSRF blocked: Empty hostname in URL")

    # Resolve IP addresses (handles both IP literals and domain names)
    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    try:
        ip_literal = ipaddress.ip_address(hostname)
        ips.append(ip_literal)
    except ValueError:
        # Not a direct IP literal; resolve via DNS
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            for item in addr_info:
                sockaddr = item[4]
                if sockaddr:
                    ip_str = sockaddr[0]
                    ips.append(ipaddress.ip_address(ip_str))
        except socket.gaierror as e:
            raise ValueError(f"SSRF blocked: Could not resolve hostname '{hostname}': {e}") from e

    if not ips:
        raise ValueError(f"SSRF blocked: No IP addresses resolved for hostname '{hostname}'")

    for ip_obj in ips:
        # Handle IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1)
        if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped:
            ip_obj = ip_obj.ipv4_mapped

        if (
            ip_obj.is_loopback
            or ip_obj.is_private
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
            or ip_obj.is_unspecified
            or any(ip_obj in net for net in DISALLOWED_NETWORKS)
        ):
            raise ValueError(
                f"SSRF blocked: Access to private/loopback network disallowed ({ip_obj})"
            )

    return url


async def fetch_with_ssrf_guard(
    url: str,
    client: Optional[httpx.AsyncClient] = None,
    max_redirects: int = 3,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> httpx.Response:
    """Fetch an HTTP URL with strict per-hop SSRF validation across redirects."""
    current_url = validate_safe_url(url)
    redirect_count = 0
    current_method = method

    should_close_client = False
    active_client = client
    if active_client is None:
        active_client = httpx.AsyncClient(timeout=10.0)
        should_close_client = True

    try:
        while True:
            resp = await active_client.request(
                current_method,
                current_url,
                follow_redirects=False,
                headers=headers,
                **kwargs,
            )

            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                if not location:
                    return resp

                redirect_target = urllib.parse.urljoin(current_url, location)
                # Validates redirect target against SSRF rules before following
                current_url = validate_safe_url(redirect_target)
                redirect_count += 1
                if redirect_count > max_redirects:
                    raise ValueError(f"Too many redirects (max {max_redirects})")

                if resp.status_code == 303 or (
                    current_method != "GET" and resp.status_code in (301, 302)
                ):
                    current_method = "GET"
                continue

            return resp
    finally:
        if should_close_client and active_client is not None:
            await active_client.aclose()


class SafeAsyncClient:
    """Context-managed async HTTP client wrapper with built-in SSRF protection."""

    def __init__(
        self,
        client: Optional[httpx.AsyncClient] = None,
        max_redirects: int = 3,
        **client_kwargs: Any,
    ):
        self._external_client = client
        self._client_kwargs = client_kwargs
        self._max_redirects = max_redirects
        self._client: Optional[httpx.AsyncClient] = client

    async def __aenter__(self) -> SafeAsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(**self._client_kwargs)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._external_client is None and self._client is not None:
            await self._client.aclose()

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return await fetch_with_ssrf_guard(
            url, client=self._client, max_redirects=self._max_redirects, method="GET", **kwargs
        )

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return await fetch_with_ssrf_guard(
            url, client=self._client, max_redirects=self._max_redirects, method="POST", **kwargs
        )
