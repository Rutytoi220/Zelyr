"""Network security validation and SSRF protection for Zelyr."""

from __future__ import annotations

import ipaddress
import logging
import socket
import urllib.parse
from typing import Any, Dict, List, Optional

import httpcore
from httpcore._backends.auto import AutoBackend
import httpx

logger = logging.getLogger(__name__)


# Disallowed reserved / private subnets
DISALLOWED_NETWORKS = (
    ipaddress.ip_network("100.64.0.0/10"),    # RFC 6598 Carrier-Grade NAT / Tailscale private mesh
    ipaddress.ip_network("0.0.0.0/8"),        # RFC 1122 Current network / local host
    ipaddress.ip_network("2001:db8::/32"),    # RFC 3849 IPv6 Documentation
    ipaddress.ip_network("198.18.0.0/15"),    # RFC 2544 Benchmark testing
    ipaddress.ip_network("192.0.2.0/24"),     # RFC 5737 TEST-NET-1
    ipaddress.ip_network("198.51.100.0/24"),  # RFC 5737 TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),   # RFC 5737 TEST-NET-3
)


def is_safe_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Determine whether an IP address is safe and not in private/reserved ranges."""
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
        return False
    return True


def resolve_and_validate_host(hostname: str) -> List[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve a hostname or IP string and verify all resolved IPs against SSRF constraints."""
    if not hostname or not isinstance(hostname, str):
        raise ValueError("SSRF blocked: Hostname must be a non-empty string")

    clean_host = hostname.strip("[]").strip().rstrip(".").lower()
    if not clean_host:
        raise ValueError("SSRF blocked: Empty hostname in URL")

    # Reject cloud metadata hostnames immediately
    blocked_hostnames = {
        "metadata.google.internal",
        "metadata.internal",
        "metadata",
        "instance-data",
    }
    if clean_host in blocked_hostnames or clean_host.endswith(".google.internal"):
        raise ValueError(f"SSRF blocked: Cloud metadata endpoint disallowed ({clean_host})")

    ips: List[ipaddress.IPv4Address | ipaddress.IPv6Address] = []

    # 1. Direct IP literal
    try:
        ips.append(ipaddress.ip_address(clean_host))
    except ValueError:
        # 2. Integer or Hex IP literal (e.g. 2130706433 or 0x7f000001)
        if clean_host.isdigit():
            try:
                ips.append(ipaddress.IPv4Address(int(clean_host)))
            except (ValueError, OverflowError):
                pass
        elif clean_host.startswith("0x"):
            try:
                ips.append(ipaddress.IPv4Address(int(clean_host, 16)))
            except (ValueError, OverflowError):
                pass
        else:
            # 3. Octal / dotted quad notation with leading zeroes (e.g. 0177.0.0.1)
            parts = clean_host.split(".")
            if len(parts) == 4:
                try:
                    def _parse_octet(p: str) -> int:
                        if p.startswith(("0x", "0X")):
                            return int(p, 16)
                        if p.startswith("0") and len(p) > 1 and p.isdigit():
                            return int(p, 8)
                        return int(p, 10)

                    octets = [_parse_octet(p) for p in parts]
                    if all(0 <= o <= 255 for o in octets):
                        ips.append(ipaddress.IPv4Address(bytes(octets)))
                except (ValueError, OverflowError):
                    pass

        # 4. If not an obfuscated IP literal, resolve via DNS
        if not ips:
            try:
                addr_info = socket.getaddrinfo(clean_host, None)
                for item in addr_info:
                    sockaddr = item[4]
                    if sockaddr:
                        ip_str = sockaddr[0]
                        ips.append(ipaddress.ip_address(ip_str))
            except socket.gaierror as e:
                raise ValueError(f"SSRF blocked: Could not resolve hostname '{clean_host}': {e}") from e

    if not ips:
        raise ValueError(f"SSRF blocked: No IP addresses resolved for hostname '{clean_host}'")

    for ip_obj in ips:
        if not is_safe_ip(ip_obj):
            raise ValueError(
                f"SSRF blocked: Access to private/loopback network disallowed ({ip_obj})"
            )

    return ips


def validate_safe_url(url: str) -> str:
    """Validate that a URL is safe to fetch, enforcing strict SSRF protections.

    Rejects:
    - Schemes other than http and https
    - Missing or empty hostnames
    - Embedded credentials (user/password)
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

    if parsed.username or parsed.password:
        raise ValueError("SSRF blocked: Embedded credentials in URL are disallowed")

    raw_hostname = parsed.hostname
    if not raw_hostname:
        raise ValueError("SSRF blocked: Missing or invalid hostname in URL")

    # Perform DNS preflight and IP classification
    resolve_and_validate_host(raw_hostname)

    return url


class PinnedIPNetworkBackend(AutoBackend):
    """Network backend that resolves DNS, validates SSRF constraints, and connects directly to pinned IP."""

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> Any:
        safe_ips = resolve_and_validate_host(host)
        if not safe_ips:
            raise ValueError(f"SSRF blocked: No IP addresses resolved for hostname '{host}'")
        pinned_ip = str(safe_ips[0])
        return await super().connect_tcp(
            pinned_ip,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )


class PinnedIPTransport(httpx.AsyncHTTPTransport):
    """Async HTTP transport enforcing transport-level IP pinning to prevent DNS rebinding."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._backend = PinnedIPNetworkBackend()
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=self._pool._ssl_context,
            max_connections=self._pool._max_connections,
            max_keepalive_connections=self._pool._max_keepalive_connections,
            keepalive_expiry=self._pool._keepalive_expiry,
            http1=self._pool._http1,
            http2=self._pool._http2,
            network_backend=self._backend,
            retries=self._pool._retries,
        )


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
        active_client = httpx.AsyncClient(transport=PinnedIPTransport(), timeout=10.0)
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
            if "transport" not in self._client_kwargs:
                self._client_kwargs["transport"] = PinnedIPTransport()
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
