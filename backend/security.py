import ipaddress
import os


DEFAULT_ALLOWED_CIDRS = "127.0.0.1/32,::1/128,10.0.0.0/24"


def configured_allowed_networks(value: str | None = None):
    raw_value = value if value is not None else os.getenv("INTERNAL_ALLOWED_CIDRS", DEFAULT_ALLOWED_CIDRS)
    networks = []
    for raw_cidr in raw_value.split(","):
        cidr = raw_cidr.strip()
        if not cidr:
            continue
        networks.append(ipaddress.ip_network(cidr, strict=False))
    return networks


def client_ip_from_headers(headers, fallback_host: str | None):
    forwarded_for = headers.get("x-forwarded-for") if headers else None
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return fallback_host or ""


def is_ip_allowed(ip_value: str, networks) -> bool:
    try:
        client_ip = ipaddress.ip_address(ip_value)
    except ValueError:
        return False
    return any(client_ip in network for network in networks)
