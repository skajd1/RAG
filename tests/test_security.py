import unittest

import _paths  # noqa: F401

from security import client_ip_from_headers, configured_allowed_networks, is_ip_allowed


class InternalNetworkSecurityTests(unittest.TestCase):
    def test_default_allowed_networks_include_loopback_and_internal_gateway_subnet(self):
        networks = configured_allowed_networks()

        self.assertTrue(is_ip_allowed("127.0.0.1", networks))
        self.assertTrue(is_ip_allowed("::1", networks))
        self.assertTrue(is_ip_allowed("10.0.0.1", networks))
        self.assertFalse(is_ip_allowed("10.0.1.1", networks))
        self.assertFalse(is_ip_allowed("8.8.8.8", networks))

    def test_configured_allowed_networks_support_custom_cidr_values(self):
        networks = configured_allowed_networks("10.0.0.0/8,192.168.10.20/32")

        self.assertTrue(is_ip_allowed("10.20.30.40", networks))
        self.assertTrue(is_ip_allowed("192.168.10.20", networks))
        self.assertFalse(is_ip_allowed("192.168.10.21", networks))

    def test_client_ip_prefers_first_x_forwarded_for_value(self):
        headers = {"x-forwarded-for": "10.0.0.88, 10.0.0.1"}

        self.assertEqual(client_ip_from_headers(headers, "127.0.0.1"), "10.0.0.88")

    def test_client_ip_falls_back_to_request_host(self):
        self.assertEqual(client_ip_from_headers({}, "127.0.0.1"), "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
