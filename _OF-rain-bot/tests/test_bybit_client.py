from __future__ import annotations

import unittest

from trading.infrastructure.bybit import AsyncBybitTradingClient, AsyncBybitTransport, BybitAPIError


class _FakeTransport:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict]] = []

    async def request(self, method: str, endpoint: str, payload: dict) -> dict:
        self.calls.append((method, endpoint, payload))
        if not self.responses:
            raise AssertionError("No fake response queued")
        return self.responses.pop(0)


class BybitClientTests(unittest.IsolatedAsyncioTestCase):
    def test_transport_uses_v5_auth_headers_and_body(self) -> None:
        transport = AsyncBybitTransport(
            api_endpoint="https://api-test.example",
            api_key="test-key",
            api_secret="test-secret",
            recv_window="5000",
        )

        headers, params, body = transport._request_parts(
            "POST",
            {"category": "linear", "symbol": "BTCUSDT", "stopLoss": "100.0", "tpslMode": "Full", "positionIdx": 0},
        )

        self.assertIn("X-BAPI-API-KEY", headers)
        self.assertIn("X-BAPI-TIMESTAMP", headers)
        self.assertIn("X-BAPI-RECV-WINDOW", headers)
        self.assertIn("X-BAPI-SIGN", headers)
        self.assertIsNone(params)
        self.assertIn("\"tpslMode\":\"Full\"", body)
        self.assertIn("\"positionIdx\":0", body)

    async def test_get_order_status_normalizes_payload(self) -> None:
        client = AsyncBybitTradingClient(
            transport=_FakeTransport(
                [
                    {
                        "retCode": 0,
                        "result": {
                            "list": [
                                {
                                    "orderId": "abc",
                                    "symbol": "BTCUSDT",
                                    "side": "Buy",
                                    "orderStatus": "Filled",
                                    "price": "100.0",
                                    "qty": "1",
                                    "cumExecQty": "1",
                                }
                            ]
                        },
                    }
                ]
            )
        )

        status = await client.get_order_status("BTCUSDT", "abc")

        self.assertEqual(
            status,
            {
                "order_id": "abc",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "status": "Filled",
                "price": "100.0",
                "qty": "1",
                "cum_exec_qty": "1",
            },
        )

    async def test_place_limit_order_if_absent_skips_existing_matching_order(self) -> None:
        transport = _FakeTransport(
            [
                {
                    "retCode": 0,
                    "result": {
                        "list": [
                            {
                                "orderId": "existing-1",
                                "symbol": "BTCUSDT",
                                "side": "Buy",
                                "orderType": "Limit",
                                "orderStatus": "New",
                                "price": "100.0",
                            }
                        ]
                    },
                }
            ]
        )
        client = AsyncBybitTradingClient(transport=transport)

        order_id = await client.place_limit_order_if_absent("BTCUSDT", "Buy", 1, 95, 110, 100.0)

        self.assertEqual(order_id, "existing-1")
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(transport.calls[0][1], "/v5/order/realtime")

    async def test_unwrap_result_raises_on_bybit_error(self) -> None:
        with self.assertRaises(BybitAPIError):
            AsyncBybitTradingClient._unwrap_result(
                {"retCode": 10001, "retMsg": "invalid request"},
                "open_order",
            )
