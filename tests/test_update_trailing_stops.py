from types import SimpleNamespace

from scripts.update_trailing_stops import update_trailing_stops


def test_update_trailing_stops(monkeypatch):
    submitted = []

    class DummyClient:
        def get_all_positions(self):
            return [
                SimpleNamespace(symbol="AAPL", qty="5", side="long"),
                SimpleNamespace(symbol="TSLA", qty="3", side="short"),
            ]

    def fake_get_client(*, paper=True):
        return DummyClient()

    def fake_cancel_all_orders(client):
        return None

    def fake_submit_order(client, symbol, qty, *, side, order_type, trail_percent, **_):
        submitted.append((symbol, qty, side, order_type, trail_percent))

    monkeypatch.setattr("scripts.update_trailing_stops.ba.get_client", fake_get_client)
    monkeypatch.setattr("scripts.update_trailing_stops.ba.cancel_all_orders", fake_cancel_all_orders)
    monkeypatch.setattr("scripts.update_trailing_stops.ba.submit_order", fake_submit_order)

    mapping = {"AAPL": 25.0, "TSLA": 20.0}
    update_trailing_stops(symbol_trail_pct=mapping)

    assert submitted == [
        ("AAPL", 5, "sell", "trailing_stop", 25.0),
        ("TSLA", 3, "buy", "trailing_stop", 20.0),
    ]
