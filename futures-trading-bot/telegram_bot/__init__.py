__all__ = [
    "get_position_icon",
    "send_breakeven_message",
    "send_message",
    "send_pompilo_order_message",
    "test_run",
]


def __getattr__(name):
    """Lazily import Telegram helpers to keep package import safe without telegram dependency."""
    if name in {
        "get_position_icon",
        "send_breakeven_message",
        "send_message",
        "send_pompilo_order_message",
        "test_run",
    }:
        from .bot import (
            get_position_icon,
            send_breakeven_message,
            send_message,
            send_pompilo_order_message,
            test_run,
        )

        mapping = {
            "get_position_icon": get_position_icon,
            "send_breakeven_message": send_breakeven_message,
            "send_message": send_message,
            "send_pompilo_order_message": send_pompilo_order_message,
            "test_run": test_run,
        }
        return mapping[name]
    raise AttributeError(name)
