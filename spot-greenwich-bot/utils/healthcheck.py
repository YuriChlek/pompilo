from __future__ import annotations

import logging
from datetime import datetime, timezone

from utils.config import HEALTHCHECK_ENABLED, HEALTHCHECK_HOST, HEALTHCHECK_PORT

logger = logging.getLogger(__name__)

_last_cycle_ts: str | None = None


def mark_cycle_completed() -> None:
    global _last_cycle_ts
    _last_cycle_ts = datetime.now(tz=timezone.utc).isoformat()


async def start_healthcheck_server() -> None:
    if not HEALTHCHECK_ENABLED:
        logger.info("healthcheck_disabled")
        return

    try:
        from aiohttp import web
    except ModuleNotFoundError:
        logger.warning("healthcheck_unavailable reason=aiohttp_missing")
        return

    async def health(request):
        return web.json_response({"status": "ok", "last_cycle": _last_cycle_ts})

    app = web.Application()
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HEALTHCHECK_HOST, HEALTHCHECK_PORT)
    await site.start()
    logger.info("healthcheck_started host=%s port=%s", HEALTHCHECK_HOST, HEALTHCHECK_PORT)
