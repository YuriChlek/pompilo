from __future__ import annotations


async def dispatch_command(args, handlers) -> None:
    """Route parsed CLI arguments to the matching async command handler."""

    if args.command == "sync":
        await handlers.sync(days=args.period)
        return
    if args.command == "sync-3y":
        await handlers.sync_3y()
        return
    if args.command == "sync-4h":
        await handlers.sync_4h(days=args.period)
        return
    if args.command == "sync-full":
        await handlers.sync_full()
        return
    if args.command == "analyze":
        await handlers.analyze(symbol=args.symbol, dry_run=args.dry_run, timeframe=args.timeframe)
        return
    if args.command == "init-db":
        await handlers.init_db()
        return
    if args.command == "migrate":
        await handlers.migrate()
        return
    await handlers.live(dry_run=args.dry_run)


__all__ = ["dispatch_command"]
