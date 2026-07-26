from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from bot_platform_service.domain import BotRunResult, BotRunStatus, BotStateChange, BotStateChangeOperation, StateStore

StateChangeKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class StateChangeApplicationResult:
    """Summary of platform-owned state changes applied for one run."""

    applied: int
    skipped: int


class StateChangeApplicationError(ValueError):
    """Raised when a returned state change cannot be applied safely."""


class StateChangeApplierService:
    """Apply returned bot state changes through the platform StateStore boundary."""

    async def apply(
        self,
        result: BotRunResult,
        *,
        state_store: StateStore,
        already_applied_keys: Iterable[StateChangeKey] = (),
    ) -> StateChangeApplicationResult:
        """Persist successful run state changes once per state key."""
        if result.status is not BotRunStatus.COMPLETE:
            return StateChangeApplicationResult(applied=0, skipped=len(result.state_changes))

        applied_keys = set(already_applied_keys)
        applied = 0
        skipped = 0
        for change in result.state_changes:
            key = state_change_key(change)
            if key in applied_keys:
                skipped += 1
                continue
            await _apply_state_change(change, state_store=state_store)
            applied_keys.add(key)
            applied += 1
        return StateChangeApplicationResult(applied=applied, skipped=skipped)


@dataclass(slots=True)
class RunScopedStateStore:
    """StateStore wrapper that records writes performed inside one bot run."""

    delegate: StateStore
    saved_keys: set[StateChangeKey] = field(default_factory=set)

    async def load(self, *, instance_id: str, namespace: str, state_key: str) -> object | None:
        return await self.delegate.load(instance_id=instance_id, namespace=namespace, state_key=state_key)

    async def save(self, *, instance_id: str, namespace: str, state_key: str, value: object) -> None:
        self.saved_keys.add((instance_id, namespace, state_key))
        await self.delegate.save(instance_id=instance_id, namespace=namespace, state_key=state_key, value=value)


def state_change_key(change: BotStateChange) -> StateChangeKey:
    """Return the idempotency key for a state change within one run."""
    return (change.instance_id, change.namespace, change.state_key)


async def _apply_state_change(change: BotStateChange, *, state_store: StateStore) -> None:
    if change.operation is not BotStateChangeOperation.UPSERT:
        raise StateChangeApplicationError("Only UPSERT state changes are supported")
    if change.value is None:
        raise StateChangeApplicationError("UPSERT state changes require a value")
    await state_store.save(
        instance_id=change.instance_id,
        namespace=change.namespace,
        state_key=change.state_key,
        value=change.value,
    )
