"""Offline budget replay primitives for LongMemEval-V2 cost fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique


@unique
class BudgetStatus(StrEnum):
    """Budget decision variants emitted by offline preflight replay."""

    APPROVED = "approved"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"
    SETTLED = "settled"
    BREACHED = "breached"


@dataclass(frozen=True, slots=True)
class ReservationRequest:
    """One offline preflight record, optionally tied to a settled cost event."""

    reservation_id: str
    estimate_micro_usd: int | None
    settlement_event_id: str | None


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """JSON-safe budget decision for report fixtures and future CLI output."""

    status: BudgetStatus
    reason: str
    reservation_id: str | None
    limit_micro_usd: int | None
    estimate_micro_usd: int | None
    spent_micro_usd: int | None
    reserved_micro_usd: int | None
    over_micro_usd: int | None


@dataclass(frozen=True, slots=True)
class BudgetState:
    """Immutable replay state with known spend, reservations, and unknown latches."""

    limit_micro_usd: int | None
    spent_micro_usd: int
    reserved_micro_usd: int
    has_unknown: bool = False
    blocked_over_micro_usd: int | None = None

    def decide_total(self, total_micro_usd: int | None) -> BudgetDecision:
        """Decide whether known final cost fits the offline cap."""
        if self.limit_micro_usd is None:
            return _decision(BudgetStatus.UNKNOWN, "budget_micro_usd is not set", self)
        if self.blocked_over_micro_usd is not None:
            return _decision(
                BudgetStatus.BLOCKED,
                "budget preflight or settlement was blocked",
                self,
                over=self.blocked_over_micro_usd,
            )
        if self.has_unknown or total_micro_usd is None:
            return _decision(BudgetStatus.UNKNOWN, "cost or reservation coverage is unknown", self)
        exposure = max(total_micro_usd, self.spent_micro_usd) + self.reserved_micro_usd
        over = exposure - self.limit_micro_usd
        if over > 0:
            return _decision(
                BudgetStatus.BLOCKED,
                "known spend plus reservations exceeds budget",
                self,
                over=over,
            )
        return _decision(
            BudgetStatus.APPROVED,
            "known spend plus reservations is within budget",
            self,
        )

    def apply(
        self, request: ReservationRequest, settlement_cost_micro_usd: int | None
    ) -> tuple[BudgetState, BudgetDecision]:
        """Replay one reservation against this offline state."""
        if self.limit_micro_usd is None:
            state = BudgetState(
                None,
                self.spent_micro_usd,
                self.reserved_micro_usd,
                True,
                self.blocked_over_micro_usd,
            )
            return state, _decision(
                BudgetStatus.UNKNOWN,
                "budget_micro_usd is not set",
                state,
                request=request,
            )
        limit = self.limit_micro_usd
        if self.has_unknown:
            return self, _decision(
                BudgetStatus.UNKNOWN,
                "earlier reservation has unknown cost",
                self,
                request=request,
            )
        estimate = request.estimate_micro_usd
        if estimate is None:
            state = BudgetState(
                limit,
                self.spent_micro_usd,
                self.reserved_micro_usd,
                True,
                self.blocked_over_micro_usd,
            )
            return state, _decision(
                BudgetStatus.UNKNOWN,
                "reservation estimate is unknown",
                state,
                request=request,
            )
        projected = self.spent_micro_usd + self.reserved_micro_usd + estimate
        over = projected - limit
        if over > 0:
            state = BudgetState(
                limit,
                self.spent_micro_usd,
                self.reserved_micro_usd,
                self.has_unknown,
                _max_over(self.blocked_over_micro_usd, over),
            )
            return state, _decision(
                BudgetStatus.BLOCKED,
                "insufficient remaining budget",
                state,
                request=request,
                over=over,
            )
        reserved_state = BudgetState(
            limit,
            self.spent_micro_usd,
            self.reserved_micro_usd + estimate,
            self.has_unknown,
            self.blocked_over_micro_usd,
        )
        if request.settlement_event_id is None:
            return reserved_state, _decision(
                BudgetStatus.APPROVED,
                "reservation accepted",
                reserved_state,
                request=request,
            )
        if settlement_cost_micro_usd is None:
            state = BudgetState(
                limit,
                self.spent_micro_usd,
                self.reserved_micro_usd,
                True,
                self.blocked_over_micro_usd,
            )
            return state, _decision(
                BudgetStatus.UNKNOWN,
                "settlement cost is unknown",
                state,
                request=request,
            )
        settled_state = BudgetState(
            limit,
            self.spent_micro_usd + settlement_cost_micro_usd,
            self.reserved_micro_usd,
            self.has_unknown,
            self.blocked_over_micro_usd,
        )
        settled_over = settled_state.spent_micro_usd + settled_state.reserved_micro_usd - limit
        status = BudgetStatus.BREACHED if settled_over > 0 else BudgetStatus.SETTLED
        reason = (
            "settlement breached budget exposure"
            if settled_over > 0
            else "reservation settled with actual event cost"
        )
        final_state = BudgetState(
            limit,
            settled_state.spent_micro_usd,
            settled_state.reserved_micro_usd,
            settled_state.has_unknown,
            _max_over(self.blocked_over_micro_usd, settled_over)
            if settled_over > 0
            else self.blocked_over_micro_usd,
        )
        return final_state, _decision(
            status,
            reason,
            final_state,
            request=request,
            spent=settlement_cost_micro_usd,
            over=settled_over if settled_over > 0 else None,
        )


def _decision(
    status: BudgetStatus,
    reason: str,
    state: BudgetState,
    request: ReservationRequest | None = None,
    spent: int | None = None,
    over: int | None = None,
) -> BudgetDecision:
    return BudgetDecision(
        status=status,
        reason=reason,
        reservation_id=None if request is None else request.reservation_id,
        limit_micro_usd=state.limit_micro_usd,
        estimate_micro_usd=None if request is None else request.estimate_micro_usd,
        spent_micro_usd=state.spent_micro_usd if spent is None else spent,
        reserved_micro_usd=state.reserved_micro_usd,
        over_micro_usd=over,
    )


def _max_over(current: int | None, new: int) -> int:
    if current is None:
        return new
    return max(current, new)
