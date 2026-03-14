"""Memorize correct spells from the spellbook.

Intelligent approach: reads spellbook and gem arrays from memory to determine
exactly what's scribed, what's memorized, and what needs to change. Only
clears and memorizes the specific gems that are wrong. Falls back to lower-tier
spells when the optimal spell isn't scribed.

Non-blocking: all UI interactions use a phase state machine. Each tick()
performs at most one action + sets a _resume_after timestamp, then returns
RUNNING so the brain can evaluate FLEE between every wait.

Flow:
1. Read spellbook from memory -> know all scribed spells
2. Compute desired loadout filtered by scribed spells (auto-fallback)
3. Read current memorized gems -> compare to desired
4. If all correct, skip entirely (instant)
5. If changes needed: clear wrong gems, navigate directly to spell page/slot
6. Memorize, verify, reconfigure
"""

from __future__ import annotations

import logging
import time
import tomllib
from collections.abc import Callable
from enum import Enum, auto
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from brain.context import AgentContext
    from perception.reader import MemoryReader
    from perception.state import GameState

from core.timing import interruptible_sleep
from core.types import ReadStateFn
from eq.loadout import (
    compute_desired_loadout,
    configure_from_memory,
    configure_loadout,
    get_spell_db,
)
from eq.spells import SpellDB
from motor.actions import (
    get_game_window_rect,
    left_click_at,
    right_click_at,
    sit,
    stand,
    toggle_spellbook,
)
from routines.base import RoutineBase, RoutineStatus, make_flee_predicate

log = logging.getLogger(__name__)


def compute_gem_changes(
    scribed: set[int],
    current_gems: dict[int, int],
    desired: dict[int, int],
    spellbook_slot_fn: Callable[[int], int | None] | None = None,
    db: SpellDB | None = None,
) -> tuple[list[tuple[int, int, str, int | None]], list[tuple[int, int, str]]]:
    """Pure function: compute which spells need memorizing.

    Args:
        scribed: set of spell IDs in spellbook
        current_gems: {gem_slot: spell_id} currently memorized
        desired: {gem_slot: spell_id} desired loadout (filtered to scribed)
        spellbook_slot_fn: callable(spell_id) -> slot_index or None
        db: SpellDB for name resolution

    Returns:
        (memorizable, not_scribed) where:
        - memorizable: [(gem, spell_id, name, book_slot)]
        - not_scribed: [(gem, spell_id, name)]
    """
    desired_ids = set(desired.values())
    current_ids = set(current_gems.values())

    if desired_ids and desired_ids.issubset(current_ids):
        return [], []  # all correct

    missing_ids = desired_ids - current_ids
    to_memorize = []
    for spell_id in sorted(missing_ids):
        name = f"spell_{spell_id}"
        if db is not None:
            sd = db.get(spell_id)
            if sd:
                name = sd.name
        book_slot = spellbook_slot_fn(spell_id) if spellbook_slot_fn else None
        gem = next((g for g, s in desired.items() if s == spell_id), 1)
        to_memorize.append((gem, spell_id, name, book_slot))

    memorizable: list[tuple[int, int, str, int | None]] = [
        (g, sid, n, s) for g, sid, n, s in to_memorize if s is not None
    ]
    not_scribed = [(g, sid, n) for g, sid, n, s in to_memorize if s is None]
    return memorizable, not_scribed


# -- UI element offsets --
_GEM_X = 20
_GEM_Y_START = 16
_GEM_Y_SPACE = 33

# Spell slot positions from EQUI_SpellBookWnd.xml (relative to window)
# 16 slots visible: left page (0-7) + right page (8-15)
# Add ~26px to center click on the 44x44 icon
_BOOK_SLOT_POS = [
    (106, 50),
    (197, 50),  # slots 0-1  (left page row 1)
    (106, 123),
    (197, 123),  # slots 2-3  (left page row 2)
    (106, 195),
    (197, 195),  # slots 4-5  (left page row 3)
    (106, 268),
    (197, 268),  # slots 6-7  (left page row 4)
    (318, 50),
    (409, 50),  # slots 8-9  (right page row 1)
    (318, 123),
    (409, 123),  # slots 10-11 (right page row 2)
    (318, 195),
    (409, 195),  # slots 12-13 (right page row 3)
    (318, 268),
    (409, 268),  # slots 14-15 (right page row 4)
]

_BOOK_NEXT_X = 468
_BOOK_NEXT_Y = 166
_BOOK_PREV_X = 468
_BOOK_PREV_Y = 142

MEMORIZE_WAIT = 11.0
CAST_CHECK_DELAY = 1.5
SLOTS_PER_PAGE = 16  # EQ shows 2 pages (left+right) = 16 slots per view


# -- Phase state machine for non-blocking tick() --


class _Phase(Enum):
    IDLE = auto()  # Initial: analyze spellbook and compute work
    CLEAR_GEMS = auto()  # Right-click a gem to clear it
    CLEAR_GEM_WAIT = auto()  # Wait after clearing a gem
    SIT = auto()  # Sit down if not sitting
    SIT_WAIT = auto()  # Wait for sit animation
    OPEN_BOOK = auto()  # Toggle spellbook open
    OPEN_BOOK_WAIT = auto()  # Wait for spellbook to render
    RESET_PAGES = auto()  # Click prev button to go to page 0
    RESET_PAGE_WAIT = auto()  # Wait between prev-page clicks
    RESET_PAGES_DONE = auto()  # Final wait after all prev clicks
    NAV_PAGE = auto()  # Click next/prev to reach target page
    NAV_PAGE_WAIT = auto()  # Wait between nav clicks
    NAV_PAGES_DONE = auto()  # Final wait after page navigation
    CLICK_SLOT = auto()  # Click spell slot in spellbook
    CLICK_SLOT_WAIT = auto()  # Wait after slot click
    CLICK_GEM = auto()  # Click gem to assign spell
    CAST_CHECK_WAIT = auto()  # Wait CAST_CHECK_DELAY then check
    WAITING_MEMORIZE = auto()  # Wait MEMORIZE_WAIT for memorization
    CLOSE_BOOK = auto()  # Toggle spellbook closed
    CLOSE_BOOK_WAIT = auto()  # Wait after closing
    STAND_UP = auto()  # Stand up
    STAND_WAIT = auto()  # Wait for stand, then reconfigure


def _detect_resolution(eq_rect: tuple[int, int, int, int]) -> str:
    w = eq_rect[2] - eq_rect[0]
    h = eq_rect[3] - eq_rect[1]
    return f"{w}x{h}"


def _read_wnd_pos(
    wnd_name: str, resolution: str, client_path: str = "", default: tuple[int, int] = (0, 0)
) -> tuple[int, int]:
    return default


def _gem_screen_pos(
    gem_index: int, eq_rect: tuple[int, int, int, int], cast_wnd: tuple[int, int]
) -> tuple[int, int]:
    return (
        eq_rect[0] + cast_wnd[0] + _GEM_X,
        eq_rect[1] + cast_wnd[1] + _GEM_Y_START + gem_index * _GEM_Y_SPACE,
    )


def _book_slot_screen_pos(
    slot: int, eq_rect: tuple[int, int, int, int], book_wnd: tuple[int, int]
) -> tuple[int, int]:
    """Map slot index (0-15) to screen position in the spellbook.

    Coordinates from EQUI_SpellBookWnd.xml. EQ shows 2 pages (16 slots).
    """
    if slot < 0 or slot >= len(_BOOK_SLOT_POS):
        slot = 0  # safety fallback
    ox, oy = _BOOK_SLOT_POS[slot]
    return (eq_rect[0] + book_wnd[0] + ox, eq_rect[1] + book_wnd[1] + oy)


def _page_btn_screen_pos(
    direction: str, eq_rect: tuple[int, int, int, int], book_wnd: tuple[int, int]
) -> tuple[int, int]:
    if direction == "next":
        return (eq_rect[0] + book_wnd[0] + _BOOK_NEXT_X, eq_rect[1] + book_wnd[1] + _BOOK_NEXT_Y)
    return (eq_rect[0] + book_wnd[0] + _BOOK_PREV_X, eq_rect[1] + book_wnd[1] + _BOOK_PREV_Y)


def _load_client_path() -> str:
    """Load EQ path from config, returning empty string on failure."""
    try:
        with open("config/settings.toml", "rb") as f:
            return str(tomllib.load(f).get("general", {}).get("client_path", ""))
    except (
        OSError,
        tomllib.TOMLDecodeError,
    ):
        return ""


class MemorizeSpellsRoutine(RoutineBase):
    """Intelligent spell memorization using memory reads.

    Fully non-blocking: tick() never sleeps. Uses a phase state machine with
    _resume_after timestamps so the brain can evaluate FLEE between every wait.
    """

    def __init__(self, ctx: AgentContext | None = None, read_state_fn: ReadStateFn | None = None) -> None:
        self._ctx = ctx
        self._read_state_fn = read_state_fn
        self._book_open = False
        self._phase = _Phase.IDLE
        self._resume_after = 0.0
        # Work queue: list of (gem, spell_id, name, book_slot)
        self._memorizable: list[tuple[int, int, str, int | None]] = []
        self._spell_index = 0  # index into _memorizable for current spell
        self._current_page = 0  # current spellbook page
        # For page navigation sub-loops
        self._page_clicks_remaining = 0
        self._page_direction = "next"
        # Reset-pages sub-loop (go to page 0 on open)
        self._reset_clicks_remaining = 0
        # Screen coordinate caches (set once per memorization session)
        self._eq_rect: tuple[int, int, int, int] | None = None
        self._cast_wnd: tuple[int, int] = (3, 2)
        self._book_wnd: tuple[int, int] = (232, 154)
        # Gem clearing sub-loop
        self._gems_to_clear: list[int] = []
        self._clear_index = 0
        # Current memorized gems (for clearing check)
        self._current_gems: dict[int, int] = {}
        # True when RESET_PAGES is resetting before CLOSE_BOOK (not before memorize)
        self._closing = False
        self._flee_check: Callable[[], bool] | None = None

    def _reader(self) -> MemoryReader | None:
        return getattr(self._ctx, "reader", None) if self._ctx else None

    @override
    def enter(self, state: GameState) -> None:
        self._book_open = False
        self._phase = _Phase.IDLE
        self._resume_after = 0.0
        self._memorizable = []
        self._spell_index = 0
        self._current_page = 0
        self._page_clicks_remaining = 0
        self._reset_clicks_remaining = 0
        self._gems_to_clear = []
        self._clear_index = 0
        self._current_gems = {}
        self._closing = False
        if self._read_state_fn and self._ctx:
            self._flee_check = make_flee_predicate(self._read_state_fn, self._ctx)
        else:
            self._flee_check = None
        log.info("[CAST] MemSpells: checking spell loadout")

    @override
    def tick(self, state: GameState) -> RoutineStatus:
        reader = self._reader()
        if not reader or not self._read_state_fn:
            log.warning("[CAST] MemSpells: no reader/state function")
            return RoutineStatus.FAILURE

        # -- Timer gate: if we're waiting, return RUNNING immediately --
        if self._resume_after > 0.0 and time.time() < self._resume_after:
            return RoutineStatus.RUNNING

        # -- Phase dispatch --
        handler = self._PHASE_DISPATCH.get(self._phase)
        if handler is not None:
            return handler(self, state, reader)

        log.error("[CAST] MemSpells: unknown phase %s", self._phase)
        return RoutineStatus.FAILURE

    # -- Wait-phase transition handlers (uniform signature for dispatch) -----

    def _on_clear_gem_wait(self, state: GameState, reader: MemoryReader) -> RoutineStatus:
        self._clear_index += 1
        self._phase = _Phase.CLEAR_GEMS
        return RoutineStatus.RUNNING

    def _on_sit_wait(self, state: GameState, reader: MemoryReader) -> RoutineStatus:
        log.debug("[CAST] MemSpells: sit complete, opening book")
        self._phase = _Phase.OPEN_BOOK
        return RoutineStatus.RUNNING

    def _on_open_book_wait(self, state: GameState, reader: MemoryReader) -> RoutineStatus:
        self._reset_clicks_remaining = 5
        self._phase = _Phase.RESET_PAGES
        return RoutineStatus.RUNNING

    def _on_reset_page_wait(self, state: GameState, reader: MemoryReader) -> RoutineStatus:
        self._reset_clicks_remaining -= 1
        self._phase = _Phase.RESET_PAGES
        return RoutineStatus.RUNNING

    def _on_reset_pages_done(self, state: GameState, reader: MemoryReader) -> RoutineStatus:
        self._current_page = 0
        if self._closing:
            self._closing = False
            self._phase = _Phase.CLOSE_BOOK
        elif self._gems_to_clear:
            self._spell_index = 0
            self._page_clicks_remaining = 0
            self._phase = _Phase.CLEAR_GEMS
        else:
            self._spell_index = 0
            self._page_clicks_remaining = 0
            self._phase = _Phase.NAV_PAGE
        return RoutineStatus.RUNNING

    def _on_nav_page_wait(self, state: GameState, reader: MemoryReader) -> RoutineStatus:
        self._page_clicks_remaining -= 1
        self._phase = _Phase.NAV_PAGE
        return RoutineStatus.RUNNING

    def _on_nav_pages_done(self, state: GameState, reader: MemoryReader) -> RoutineStatus:
        self._phase = _Phase.CLICK_SLOT
        return RoutineStatus.RUNNING

    def _on_click_slot_wait(self, state: GameState, reader: MemoryReader) -> RoutineStatus:
        self._phase = _Phase.CLICK_GEM
        return RoutineStatus.RUNNING

    def _on_close_book_wait(self, state: GameState, reader: MemoryReader) -> RoutineStatus:
        self._phase = _Phase.STAND_UP
        return RoutineStatus.RUNNING

    # -- Phase dispatch table (class-level, references unbound methods) -----
    # All handlers accept (self, state, reader) for uniform dispatch.

    _PHASE_DISPATCH: dict[_Phase, Callable[..., RoutineStatus]] = {
        _Phase.IDLE: lambda self, st, rd: self._tick_idle(st, rd),
        _Phase.CLEAR_GEMS: lambda self, st, rd: self._tick_clear_gems(),
        _Phase.CLEAR_GEM_WAIT: _on_clear_gem_wait,
        _Phase.SIT: lambda self, st, rd: self._tick_sit(),
        _Phase.SIT_WAIT: _on_sit_wait,
        _Phase.OPEN_BOOK: lambda self, st, rd: self._tick_open_book(),
        _Phase.OPEN_BOOK_WAIT: _on_open_book_wait,
        _Phase.RESET_PAGES: lambda self, st, rd: self._tick_reset_pages(),
        _Phase.RESET_PAGE_WAIT: _on_reset_page_wait,
        _Phase.RESET_PAGES_DONE: _on_reset_pages_done,
        _Phase.NAV_PAGE: lambda self, st, rd: self._tick_nav_page(),
        _Phase.NAV_PAGE_WAIT: _on_nav_page_wait,
        _Phase.NAV_PAGES_DONE: _on_nav_pages_done,
        _Phase.CLICK_SLOT: lambda self, st, rd: self._tick_click_slot(),
        _Phase.CLICK_SLOT_WAIT: _on_click_slot_wait,
        _Phase.CLICK_GEM: lambda self, st, rd: self._tick_click_gem(),
        _Phase.CAST_CHECK_WAIT: lambda self, st, rd: self._tick_cast_check(),
        _Phase.WAITING_MEMORIZE: lambda self, st, rd: self._tick_waiting_memorize(),
        _Phase.CLOSE_BOOK: lambda self, st, rd: self._tick_close_book(),
        _Phase.CLOSE_BOOK_WAIT: _on_close_book_wait,
        _Phase.STAND_UP: lambda self, st, rd: self._tick_stand_up(),
        _Phase.STAND_WAIT: lambda self, st, rd: self._tick_stand_wait(st),
    }

    # -- Phase handlers --------------------------------------------------------

    def _tick_idle(self, state: GameState, reader: MemoryReader) -> RoutineStatus:
        """Analyze spellbook, compute work. Pure computation, no sleeps."""
        db = get_spell_db()

        # -- 1. Read spellbook from memory --
        scribed = reader.read_spellbook()
        log.info(
            "[CAST] MemSpells: spellbook has %d scribed spells: %s",
            len(scribed),
            ", ".join(sd.name if (sd := db.get(s)) else str(s) for s in sorted(scribed)),
        )

        # -- 2. Compute achievable loadout --
        configure_loadout(state.class_id, state.level, db=db, scribed_ids=scribed)
        desired = compute_desired_loadout(state.class_id, state.level, db=db, scribed_ids=scribed)

        desired_filtered = {g: sid for g, sid in desired.items() if sid in scribed}
        if not desired_filtered:
            log.warning("[CAST] MemSpells: no achievable loadout (nothing scribed?)")
            return RoutineStatus.FAILURE

        # -- 3. Compare gems (delegated to pure function) --
        current_gems = reader.read_memorized_spells()
        self._current_gems = current_gems

        memorizable, not_scribed = compute_gem_changes(
            scribed=scribed,
            current_gems=current_gems,
            desired=desired_filtered,
            spellbook_slot_fn=reader.spellbook_slot_for,
            db=db,
        )

        if not memorizable and not not_scribed:
            log.info("[CAST] MemSpells: all desired spells already memorized")
            configure_from_memory(current_gems, state.class_id)
            if self._ctx:
                self._ctx.plan.active = None
            return RoutineStatus.SUCCESS

        for gem, sid, name in not_scribed:
            log.warning("[CAST] MemSpells: '%s' NOT in spellbook -- skipping", name)
        for gem, sid, name, slot in memorizable:
            page = slot // SLOTS_PER_PAGE if slot is not None else -1
            pos = slot % SLOTS_PER_PAGE if slot is not None else -1
            log.info("[CAST] MemSpells: need gem %d = %s (page %d slot %d)", gem, name, page, pos)

        if not memorizable:
            log.info("[CAST] MemSpells: nothing memorizable")
            configure_from_memory(current_gems, state.class_id)
            if self._ctx:
                self._ctx.plan.active = None
            return RoutineStatus.SUCCESS

        # -- 6. Get screen coordinates (memory-first, INI fallback) --
        eq_rect = get_game_window_rect()
        if not eq_rect:
            log.error("[CAST] MemSpells: can't find target window")
            return RoutineStatus.FAILURE

        from perception import offsets as _ofs

        cast_wnd = reader.read_window_pos(_ofs.CAST_SPELL_WND_PTR)
        book_wnd = reader.read_window_pos(_ofs.SPELL_BOOK_WND_PTR)
        source = "memory"

        if cast_wnd is None or book_wnd is None:
            # Fallback to INI config
            client_path = _load_client_path()
            resolution = _detect_resolution(eq_rect)
            if cast_wnd is None:
                cast_wnd = _read_wnd_pos("CastSpellWnd", resolution, client_path, default=(3, 2))
            if book_wnd is None:
                book_wnd = _read_wnd_pos("SpellBookWnd", resolution, client_path, default=(232, 154))
            source = "INI fallback"

        log.info(
            "[CAST] MemSpells: eq_rect=%s cast_wnd=%s book_wnd=%s (%s)", eq_rect, cast_wnd, book_wnd, source
        )

        # Cache coordinates for entire session
        self._eq_rect = eq_rect
        self._cast_wnd = cast_wnd
        self._book_wnd = book_wnd
        self._memorizable = memorizable

        # Build gem-clear list: gems that currently have a spell assigned.
        # Clearing happens AFTER the spellbook is open (RESET_PAGES_DONE),
        # because right-clicking a gem while standing CASTS the spell
        # instead of clearing it.
        self._gems_to_clear = [gem for gem, _, _, _ in memorizable if gem in current_gems]
        self._clear_index = 0
        self._phase = _Phase.SIT
        return RoutineStatus.RUNNING

    def _tick_clear_gems(self) -> RoutineStatus:
        """Right-click one gem to clear it, then wait."""
        if self._clear_index >= len(self._gems_to_clear):
            # All gems cleared (book is open), proceed to memorize
            log.debug("[CAST] MemSpells: all gems cleared")
            self._phase = _Phase.NAV_PAGE
            return RoutineStatus.RUNNING

        gem = self._gems_to_clear[self._clear_index]
        assert self._eq_rect is not None
        gx, gy = _gem_screen_pos(gem - 1, self._eq_rect, self._cast_wnd)
        log.info("[CAST] MemSpells: clearing gem %d", gem)
        right_click_at(gx, gy)
        self._phase = _Phase.CLEAR_GEM_WAIT
        self._resume_after = time.time() + 0.3
        return RoutineStatus.RUNNING

    def _tick_sit(self) -> RoutineStatus:
        """Sit down if not sitting, then wait for animation."""
        if self._read_state_fn:
            current = self._read_state_fn()
            if not current.is_sitting:
                log.debug("[CAST] MemSpells: sitting down")
                sit()
                self._phase = _Phase.SIT_WAIT
                self._resume_after = time.time() + 1.0
                return RoutineStatus.RUNNING
        # Already sitting or no state fn, skip wait
        log.debug("[CAST] MemSpells: already sitting, proceeding to open book")
        self._phase = _Phase.OPEN_BOOK
        return RoutineStatus.RUNNING

    def _tick_open_book(self) -> RoutineStatus:
        """Toggle spellbook open."""
        log.info("[CAST] MemSpells: opening spellbook")
        toggle_spellbook()
        self._book_open = True
        self._phase = _Phase.OPEN_BOOK_WAIT
        self._resume_after = time.time() + 1.5
        return RoutineStatus.RUNNING

    def _tick_reset_pages(self) -> RoutineStatus:
        """Click prev button to return to page 0 (one click per tick)."""
        if self._reset_clicks_remaining <= 0:
            # Done resetting, wait a moment then start memorizing
            self._phase = _Phase.RESET_PAGES_DONE
            self._resume_after = time.time() + 0.5
            return RoutineStatus.RUNNING

        assert self._eq_rect is not None
        px, py = _page_btn_screen_pos("prev", self._eq_rect, self._book_wnd)
        left_click_at(px, py)
        self._phase = _Phase.RESET_PAGE_WAIT
        self._resume_after = time.time() + 0.2
        return RoutineStatus.RUNNING

    def _tick_nav_page(self) -> RoutineStatus:
        """Navigate to the target page for the current spell."""
        if self._spell_index >= len(self._memorizable):
            # All spells processed -- reset pages to 0 then close book
            log.info("[CAST] MemSpells: all spells memorized (or attempted)")
            if self._current_page > 0:
                self._closing = True
                self._reset_clicks_remaining = self._current_page
                self._phase = _Phase.RESET_PAGES
            else:
                self._phase = _Phase.CLOSE_BOOK
            return RoutineStatus.RUNNING

        gem, spell_id, name, book_slot = self._memorizable[self._spell_index]
        assert book_slot is not None  # filtered in compute_gem_changes
        target_page = book_slot // SLOTS_PER_PAGE

        if self._page_clicks_remaining > 0:
            # Still navigating pages
            assert self._eq_rect is not None
            btn_pos = _page_btn_screen_pos(self._page_direction, self._eq_rect, self._book_wnd)
            left_click_at(btn_pos[0], btn_pos[1])
            self._page_clicks_remaining -= 1
            if self._page_clicks_remaining > 0:
                self._phase = _Phase.NAV_PAGE_WAIT
                self._resume_after = time.time() + 0.3
            else:
                # Last page click done, wait before proceeding
                self._phase = _Phase.NAV_PAGES_DONE
                self._resume_after = time.time() + 0.3
            return RoutineStatus.RUNNING

        if target_page != self._current_page:
            # Need to navigate: set up click count and direction
            if target_page > self._current_page:
                self._page_direction = "next"
                self._page_clicks_remaining = target_page - self._current_page
            else:
                self._page_direction = "prev"
                self._page_clicks_remaining = self._current_page - target_page
            self._current_page = target_page
            # Do first click on next tick entry
            return self._tick_nav_page()

        # Already on correct page, click the slot
        self._phase = _Phase.CLICK_SLOT
        return RoutineStatus.RUNNING

    def _refresh_window_pos(self) -> None:
        """Re-read window positions from live memory before each click."""
        reader = self._reader()
        if not reader:
            return
        from perception import offsets as _ofs

        cast = reader.read_window_pos(_ofs.CAST_SPELL_WND_PTR)
        book = reader.read_window_pos(_ofs.SPELL_BOOK_WND_PTR)
        eq_rect = get_game_window_rect()
        if cast:
            self._cast_wnd = cast
        if book:
            self._book_wnd = book
        if eq_rect:
            self._eq_rect = eq_rect

    def _tick_click_slot(self) -> RoutineStatus:
        """Click the spell slot in the spellbook."""
        self._refresh_window_pos()
        gem, spell_id, name, book_slot = self._memorizable[self._spell_index]
        assert book_slot is not None  # filtered in compute_gem_changes
        target_page = book_slot // SLOTS_PER_PAGE
        target_slot = book_slot % SLOTS_PER_PAGE

        ox, oy = _BOOK_SLOT_POS[target_slot] if target_slot < len(_BOOK_SLOT_POS) else (0, 0)
        assert self._eq_rect is not None
        sx, sy = _book_slot_screen_pos(target_slot, self._eq_rect, self._book_wnd)
        log.info(
            "[CAST] MemSpells: CLICK book slot %d (page %d) '%s' -> "
            "screen=(%d,%d) eq_rect=%s book_wnd=(%d,%d) slot_offset=(%d,%d)",
            target_slot,
            target_page,
            name,
            sx,
            sy,
            self._eq_rect,
            self._book_wnd[0],
            self._book_wnd[1],
            ox,
            oy,
        )
        left_click_at(sx, sy)
        self._phase = _Phase.CLICK_SLOT_WAIT
        self._resume_after = time.time() + 0.3
        return RoutineStatus.RUNNING

    def _tick_click_gem(self) -> RoutineStatus:
        """Click the target gem to start memorization."""
        self._refresh_window_pos()
        gem, spell_id, name, book_slot = self._memorizable[self._spell_index]
        assert self._eq_rect is not None
        gx, gy = _gem_screen_pos(gem - 1, self._eq_rect, self._cast_wnd)
        log.info(
            "[CAST] MemSpells: CLICK gem %d '%s' -> "
            "screen=(%d,%d) eq_rect=%s cast_wnd=(%d,%d) "
            "gem_offset=(%d,%d+%d*%d)",
            gem,
            name,
            gx,
            gy,
            self._eq_rect,
            self._cast_wnd[0],
            self._cast_wnd[1],
            _GEM_X,
            _GEM_Y_START,
            gem - 1,
            _GEM_Y_SPACE,
        )
        left_click_at(gx, gy)
        self._phase = _Phase.CAST_CHECK_WAIT
        self._resume_after = time.time() + CAST_CHECK_DELAY
        return RoutineStatus.RUNNING

    def _tick_cast_check(self) -> RoutineStatus:
        """Check if memorization started after clicking gem."""
        gem, spell_id, name, book_slot = self._memorizable[self._spell_index]
        assert book_slot is not None  # filtered in compute_gem_changes
        target_page = book_slot // SLOTS_PER_PAGE
        target_slot = book_slot % SLOTS_PER_PAGE

        assert self._read_state_fn is not None
        mid = self._read_state_fn()
        if mid.casting_mode != 0:
            log.info(
                "[CAST] MemSpells: gem %d memorizing '%s' (casting_mode=%d page=%d slot=%d)",
                gem,
                name,
                mid.casting_mode,
                target_page,
                target_slot,
            )
            self._phase = _Phase.WAITING_MEMORIZE
            self._memorize_deadline = time.time() + MEMORIZE_WAIT
            self._resume_after = time.time() + 0.5  # poll every 0.5s
            return RoutineStatus.RUNNING
        else:
            log.warning(
                "[CAST] MemSpells: gem %d FAILED to memorize '%s' -- "
                "casting_mode=%d (expected !=0). "
                "Click may have missed. "
                "eq_rect=%s cast_wnd=(%d,%d) book_wnd=(%d,%d) "
                "page=%d slot=%d",
                gem,
                name,
                mid.casting_mode,
                self._eq_rect,
                self._cast_wnd[0],
                self._cast_wnd[1],
                self._book_wnd[0],
                self._book_wnd[1],
                target_page,
                target_slot,
            )
            self._spell_index += 1
            self._page_clicks_remaining = 0
            self._phase = _Phase.NAV_PAGE
            return RoutineStatus.RUNNING

    def _tick_waiting_memorize(self) -> RoutineStatus:
        """Poll gem array to detect memorization complete.

        Checks every tick if the spell appeared in the gem slot (0.5s polls
        via _resume_after). Falls back to timeout after MEMORIZE_WAIT.
        """
        gem, spell_id, name, book_slot = self._memorizable[self._spell_index]

        # Poll memory: is the spell in the gem yet?
        deadline = getattr(self, "_memorize_deadline", 0)
        try:
            reader = self._reader()
            if reader:
                current_gems = reader.read_memorized_spells()
                if current_gems.get(gem) == spell_id:
                    log.info("[CAST] MemSpells: gem %d memorized '%s' (verified by memory)", gem, name)
                elif time.time() < deadline:
                    # Not ready yet, poll again in 0.5s
                    self._resume_after = time.time() + 0.5
                    return RoutineStatus.RUNNING
                else:
                    log.warning("[CAST] MemSpells: gem %d timeout waiting for '%s' -- proceeding", gem, name)
        except (OSError, RuntimeError, ValueError):  # fmt: skip
            if time.time() < deadline:
                self._resume_after = time.time() + 0.5
                return RoutineStatus.RUNNING
            log.info("[CAST] MemSpells: gem %d memorized '%s' (timer fallback)", gem, name)

        self._spell_index += 1
        self._page_clicks_remaining = 0
        if self._spell_index < len(self._memorizable):
            # More spells to memorize, navigate to next
            self._phase = _Phase.NAV_PAGE
        else:
            # All done, close spellbook
            self._phase = _Phase.CLOSE_BOOK
        return RoutineStatus.RUNNING

    def _tick_close_book(self) -> RoutineStatus:
        """Close the spellbook."""
        log.info("[CAST] MemSpells: closing spellbook (all spells memorized)")
        toggle_spellbook()
        self._book_open = False
        self._phase = _Phase.CLOSE_BOOK_WAIT
        self._resume_after = time.time() + 0.5
        return RoutineStatus.RUNNING

    def _tick_stand_up(self) -> RoutineStatus:
        """Stand up after memorization."""
        import motor.actions as _actions

        _actions.mark_sitting()
        stand()
        self._phase = _Phase.STAND_WAIT
        self._resume_after = time.time() + 0.5
        return RoutineStatus.RUNNING

    def _tick_stand_wait(self, state: GameState) -> RoutineStatus:
        """Reconfigure from memory and return SUCCESS."""
        reader = self._reader()
        if reader:
            final_gems = reader.read_memorized_spells()
            configure_from_memory(final_gems, state.class_id)
            db = get_spell_db()
            log.info(
                "[CAST] MemSpells: final loadout: %s",
                ", ".join(
                    f"gem{g}={sd.name if (sd := db.get(s)) else s}" for g, s in sorted(final_gems.items())
                ),
            )
        if self._ctx:
            self._ctx.plan.active = None
        return RoutineStatus.SUCCESS

    @override
    def exit(self, state: GameState) -> None:
        """Cleanup on interrupt or completion.

        Must close spellbook and stand up so that any routine taking over
        (e.g. IN_COMBAT via auto-engage) starts with a clean motor state.
        Without standing here, the combat routine enters while the player is
        sitting, heading is locked, and face_heading loops into timeouts.
        """
        if self._book_open:
            log.info("[CAST] MemSpells: exit -- closing spellbook (interrupted)")
            toggle_spellbook()
            self._book_open = False
            interruptible_sleep(0.3, self._flee_check)
        # Stand up unconditionally: memorization always leaves player sitting.
        # Use the internal flag path so we don't double-toggle.
        import motor.actions as _actions

        if _actions.is_sitting():
            log.info("[CAST] MemSpells: exit -- standing up after interrupt")
            stand()
            interruptible_sleep(
                0.8, self._flee_check
            )  # EQ stand animation ~1s; heading locked until complete
