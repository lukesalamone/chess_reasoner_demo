"""The chess tools the model calls while it reasons.

Each tool answers a narrow, factual question about the position in focus and
returns a short string. The model finds moves by combining these facts, and by
playing candidate lines on a scratch board with try() / undo() / root().

Be careful in changing the tools or wording here. Changing things will cause the
inputs to be out of distribution, and might harm prediction quality. The wordings
used here match what the model saw during training.
"""
from __future__ import annotations

import re

import chess

PIECE_LETTER = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B",
                chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}
PIECE_WORD = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
              chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
COLOR_WORD = {chess.WHITE: "White", chess.BLACK: "Black"}
# in pawns. material() leaves kings out; in exchange() and threats() the king's
# large value makes it the last piece to recapture
PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
               chess.QUEEN: 9, chess.KING: 100}


def _ref(board: chess.Board, sq: chess.Square) -> str:
    """'Nf3' for a piece, 'e4-pawn' for a pawn, 'e4' for an empty square."""
    p = board.piece_at(sq)
    name = chess.square_name(sq)
    if p is None:
        return name
    if p.piece_type == chess.PAWN:
        return f"{name}-pawn"
    return f"{PIECE_LETTER[p.piece_type]}{name}"


def _reflist(board: chess.Board, squares) -> str:
    squares = sorted(squares)
    return ", ".join(_ref(board, s) for s in squares) if squares else "none"


class ToolError(Exception):
    """A malformed or illegal call; it becomes an 'error: ...' result."""


class ToolSession:
    MAX_DEPTH = 4       # plies try() may stack above the starting position

    def __init__(self, fen: str):
        self.stack = [chess.Board(fen)]   # stack[0] is the starting position
        self.line: list[str] = []         # SAN of each try(), parallel to stack[1:]

    @property
    def board(self) -> chess.Board:
        return self.stack[-1]

    @property
    def depth(self) -> int:
        return len(self.stack) - 1

    def _stm(self) -> str:
        return COLOR_WORD[self.board.turn]

    def _parse_square(self, s: str) -> chess.Square:
        try:
            return chess.parse_square((s or "").strip())
        except ValueError:
            raise ToolError(f"'{s}' is not a square (use e.g. e4)")

    def _parse_color(self, s: str) -> chess.Color:
        s = (s or "").strip().lower()
        if s in ("white", "w"):
            return chess.WHITE
        if s in ("black", "b"):
            return chess.BLACK
        raise ToolError(f"'{s}' is not a color (use white or black)")

    def _parse_move(self, s: str) -> chess.Move:
        """A legal move in the position in focus, given in UCI or SAN."""
        s = (s or "").strip().rstrip(".")
        for parse in (self.board.parse_uci, self.board.parse_san):
            try:
                mv = parse(s)
            except Exception:
                continue
            if mv in self.board.legal_moves:
                return mv
        legal = sorted(self.board.san(m) for m in self.board.legal_moves)
        shown = ", ".join(legal[:40]) + (" ..." if len(legal) > 40 else "")
        hint = " (call root() if you meant to test a fresh candidate from the start)" \
            if self.depth > 0 else ""
        raise ToolError(f"'{s}' is not legal here (depth {self.depth}, {self._stm()} "
                        f"to move). Legal moves here: {shown}.{hint}")

    # --- what is on the board ---------------------------------------------
    def piece_on(self, square: str) -> str:
        sq = self._parse_square(square)
        p = self.board.piece_at(sq)
        name = chess.square_name(sq)
        if p is None:
            return f"{name}: empty"
        return f"{name}: {COLOR_WORD[p.color].lower()} {PIECE_WORD[p.piece_type]}"

    def _roster(self, color: chess.Color) -> str:
        parts = []
        for pt in (chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP,
                   chess.KNIGHT, chess.PAWN):
            for sq in sorted(self.board.pieces(pt, color)):
                parts.append(chess.square_name(sq) if pt == chess.PAWN
                             else f"{PIECE_LETTER[pt]}{chess.square_name(sq)}")
        return " ".join(parts) if parts else "(none)"

    def pieces(self) -> str:
        return (f"White: {self._roster(chess.WHITE)} / "
                f"Black: {self._roster(chess.BLACK)}")

    def _count(self, color: chess.Color) -> str:
        counts = []
        for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
            n = len(self.board.pieces(pt, color))
            if n:
                counts.append(f"{n}{PIECE_LETTER[pt]}")
        return "+".join(counts) if counts else "K only"

    def material(self) -> str:
        diff = sum((PIECE_VALUE[p.piece_type] if p.color == chess.WHITE else -PIECE_VALUE[p.piece_type])
                   for p in self.board.piece_map().values() if p.piece_type != chess.KING)
        if diff == 0:
            verdict = "equal"
        else:
            verdict = f"{COLOR_WORD[chess.WHITE if diff > 0 else chess.BLACK]} +{abs(diff)}"
        return (f"White {self._count(chess.WHITE)}, Black {self._count(chess.BLACK)}. "
                f"Material: {verdict}")

    # --- relations between pieces -----------------------------------------
    def _hanging_squares(self, color: chess.Color) -> list[chess.Square]:
        """Pieces (not the king) that are attacked and have no defender."""
        b = self.board
        return [sq for sq, p in b.piece_map().items()
                if p.color == color and p.piece_type != chess.KING
                and b.attackers(not color, sq) and not b.attackers(color, sq)]

    def _pinned_squares(self, color: chess.Color) -> list[chess.Square]:
        """Pieces (not the king) pinned against their own king."""
        b = self.board
        return [sq for sq, p in b.piece_map().items()
                if p.color == color and p.piece_type != chess.KING and b.is_pinned(color, sq)]

    def attackers(self, square: str) -> str:
        sq = self._parse_square(square)
        w = self.board.attackers(chess.WHITE, sq)
        b = self.board.attackers(chess.BLACK, sq)
        p = self.board.piece_at(sq)
        head = chess.square_name(sq)
        if p is not None:
            head += f" ({COLOR_WORD[p.color].lower()} {PIECE_WORD[p.piece_type]})"
            enemy = b if p.color == chess.WHITE else w
            friend = w if p.color == chess.WHITE else b
            atk = (f"attacked by {_reflist(self.board, enemy)}" if enemy
                   else "NOT attacked (safe — no enemy piece can capture it)")
            dfn = (f"defended by {_reflist(self.board, friend)}" if friend
                   else "undefended")
            return f"{head}: {atk}; {dfn}."
        return (f"{head} (empty): White attacks with {_reflist(self.board, w)}; "
                f"Black attacks with {_reflist(self.board, b)}")

    def targets(self, square: str) -> str:
        sq = self._parse_square(square)
        p = self.board.piece_at(sq)
        if p is None:
            raise ToolError(f"{chess.square_name(sq)} is empty — no piece to move")
        attacks_enemy, defends_friend = [], []
        for t in self.board.attacks(sq):
            tp = self.board.piece_at(t)
            if tp is None:
                continue
            (defends_friend if tp.color == p.color else attacks_enemy).append(t)
        return (f"{_ref(self.board, sq)} attacks {_reflist(self.board, attacks_enemy)}; "
                f"defends {_reflist(self.board, defends_friend)}")

    def hanging(self, color: str) -> str:
        col = self._parse_color(color)
        out = [f"{_ref(self.board, sq)} (attacked by "
               f"{_reflist(self.board, self.board.attackers(not col, sq))})"
               for sq in sorted(self._hanging_squares(col))]
        return f"{COLOR_WORD[col]} hanging: {'; '.join(out) if out else 'none'}"

    def pins(self, color: str) -> str:
        col = self._parse_color(color)
        out = [_ref(self.board, sq) for sq in sorted(self._pinned_squares(col))]
        return f"{COLOR_WORD[col]} pinned: {', '.join(out) if out else 'none'}"

    # --- candidate moves ----------------------------------------------------
    def forcing_moves(self) -> str:
        checks, captures, promos = [], [], []
        for mv in self.board.legal_moves:
            san = self.board.san(mv)
            if mv.promotion:
                promos.append(san)
            elif self.board.is_capture(mv):
                captures.append(san)
            if self.board.gives_check(mv) and san not in checks:
                checks.append(san)

        def fmt(lst):
            return ", ".join(sorted(lst)) if lst else "none"
        if not (checks or captures or promos):
            return ("checks: none; captures: none; promotions: none. NO forcing moves "
                    "exist — this does NOT mean checkmate or no options; call "
                    "quiet_moves()/legal_moves() and choose a QUIET move.")
        return (f"checks: {fmt(checks)}; captures: {fmt(captures)}; "
                f"promotions: {fmt(promos)}")

    def quiet_moves(self) -> str:
        quiet = [self.board.san(mv) for mv in self.board.legal_moves
                 if not (mv.promotion or self.board.is_capture(mv)
                         or self.board.gives_check(mv))]
        return f"quiet moves ({len(quiet)}): {', '.join(sorted(quiet)) if quiet else 'none'}"

    def legal_moves(self) -> str:
        moves = sorted(self.board.san(m) for m in self.board.legal_moves)
        if not moves:
            state = ("checkmate" if self.board.is_checkmate()
                     else "stalemate" if self.board.is_stalemate() else "no legal moves")
            return f"legal moves: none — {state}"
        chk = " (in check)" if self.board.is_check() else ""
        return f"legal moves{chk} ({len(moves)}): {', '.join(moves)}"

    # --- playing lines on the scratch board ----------------------------------
    def try_(self, move: str) -> str:
        if self.depth >= self.MAX_DEPTH:
            raise ToolError(f"already {self.MAX_DEPTH} plies deep — undo() before "
                            f"going further")
        mv = self._parse_move(move)
        san = self.board.san(mv)
        nb = self.board.copy(stack=False)
        nb.push(mv)
        self.stack.append(nb)
        self.line.append(san)
        if nb.is_checkmate():
            turn = "CHECKMATE"
        elif nb.is_stalemate():
            turn = "stalemate"
        else:
            chk = ", in check" if nb.is_check() else ""
            turn = f"{COLOR_WORD[nb.turn]} to move{chk}"
        return (f"NOW AT depth {self.depth}/{self.MAX_DEPTH}, {turn}. "
                f"played {san}; line from root: {' '.join(self.line)}.")

    def undo(self) -> str:
        if self.depth == 0:
            raise ToolError("already at the root — nothing to undo")
        undone = self.line.pop()
        self.stack.pop()
        where = "root position" if self.depth == 0 else f"line from root: {' '.join(self.line)}"
        return (f"NOW AT depth {self.depth}/{self.MAX_DEPTH}, {self._stm()} to move. "
                f"undid {undone}; {where}.")

    def root(self) -> str:
        if self.depth == 0:
            return (f"NOW AT depth 0/{self.MAX_DEPTH}, {self._stm()} to move. "
                    f"already at the root position.")
        cleared = " ".join(self.line)
        self.stack = self.stack[:1]
        self.line = []
        return (f"NOW AT depth 0/{self.MAX_DEPTH}, {self._stm()} to move. "
                f"reset to the root (cleared line: {cleared}).")

    # --- threats and exchanges -----------------------------------------------
    def _see_move(self, b: chess.Board, move: chess.Move) -> tuple[int, list[str]]:
        """Net material for the side making a capture, when both sides keep
        recapturing on that square with their least valuable piece and may stop
        at any point. Recaptures are legal moves, so pins are respected."""
        victim = b.piece_at(move.to_square)
        gain = PIECE_VALUE[victim.piece_type] if victim else (1 if b.is_en_passant(move) else 0)
        san = b.san(move)
        b2 = b.copy(stack=False)
        b2.push(move)
        reply, line = self._see_square(b2, move.to_square)
        if reply > 0:
            return gain - reply, [san] + line
        return gain, [san]

    def _see_square(self, b: chess.Board, sq: chess.Square) -> tuple[int, list[str]]:
        caps = [m for m in b.legal_moves if m.to_square == sq and b.is_capture(m)]
        if not caps:
            return 0, []
        m = min(caps, key=lambda m: PIECE_VALUE[b.piece_at(m.from_square).piece_type])
        return self._see_move(b, m)

    def threats(self) -> str:
        """What the opponent could do if it were their move now (a null move)."""
        b = self.board.copy(stack=False)
        if b.is_check():
            return "the side to move is in check; the opponent's threat is already on the board"
        b.push(chess.Move.null())
        opp = "White" if b.turn == chess.WHITE else "Black"
        mates, checks, wins = [], [], []
        for m in list(b.legal_moves):
            san = b.san(m)
            b.push(m)
            mate = b.is_checkmate()
            b.pop()
            if mate:
                mates.append(san)
            elif b.gives_check(m):
                checks.append(san)
            if b.is_capture(m):
                net, _ = self._see_move(b, m)
                if net > 0:
                    wins.append(f"{san} (+{net})")
        return (f"if it were {opp}'s move: mates: {', '.join(mates) or 'none'}; "
                f"checks: {', '.join(checks) or 'none'}; "
                f"captures that win material: {', '.join(wins) or 'none'}")

    def exchange(self, square: str) -> str:
        """The best capture on a square for the side to move, played out to the end."""
        sq = self._parse_square(square)
        b = self.board
        caps = [m for m in b.legal_moves if m.to_square == sq and b.is_capture(m)]
        name = chess.square_name(sq)
        if not caps:
            return f"{name}: {self._stm()} has no capture there"
        net, line = max((self._see_move(b, m) for m in caps), key=lambda t: t[0])
        if line[0].endswith("#"):
            return f"{name}: {line[0]} is checkmate"
        if net == 0:
            result = "comes out even"
        else:
            result = f"{'wins' if net > 0 else 'loses'} {abs(net)} (pawn = 1)"
        return f"{name}: best capture {line[0]} {result}; exchange: {' '.join(line)}"

    # --- dispatch -------------------------------------------------------------
    # tool name -> (method, number of arguments); `try` is a reserved word in Python
    NAMES = {
        "piece_on": ("piece_on", 1), "pieces": ("pieces", 0),
        "material": ("material", 0), "attackers": ("attackers", 1),
        "targets": ("targets", 1), "hanging": ("hanging", 1),
        "pins": ("pins", 1), "forcing_moves": ("forcing_moves", 0),
        "legal_moves": ("legal_moves", 0), "quiet_moves": ("quiet_moves", 0),
        "try": ("try_", 1), "undo": ("undo", 0), "root": ("root", 0),
        "threats": ("threats", 0), "exchange": ("exchange", 1),
    }
    ALIASES = {"try_move": "try"}
    _CALL_RE = re.compile(r"^\s*([a-z_]+)\s*\((.*)\)\s*$", re.DOTALL)

    def call(self, name: str, args: list) -> str:
        """Run one tool; a bad call returns 'error: ...' and never raises."""
        name = self.ALIASES.get(name, name)
        spec = self.NAMES.get(name)
        if spec is None:
            return f"error: no tool '{name}'. Available: " + ", ".join(self.NAMES)
        method, arity = spec
        if len(args) != arity:
            return f"error: {name}() takes {arity} argument(s), got {len(args)}"
        try:
            return getattr(self, method)(*args)
        except ToolError as e:
            return f"error: {e}"

    def dispatch(self, call_str: str) -> str:
        """'attackers(e4)' -> the tool's result."""
        m = self._CALL_RE.match(call_str)
        if not m:
            return (f"error: could not parse '{call_str.strip()}' "
                    f"(expected name(args), e.g. attackers(e4))")
        raw = m.group(2).strip()
        args = [a.strip() for a in raw.split(",")] if raw else []
        return self.call(m.group(1), args)
