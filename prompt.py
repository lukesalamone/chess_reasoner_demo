"""The prompts the model was trained with.

The system prompt lives in system_prompt.txt, identical to training. The
model is sensitive to it, so edit it only to experiment.
"""
from pathlib import Path

import chess

SYSTEM_PROMPT = (Path(__file__).resolve().parent / "system_prompt.txt") \
    .read_text(encoding="utf-8").removesuffix("\n")


def build_user_prompt(fen: str) -> str:
    """The user turn for one position: the FEN, the side to move and the legal moves."""
    board = chess.Board(fen)
    color = "White" if board.turn == chess.WHITE else "Black"
    moves = " ".join(board.san(m) for m in board.legal_moves)
    return (f"Position (FEN): {fen}\n"
            f"Side to move: {color}\n"
            f"Legal moves (SAN): {moves}\n"
            f"Find the best move for {color}, using the tools to justify every step.")
