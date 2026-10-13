


# Chess Reasoner

A small language model (0.8B parameters) that finds chess moves by using tools.

The model calls tools to gather factual question about the board e.g. what pieces attack a square, which piece are hanging, what happens after Nxe5. Each tool call requires a sentence justifying the purpose of the tool call. After the model has checked candidate moves, it answers with its chosen move.

https://github.com/user-attachments/assets/6e328288-033e-441c-9689-ae364d27de01

## Run it

```bash
# tested with python 3.13
git clone https://github.com/lukesalamone/chess-reasoner
cd chess-reasoner
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./serve.sh
```

The first run will download the model from Hugging Face ([lukesalamone/chess-reasoner-0.8b](https://huggingface.co/lukesalamone/chess-reasoner-0.8b)). When it's done, open http://127.0.0.1:8765.

You can paste a FEN or press the dice button for a random position. Then press Analyze. You can also click any tool call in the reasoning trace to see what it returned.

## The agent loop

The model makes one tool call at a time. Example:

```
<tool>attackers(e5) | why: Is the pawn on e5 defended? If not, Nxe5 wins it.</tool>
```

The harness then runs the call on a scratch board and sends back the result:

```
<result>e5 (black pawn): attacked by Nf3; defended by Nc6.</result>
```

The model then either makes another tool call or gives a final answer. The harness limits the model to 24 tool calls, and requires a justification for making the tool call itself to prevent the reasoning trace from devolving into a list of bare tool calls. There is some basic validation on the justifications to prevent malformed, short, long, or repeated justifications.

| Tool Call         | Description |
| --- | --- |
| `piece_on(square)`  | What piece is on a square                                                                      |
| `pieces()`          | Every piece on the board and where it is (both sides)                                          |
| `material()`        | Material count for both sides in pawn units                                                    |
| `attackers(square)` | Which pieces (any side) attack a square. If occupied, distinguishes attackers from defenders   |
| `targets(square)`   | What enemy pieces the piece on square attacks, and what friendly pieces it defends             |
| `hanging(color)`    | What pieces of this color are hanging                                                          |
| `pins(color)`       | What pins against the king for color                                                           |
| `forcing_moves()`   | Legal checks, captures, and promotions are available in this position                          |
| `legal_moves()`     | Every legal move in this position, plus whether the side to move is in check                   |
| `try(move)`         | Push a move onto the stack to evaluate one ply deeper                                          |
| `undo()`            | Pop off the stack to back out of a search                                                      |
| `quiet_moves()`     | Legal moves that are not in `forcing_moves()`                                                  |
| `root()`            | Pop the whole stack to return to the starting position                                         |
| `threats()`         | What the opponent could do if it were their move: mates, checks, captures that win material    |
| `exchange(square)`  | Result of the capture sequence on a square, starting with the best first capture               |

## More information

You can find more information in my blog post: https://blog.lukesalamone.com/posts/chess-reasoning-llm/

## Legal

- `chess.js v1.4.0` - BSD 2-Clause
- `Stockfish.js v19` - GPLv3 (see `COPYING-stockfish.txt`)
- `positions.txt` - Lichess CC0
- chess pieces by [greyj](https://graphicriver.net/item/chess-pieces-/10750620)

Everything else is [MIT Licensed](LICENSE).
