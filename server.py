"""Web demo: enter a position and watch the chess reasoner think.

The page posts a FEN to /api/analyze and reads a stream of server-sent events:
every token as the model writes it, each tool call with its why and its result,
and the final answer. The model runs in vLLM's OpenAI-compatible server; this
file runs the tool loop around it, with the same prompts, stop strings, why
check, call budget and reminders as in training.

    ./serve.sh        # starts vLLM on :8000 and this server on :8765
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

import chess
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from prompt import SYSTEM_PROMPT, build_user_prompt
from tools import ToolSession
from why_check import check_why, split_call

HERE = Path(__file__).resolve().parent
VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = os.environ.get("VLLM_MODEL", "chess-reasoner")      # vLLM's --served-model-name
TITLE = os.environ.get("DEMO_TITLE", "Chess reasoner 0.8B")
POSITIONS = HERE / "positions.txt"

# maximum tool calls before model will be required to answer
MAX_CALLS = 24

# a turn is a justification plus a tool call, or the answer
MAX_TOKENS_PER_TURN = 512
STOP = ["</tool>", "</answer>"]
TOOL_RE = re.compile(r"<tool>\s*(.*?)\s*$", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*$", re.DOTALL)
NUDGE_FORMAT = "Make exactly one <tool>...</tool> call, or give your <answer>...</answer> now."
NUDGE_BUDGET = ("You have used your call budget. Give your final "
                "<answer>...</answer> now, choosing from the legal moves.")

app = FastAPI()


def sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def parse_move(board: chess.Board, text: str) -> chess.Move | None:
    """A legal move from the model's answer, in SAN or UCI."""
    text = text.strip().rstrip(".")
    for parse in (board.parse_uci, board.parse_san):
        try:
            mv = parse(text)
        except ValueError:
            continue
        if mv in board.legal_moves:
            return mv
    return None


async def stream_turn(client: httpx.AsyncClient, messages: list, temperature: float):
    """One assistant turn, streamed. Yields ("tok", {"s", "n"}) for each piece of
    text, then ("end", {"text", "finish", "tokens", "ttft", "gen_s"})."""
    # Tool results go back as user turns: the model reads the <result> text, and
    # the chat template has no tool role without structured tool calls.
    wire = [{"role": "user" if m["role"] == "tool" else m["role"], "content": m["content"]}
            for m in messages]
    # Every sampling parameter is set here. vLLM would otherwise fill the unset ones
    # from the model's generation_config.json.
    payload = {"model": MODEL, "messages": wire, "stream": True,
               "stream_options": {"include_usage": True, "continuous_usage_stats": True},
               "temperature": temperature, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0,
               "max_tokens": MAX_TOKENS_PER_TURN, "stop": STOP, "include_stop_str_in_output": True}
    t0 = time.perf_counter()
    t_first = None
    text, finish, tokens = "", None, 0
    async with client.stream("POST", VLLM_URL, json=payload, timeout=None) as r:
        if r.status_code != 200:
            body = (await r.aread()).decode()[:300]
            raise RuntimeError(f"vLLM HTTP {r.status_code}: {body}")
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            d = json.loads(data)
            if d.get("usage"):
                tokens = d["usage"].get("completion_tokens", tokens)
            for ch in d.get("choices") or []:
                delta = (ch.get("delta") or {}).get("content") or ""
                if delta:
                    if t_first is None:
                        t_first = time.perf_counter()
                    text += delta
                    yield "tok", {"s": delta, "n": tokens}
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t1 = time.perf_counter()
    yield "end", {"text": text, "finish": finish, "tokens": tokens,
                  "ttft": (t_first or t1) - t0, "gen_s": t1 - (t_first or t1)}


async def analyze(fen: str, temperature: float, request: Request):
    """The tool loop for one position, as a stream of events for the page."""
    board = chess.Board(fen)
    session = ToolSession(fen)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(fen)}]
    n_calls, nudged, whys = 0, False, []
    total_tok, total_gen = 0, 0.0
    yield sse({"t": "start", "fen": fen, "budget": MAX_CALLS})

    def done(**kw):
        return sse({"t": "done", "calls": n_calls, "tokens": total_tok,
                    "tps": round(total_tok / total_gen, 1) if total_gen else None, **kw})

    async with httpx.AsyncClient() as client:
        while True:
            if await request.is_disconnected():
                return
            yield sse({"t": "turn"})
            end = None
            try:
                async for kind, ev in stream_turn(client, messages, temperature):
                    if kind == "tok":
                        yield sse({"t": "tok", **ev})
                    else:
                        end = ev
            except Exception as e:
                yield sse({"t": "error", "s": f"{type(e).__name__}: {e}"})
                yield done(ok=False, reason="server error")
                return
            total_tok += end["tokens"]
            total_gen += end["gen_s"]
            yield sse({"t": "turn_end"})

            # The stop string stays in the text sent back to the model, so its next
            # prompt holds exactly what it wrote. Parsing ignores it.
            text = end["text"]
            body = text
            for s in STOP:
                if body.endswith(s):
                    body = body[: -len(s)]
                    break
            messages.append({"role": "assistant", "content": text})
            ti, ai = body.rfind("<tool>"), body.rfind("<answer>")

            if ai > ti:
                m = ANSWER_RE.search(body[ai:])
                raw = m.group(1).strip().split()[0] if m and m.group(1).strip() else ""
                mv = parse_move(board, raw)
                if mv is None:
                    yield sse({"t": "answer", "ok": False, "raw": raw})
                    yield done(ok=False, reason=f"illegal move '{raw}'")
                    return
                san = board.san(mv)
                yield sse({"t": "answer", "ok": True, "uci": mv.uci(), "san": san})
                yield done(ok=True, uci=mv.uci(), san=san)
                return

            m = TOOL_RE.search(body[ti:]) if ti != -1 else None
            if not m:
                if end["finish"] == "length":
                    yield done(ok=False, reason="turn hit the token limit without a tool call or answer")
                    return
                if not nudged:
                    nudged = True
                    messages.append({"role": "user", "content": NUDGE_FORMAT})
                    yield sse({"t": "nudge", "s": NUDGE_FORMAT})
                    continue
                yield done(ok=False, reason="no tool call or answer")
                return

            call, why = split_call(m.group(1))
            why, why_err = check_why(why, whys)
            if why_err:
                result = why_err
            else:
                whys.append(why)
                result = session.dispatch(call)
            n_calls += 1
            messages.append({"role": "tool", "content": f"<result>{result}</result>"})
            yield sse({"t": "tool", "call": call, "why": why, "rejected": bool(why_err),
                       "error": result.startswith("error"), "result": result, "n": n_calls})
            if n_calls >= MAX_CALLS and not nudged:
                nudged = True
                messages.append({"role": "user", "content": NUDGE_BUDGET})
                yield sse({"t": "nudge", "s": NUDGE_BUDGET})
            elif n_calls >= MAX_CALLS + 2:
                yield done(ok=False, reason="over the call budget without an answer")
                return


@app.post("/api/analyze")
async def api_analyze(request: Request):
    req = await request.json()
    try:
        board = chess.Board((req.get("fen") or "").strip())
    except ValueError as e:
        return JSONResponse({"error": f"invalid FEN: {e}"}, status_code=400)
    if not board.is_valid():
        return JSONResponse({"error": "illegal position"}, status_code=400)
    if board.is_game_over():
        return JSONResponse({"error": "the game is over in this position"}, status_code=400)
    temperature = min(max(float(req.get("temperature", 0.6)), 0.0), 1.5)
    return StreamingResponse(analyze(board.fen(), temperature, request), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/info")
async def api_info():
    return {"title": TITLE, "budget": MAX_CALLS}


_FENS = [l.strip() for l in open(POSITIONS) if l.strip()] if POSITIONS.exists() else []


@app.get("/api/random")
async def api_random():
    """A random position from positions.txt (none of them were used in training)."""
    return {"fen": random.choice(_FENS) if _FENS else chess.STARTING_FEN}


app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("DEMO_HOST", "127.0.0.1"),
                port=int(os.environ.get("DEMO_PORT", 8765)), log_level="warning")
