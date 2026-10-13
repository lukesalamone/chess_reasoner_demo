"""Checks the justification that the model gives with a tool call.

The model writes every call as <tool>attackers(e5) | why: Is the knight defended?</tool>.

A call runs only if its justification is
 - one or two sentences of 6-35 words,
 - is not a near-repeat of an earlier justification,
 - is not garbled or nonsense

Otherwise the model gets an error back and must try again. These are the rules it was trained under.
"""
import re

WHY_MIN_WORDS, WHY_MAX_WORDS = 6, 35


def split_call(text):
    """'attackers(e5) | why: Is it defended?' -> ('attackers(e5)', 'Is it defended?' or None)"""
    call, sep, why = text.partition("|")
    if sep and why.strip().lower().startswith("why:"):
        return call.strip(), why.strip()[4:].strip()
    return text.strip(), None


# letters from other scripts: Greek, Cyrillic, Japanese kana, CJK, Hangul
_GARBLED = re.compile(r"[Ͱ-ϿЀ-ӿ぀-ヿ㐀-鿿가-힯]")
# a word, a move or square, a number, a '...e5' reply, or a tool name like 'threats()'
_WORD = re.compile(r"^[A-Za-z][A-Za-z'-]*[.,;:?!)]*$|^\(?[KQRBNO]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?[+#]?[.,;:?!)]*$|"
                   r"^\(?[0-9.+/%-]+[.,;:?!)]*$|^\.\.\.[A-Za-z0-9+#=x-]+[.,;:?!)]*$|^\(?[A-Za-z0-9-]+\(\)?[.,;:?!)]*$")


def _word_set(t):
    return {w.strip(".,;:?!()'\"").lower() for w in t.split()} - {""}


def check_why(why, previous=()):
    """Returns (why, error). error is None when the call may run; otherwise it is
    the message the model gets back instead of the tool's result."""
    if not isinstance(why, str) or not why.strip():
        return None, ("error: every call needs \"why\": one sentence saying what you want to learn "
                      "from it and why. The call was not executed; call again with it.")
    why = " ".join(why.split())
    words = why.split()
    n = len(words)
    retry = " The call was not executed; call again with a better why."
    if n < WHY_MIN_WORDS or n > WHY_MAX_WORDS:
        return why, f"error: \"why\" must be {WHY_MIN_WORDS}-{WHY_MAX_WORDS} words (got {n})." + retry
    if len(re.findall(r"[.!?](\s|$)", why)) > 2:
        return why, "error: \"why\" must be one sentence (two at most)." + retry
    if (_GARBLED.search(why) or any(len(w) > 25 for w in words)
            or sum(bool(_WORD.match(w)) for w in words) < 0.8 * n
            or max(words.count(w) for w in words) > max(3, n // 4)):
        return why, "error: \"why\" is not a readable sentence." + retry
    ws = _word_set(why)
    for prev in previous:
        pw = _word_set(prev)
        if ws and pw and len(ws & pw) / len(ws | pw) >= 0.8:
            return why, ("error: this \"why\" repeats an earlier one. Say what THIS call should "
                         "reveal that earlier calls did not." + retry)
    return why, None
