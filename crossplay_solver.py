#!/usr/bin/env python3
"""NYT Crossplay move finder.

Usage: python3 crossplay_solver.py board.txt RACK [--top 15] [--defend N] [--cap N] [--no-swap]
                                   [--last-move]

board.txt: 15 lines of 15 chars. '.' = empty, UPPERCASE = tile, lowercase = blank tile.
RACK: your letters, '?' for a blank. The NWL2023 word list is downloaded from GitHub if missing.

Moves rank by value = score + leave value (0 once the bag is empty). --defend N estimates the
opponent's reply to the top N moves (exact once 7 or fewer tiles are unseen; otherwise replies of
up to --cap tiles, 4/5/7 by bag size, plus 7-tile sweeps onto an empty 2W/3W, averaged over
sampled racks) and ranks by equity = value - expected reply - turn cost. Turn cost: a move that leaves the bag non-empty is
charged (chance they empty it) x (expected reply), since the bag-emptier gets the last turn.
Swaps (bag >= --swap-min) are 0-point moves valued at the leave kept.
A sampled opponent rack that holds a blank may use it as any letter (scored 0), on the square where
that costs least. --last-move: the bag is empty and this play ends the game (you emptied the bag and
the opponent has had their final turn), so moves rank by score alone and no reply is checked.
"""
import argparse
from collections import Counter

N = 15
VALUES = dict(A=1, B=4, C=3, D=2, E=1, F=4, G=4, H=3, I=1, J=10, K=6, L=2, M=3,
              N=1, O=1, P=3, Q=10, R=1, S=1, T=1, U=2, V=6, W=5, X=8, Y=4, Z=10)
SWEEP_BONUS = 40
TILE_COUNTS = dict(A=9, B=2, C=2, D=4, E=12, F=2, G=3, H=3, I=8, J=1, K=1, L=4, M=2,
                   N=5, O=8, P=2, Q=1, R=6, S=5, T=6, U=3, V=2, W=2, X=1, Y=2, Z=1)
TILE_COUNTS["?"] = 3
assert sum(TILE_COUNTS.values()) == 100

_PREM_ROWS = [
    "3L . . 3W . . . 2L . . . 3W . . 3L",
    ". 2W . . . . 3L . 3L . . . . 2W .",
    ". . . . 2L . . . . . 2L . . . .",
    "3W . . 2L . . . 2W . . . 2L . . 3W",
    ". . 2L . . 3L . . . 3L . . 2L . .",
    ". . . . 3L . . 2L . . 3L . . . .",
    ". 3L . . . . . . . . . . . 3L .",
    "2L . . 2W . 2L . . . 2L . 2W . . 2L",
]
PREM = [r.split() for r in _PREM_ROWS]
PREM += PREM[6::-1]
assert len(PREM) == N and all(len(r) == N for r in PREM)

WORDLIST_URL = ("https://raw.githubusercontent.com/scrabblewords/scrabblewords/"
                "main/words/North-American/NWL2023.txt")


def load_words(path):
    import os
    if not os.path.exists(path):
        import urllib.request
        print(f"{path} not found, downloading NWL2023 from GitHub...")
        urllib.request.urlretrieve(WORDLIST_URL, path)
    words = set()
    with open(path) as f:
        for line in f:
            w = line.split()[0].upper() if line.strip() else ""
            if w.isalpha() and 2 <= len(w) <= N:
                words.add(w)
    return words


def load_board(path):
    rows = [l.rstrip("\n") for l in open(path) if l.strip()]
    assert len(rows) == N and all(len(r) == N for r in rows), "board must be 15x15"
    return [list(r) for r in rows]


def cell(b, r, c):
    return b[r][c] if 0 <= r < N and 0 <= c < N else "."


def tile_score(ch):
    return 0 if ch.islower() else VALUES[ch]


def cross_word(b, r, c, ch, horiz):
    dr, dc = (1, 0) if horiz else (0, 1)
    r0, c0 = r, c
    while cell(b, r0 - dr, c0 - dc) != ".":
        r0, c0 = r0 - dr, c0 - dc
    letters, base, rr, cc = [], 0, r0, c0
    while True:
        x = ch if (rr, cc) == (r, c) else cell(b, rr, cc)
        if x == ".":
            break
        letters.append(x)
        rr, cc = rr + dr, cc + dc
    if len(letters) == 1:
        return None
    p = PREM[r][c]
    lm = 3 if p == "3L" else 2 if p == "2L" else 1
    wm = 3 if p == "3W" else 2 if p == "2W" else 1
    rr, cc = r0, c0
    for x in letters:
        base += tile_score(x) * (lm if (rr, cc) == (r, c) else 1)
        rr, cc = rr + dr, cc + dc
    return "".join(letters).upper(), base * wm


import os
import pickle
import sys
TRIE_CACHE = None
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_TRIE = {}


def build_trie(words):
    root = {}
    for w in words:
        n = root
        for ch in w:
            n = n.get(ch) or n.setdefault(ch, {})
        n["$"] = 1
    return root


def get_trie(words, cache_path=None):
    key = id(words)
    if key in _TRIE:
        return _TRIE[key]
    t = None
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                t = pickle.load(f)
        except Exception:
            t = None
    if t is None:
        t = build_trie(words)
        if cache_path:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(t, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                pass
    _TRIE[key] = t
    return t


LMUL = {"3L": 3, "2L": 2}
WMUL = {"3W": 3, "2W": 2}


def make_solve(S, cache_path=None):
    """Trie + anchors + cross-check sets move generator (Appel-Jacobson style)."""
    N, PREM, tile_score, SWEEP_BONUS, VALUES = S.N, S.PREM, S.tile_score, S.SWEEP_BONUS, S.VALUES
    LM, WM = S.LMUL, S.WMUL

    def solve(board, rack, words, max_tiles=7, sink=None):
        """Every legal move, best score first. A blank goes on the square where it costs least, and
        a one-tile play that forms a word both ways is listed once.
        With `sink`, no list is built: each move goes to sink(score, real, blanks, ref), where real
        is [(letter, points that tile adds to the score)] for the non-blank tiles (see reply_model)."""
        allwords = getattr(words, "allwords", words)
        trie = get_trie(allwords, cache_path or TRIE_CACHE)
        rack_up = rack.upper()
        rack_full = Counter(rack_up)
        rc = Counter(rack_up)
        nblank = [rc.pop("?", 0)]
        nblank0 = nblank[0]
        rc0 = Counter(rc)  # untouched copy: the search below mutates rc
        empty_board = all(ch == "." for row in board for ch in row)
        moves = []
        singles = set()  # one-tile plays already recorded: (square, letter)

        for horiz in (True, False):
            B = board if horiz else [list(col) for col in zip(*board)]

            def orig(r, c):
                return (r, c) if horiz else (c, r)

            for r in range(N):
                row = B[r]
                cross = [None] * N  # None = no perpendicular word; else (allowed set, base, up, down)
                anchor = [False] * N
                for c in range(N):
                    if row[c] != ".":
                        continue
                    up_r = r
                    while up_r - 1 >= 0 and B[up_r - 1][c] != ".":
                        up_r -= 1
                    dn_r = r
                    while dn_r + 1 < N and B[dn_r + 1][c] != ".":
                        dn_r += 1
                    up = "".join(B[i][c] for i in range(up_r, r))
                    dn = "".join(B[i][c] for i in range(r + 1, dn_r + 1))
                    if up or dn:
                        U, D = up.upper(), dn.upper()
                        allowed = {ch for ch in LETTERS if U + ch + D in allwords}
                        base = sum(tile_score(x) for x in up + dn)
                        cross[c] = (allowed, base, U, D)
                        anchor[c] = True
                    elif (c > 0 and row[c - 1] != ".") or (c + 1 < N and row[c + 1] != "."):
                        anchor[c] = True
                    if empty_board and orig(r, c) == (7, 7):
                        anchor[c] = True
                if not any(anchor):
                    continue
                prem = [PREM[orig(r, c)[0]][orig(r, c)[1]] for c in range(N)]
                lmrow = [LM.get(p, 1) for p in prem]
                wmrow = [WM.get(p, 1) for p in prem]
                pre = [0]  # running total of the points of tiles already in this line
                for c in range(N):
                    pre.append(pre[-1] + (tile_score(row[c]) if row[c] != "." else 0))

                def record(start, end, placed):
                    # placed: list of (col, letter) in left-to-right order
                    k = len(placed)
                    if k == 1:
                        one = (orig(r, placed[0][0]), placed[0][1])
                        if one in singles:
                            return
                        singles.add(one)
                    wm = 1
                    for c, _ in placed:
                        wm *= wmrow[c]
                    base, cross_pts, adds = pre[end + 1] - pre[start], 0, []
                    for c, ch in placed:
                        v = VALUES[ch] * lmrow[c]
                        base += v
                        x = cross[c]
                        if x is None:
                            adds.append(v * wm)  # what this tile's own points add to the score
                        else:
                            cross_pts += (x[1] + v) * wmrow[c]
                            adds.append(v * (wm + wmrow[c]))
                    total = base * wm + cross_pts + (SWEEP_BONUS if k == 7 else 0)
                    blank_at = ()
                    if nblank[0] < nblank0:  # blanks stand in for the copies of a letter that add least
                        need = Counter(ch for _, ch in placed)
                        blank_at = set()
                        for ch, n in need.items():
                            if n > rc0[ch]:
                                spots = sorted((adds[i], i) for i in range(k) if placed[i][1] == ch)
                                blank_at.update(i for _, i in spots[:n - rc0[ch]])
                        total -= sum(adds[i] for i in blank_at)
                    if sink is not None:
                        sink(total, [(placed[i][1], adds[i]) for i in range(k) if i not in blank_at],
                             len(blank_at), (horiz, r, start, end, tuple(placed)))
                        return
                    tiles = [(c, ch.lower() if i in blank_at else ch)
                             for i, (c, ch) in enumerate(placed)]
                    newmap = dict(tiles)
                    letters = [newmap[c].upper() if c in newmap else row[c].upper()
                               for c in range(start, end + 1)]
                    formed = []
                    for c, ch in tiles:
                        x = cross[c]
                        if x is not None:
                            formed.append((x[2] + ch.upper() + x[3],
                                           (x[1] + tile_score(ch) * lmrow[c]) * wmrow[c]))
                    main = total - sum(sc for _, sc in formed) - (SWEEP_BONUS if k == 7 else 0)
                    used = Counter(ch if ch.isupper() else "?" for _, ch in tiles)
                    leave = rack_full - used
                    rs, cs = orig(r, start)
                    moves.append(dict(word="".join(letters), row=rs + 1, col=cs + 1,
                                      tiles={orig(r, c): ch for c, ch in tiles},
                                      dir="across" if horiz else "down", score=total,
                                      main=main, cross=formed,
                                      leave="".join(sorted(leave.elements()))))

                def extend(node, c, start, placed, anc):
                    if c < N and row[c] != ".":
                        nxt = node.get(row[c].upper())
                        if nxt is not None:
                            extend(nxt, c + 1, start, placed, anc)
                        return
                    if placed and c > anc and "$" in node:
                        record(start, c - 1, placed)
                    if c >= N or len(placed) >= max_tiles:
                        return
                    x = cross[c]
                    allowed = None if x is None else x[0]
                    for ch, nxt in node.items():
                        if ch == "$" or (allowed is not None and ch not in allowed):
                            continue
                        if rc[ch] > 0:
                            rc[ch] -= 1
                            placed.append((c, ch))
                            extend(nxt, c + 1, start, placed, anc)
                            placed.pop()
                            rc[ch] += 1
                        elif nblank[0] > 0:
                            nblank[0] -= 1
                            placed.append((c, ch))
                            extend(nxt, c + 1, start, placed, anc)
                            placed.pop()
                            nblank[0] += 1

                def left(node, partial, limit, anc):
                    # partial: letters of the left part, placed just before the anchor
                    start = anc - len(partial)
                    placed = [(start + i, ch) for i, ch in enumerate(partial)]
                    extend(node, anc, start, placed, anc)
                    if limit == 0 or len(partial) + 1 >= max_tiles:
                        return
                    for ch, nxt in node.items():
                        if ch == "$":
                            continue
                        if rc[ch] > 0:
                            rc[ch] -= 1
                            partial.append(ch)
                            left(nxt, partial, limit - 1, anc)
                            partial.pop()
                            rc[ch] += 1
                        elif nblank[0] > 0:
                            nblank[0] -= 1
                            partial.append(ch)
                            left(nxt, partial, limit - 1, anc)
                            partial.pop()
                            nblank[0] += 1

                for a in range(N):
                    if not anchor[a]:
                        continue
                    if a > 0 and row[a - 1] != ".":
                        s = a
                        while s - 1 >= 0 and row[s - 1] != ".":
                            s -= 1
                        node = trie
                        for x in row[s:a]:
                            node = node.get(x.upper())
                            if node is None:
                                break
                        if node is not None:
                            extend(node, a, s, [], a)
                    else:
                        limit, k = 0, a - 1
                        while k >= 0 and row[k] == "." and not anchor[k]:
                            limit += 1
                            k -= 1
                        left(trie, [], limit, a)

        moves.sort(key=lambda m: -m["score"])
        return moves

    return solve


solve = make_solve(sys.modules[__name__])


def unseen_tiles(board, rack):
    left = Counter(TILE_COUNTS)
    for row in board:
        for ch in row:
            if ch != ".":
                left["?" if ch.islower() else ch] -= 1
    left -= Counter(rack.upper())
    return +left


def apply_move(board, m):
    b = [row[:] for row in board]
    for (r, c), ch in m["tiles"].items():
        b[r][c] = ch
    return b


def opponent_threat(board, unseen, words):
    rack = "".join(unseen.elements())
    exact = len(rack) <= 7
    moves = solve(board, rack, words, max_tiles=7 if exact else 2)
    return (moves[0] if moves else None), exact


# Rough leave values (points of future equity per tile kept). Bag-empty leaves score 0.
LEAVE_VALUES = {"?": 25, "S": 8, "X": 3, "Z": 3, "E": 1.5, "A": 1, "R": 1, "H": 0.5, "C": 0.5,
                "D": 0, "M": 0, "N": 0, "T": 0, "I": -0.5, "L": -1, "P": -1, "Y": -1, "O": -1.5,
                "B": -2, "F": -2, "G": -2, "K": -2.5, "J": -3, "U": -3, "W": -4, "V": -5, "Q": -7}
VOWELS = set("AEIOU")


def leave_value(leave, bag):
    """Heuristic value of the tiles you keep. Zero once the bag is empty."""
    if bag == 0 or not leave:
        return 0.0
    c = Counter(leave)
    v = sum(LEAVE_VALUES.get(ch, 0) * n for ch, n in c.items())
    v -= sum(3 * (n - 1) for ch, n in c.items() if n > 1 and ch != "?")  # duplicates
    vow = sum(n for ch, n in c.items() if ch in VOWELS)
    con = sum(n for ch, n in c.items() if ch not in VOWELS and ch != "?")
    v -= 2 * max(0, abs(vow - con) - 1)  # vowel/consonant imbalance
    if "Q" in c and "U" not in c:
        v -= 5
    return round(v, 1)


_BY_LEN = {}


def _words_by_len(words):
    """Newline-joined word lists per length, for fast regex matching (built once)."""
    key = id(words)
    if key not in _BY_LEN:
        groups = {}
        for w in words:
            groups.setdefault(len(w), []).append(w)
        _BY_LEN[key] = {L: "\n".join(sorted(ws)) for L, ws in groups.items()}
    return _BY_LEN[key]


def premium_sweeps(board, pool, words, tiles_needed=7):
    """Every 7-tile play (sweep, +40) covering an empty 2W/3W, found by regex matching."""
    import re
    pool_c = Counter(pool.upper())
    blanks = pool_c.pop("?", 0)
    by_len = _words_by_len(words)
    cross_cache = {}

    def allowed(r, c, horiz):
        """Letters that form a valid cross word at (r, c); None if there is no cross word."""
        k = (r, c, horiz)
        if k not in cross_cache:
            dr, dc = (1, 0) if horiz else (0, 1)
            if cell(board, r - dr, c - dc) == "." and cell(board, r + dr, c + dc) == ".":
                cross_cache[k] = None
            else:
                cross_cache[k] = "".join(ch for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                         if cross_word(board, r, c, ch, horiz)[0] in words)
        return cross_cache[k]

    moves = []
    for horiz in (True, False):
        for line in range(N):
            pos = [(line, i) if horiz else (i, line) for i in range(N)]
            for s in range(N):
                if s > 0 and board[pos[s - 1][0]][pos[s - 1][1]] != ".":
                    continue
                empties, i = [], s
                while i < N and len(empties) < tiles_needed:
                    if board[pos[i][0]][pos[i][1]] == ".":
                        empties.append(i)
                    i += 1
                if len(empties) < tiles_needed:
                    break
                e = i - 1
                while e + 1 < N and board[pos[e + 1][0]][pos[e + 1][1]] != ".":
                    e += 1
                if not any(PREM[pos[j][0]][pos[j][1]] in ("2W", "3W") for j in empties):
                    continue
                parts, touches, ok = [], False, True
                for j in range(s, e + 1):
                    r, c = pos[j]
                    if board[r][c] != ".":
                        parts.append(board[r][c].upper())
                        touches = True
                        continue
                    a = allowed(r, c, horiz)
                    if a is None:
                        parts.append(".")
                    elif not a:
                        ok = False
                        break
                    else:
                        parts.append("[" + a + "]")
                        touches = True
                if not ok or not touches:
                    continue
                L = e - s + 1
                if L not in by_len:
                    continue
                for w in re.findall("^" + "".join(parts) + "$", by_len[L], re.M):
                    need = Counter(w[j - s] for j in empties)
                    if sum(max(0, n - pool_c[ch]) for ch, n in need.items()) > blanks:
                        continue
                    wm = 1
                    for j in empties:
                        wm *= WMUL.get(PREM[pos[j][0]][pos[j][1]], 1)
                    adds = {}  # square -> what a real tile's own points add to the score there
                    for j in empties:
                        r, c = pos[j]
                        p = PREM[r][c]
                        crossed = allowed(r, c, horiz) is not None
                        adds[j] = (VALUES[w[j - s]] * LMUL.get(p, 1)
                                   * (wm + (WMUL.get(p, 1) if crossed else 0)))
                    blank_at = set()  # blanks stand in for the copies that add least
                    for ch, n in need.items():
                        if n > pool_c[ch]:
                            spots = sorted((adds[j], j) for j in empties if w[j - s] == ch)
                            blank_at.update(j for _, j in spots[:n - pool_c[ch]])
                    tiles = {pos[j]: w[j - s].lower() if j in blank_at else w[j - s]
                             for j in empties}
                    total, formed, base, by_letter = 0, [], 0, {}
                    for j in range(s, e + 1):
                        r, c = pos[j]
                        if (r, c) not in tiles:
                            base += tile_score(board[r][c])
                            continue
                        ch = tiles[(r, c)]
                        base += tile_score(ch) * LMUL.get(PREM[r][c], 1)
                        cw = cross_word(board, r, c, ch, horiz)
                        if cw:
                            formed.append(cw)
                            total += cw[1]
                        if j not in blank_at:
                            by_letter.setdefault(ch, []).append(adds[j])
                    total += base * wm + SWEEP_BONUS
                    cost = {}  # letter -> cost of playing 1, 2, .. of its copies as blanks
                    for ch, xs in by_letter.items():
                        xs.sort()
                        cost[ch] = [sum(xs[:i + 1]) for i in range(len(xs))]
                    rs, cs = pos[s]
                    moves.append(dict(word=w, row=rs + 1, col=cs + 1, tiles=tiles,
                                      dir="across" if horiz else "down", score=total,
                                      main=base * wm, cross=formed, leave="", blank_cost=cost))
    return moves


def _reply_move(board, score, ref):
    """A printable move for a reply kept by reply_model."""
    horiz, r, start, end, placed = ref
    new = dict(placed)
    line = board[r] if horiz else [board[i][r] for i in range(N)]
    word = "".join(new[c] if c in new else line[c].upper() for c in range(start, end + 1))
    rs, cs = (r, start) if horiz else (start, r)
    return dict(word=word, row=rs + 1, col=cs + 1, dir="across" if horiz else "down", score=score,
                tiles={((r, c) if horiz else (c, r)): ch for c, ch in placed})


def reply_model(board, unseen, words, cap, bag_after=0):
    """The opponent's replies on `board` when their rack is 7 of the (more than 7) unseen tiles.
    Lists every reply of up to `cap` tiles the unseen tiles allow, plus 7-tile sweeps onto an empty
    2W/3W. Returns (reply, worst, sweeps): reply(rack) gives, for a Counter rack, (best score, the
    sweep if their best reply is one, whether they can empty a bag of `bag_after`); worst is their
    best reply with any rack. A blank in the rack plays as any letter, scores 0, and goes where that
    costs least."""
    from itertools import combinations_with_replacement
    from operator import itemgetter
    pool = "".join(unseen.elements())
    pool_blanks = unseen.get("?", 0)
    best = {}  # (blanks used, sorted real letters) -> (score, ref): the top reply per set of tiles
    raw = []   # every reply, as [score, blanks used, real tiles, parsed]: for racks holding a blank

    def sink(score, real, blanks, ref):
        k = (blanks, tuple(sorted([ch for ch, _ in real])))
        if k not in best or score > best[k][0]:
            best[k] = (score, ref)
        if pool_blanks:
            raw.append([score, blanks, real, None])

    solve(board, pool, words, max_tiles=cap, sink=sink)
    threats = sorted(((sc, tuple(Counter(k[1]).items()), len(k[1]))
                      for k, (sc, _ref) in best.items() if k[0] == 0),
                     key=itemgetter(0), reverse=True)
    raw.sort(key=itemgetter(0), reverse=True)
    emptiers = [need for _sc, need, n in threats if n >= bag_after] if bag_after else []
    raw_emptiers = [m for m in raw if m[1] + len(m[2]) >= bag_after] if bag_after else []

    def parsed(m):
        """[(letter, copies needed, cost of playing 1, 2, .. of them as blanks)], built on first use."""
        if m[3] is None:
            groups = {}
            for ch, add in m[2]:
                groups.setdefault(ch, []).append(add)
            m[3] = []
            for ch, xs in groups.items():
                xs.sort()
                m[3].append((ch, len(xs), [sum(xs[:i + 1]) for i in range(len(xs))]))
        return m[3]

    def with_blanks(rack, b, floor):
        """Best reply for a rack with b blanks, each covering a letter the rack lacks."""
        top = floor
        for m in raw:
            if m[0] <= top:
                break
            short = m[1]
            if short > b:
                continue
            cost = 0
            for ch, n, costs in m[3] or parsed(m):
                d = n - rack[ch]
                if d > 0:
                    short += d
                    if short > b:
                        break
                    cost += costs[d - 1]
            else:
                if m[0] - cost > top:
                    top = m[0] - cost
        return top

    def can_empty_with_blanks(rack, b):
        for m in raw_emptiers:
            short = m[1]
            if short > b:
                continue
            for ch, n, _costs in m[3] or parsed(m):
                if n > rack[ch]:
                    short += n - rack[ch]
                    if short > b:
                        break
            else:
                return True
        return False

    sweeps = premium_sweeps(board, pool, words)
    sweeps_by_rack = {}
    for m in sweeps:
        key = tuple(sorted(ch if ch.isupper() else "?" for ch in m["tiles"].values()))
        sweeps_by_rack.setdefault(key, []).append(m)
    for ms in sweeps_by_rack.values():
        ms.sort(key=lambda m: -m["score"])

    def sweep_for(fixed, b):
        """Best premium sweep for a rack (a sweep uses the whole rack, so look it up by rack)."""
        if not sweeps:
            return 0, None
        if b == 0:
            ms = sweeps_by_rack.get(tuple(fixed))
            return (ms[0]["score"], ms[0]) if ms else (0, None)
        top, pick = 0, None
        for sub in combinations_with_replacement("ABCDEFGHIJKLMNOPQRSTUVWXYZ?", b):
            for m in sweeps_by_rack.get(tuple(sorted(fixed + list(sub))), ()):
                if m["score"] <= top:
                    break
                sc = m["score"] - sum(m["blank_cost"][ch][sub.count(ch) - 1]
                                      for ch in set(sub) if ch != "?")
                if sc > top:
                    top, pick = sc, m
        return top, pick

    def reply(rack):
        b = rack["?"]
        sw_score, sw = sweep_for(sorted(ch for ch in rack.elements() if ch != "?"), b)
        top = 0
        for sc, need, _n in threats:
            if sc <= sw_score:
                break
            if all(rack[ch] >= n for ch, n in need):
                top = sc
                break
        if b:
            top = with_blanks(rack, b, max(top, sw_score))
        can_empty = bool(bag_after) and (
            (sw is not None and bag_after <= 7)
            or any(all(rack[ch] >= n for ch, n in need) for need in emptiers)
            or (b > 0 and can_empty_with_blanks(rack, b)))
        if sw is not None and sw_score >= top:
            return sw_score, sw, can_empty
        return top, None, can_empty

    worst = None
    if best:
        sc, ref = max(best.values(), key=itemgetter(0))
        worst = _reply_move(board, sc, ref)
    for m in sweeps:
        if worst is None or m["score"] > worst["score"]:
            worst = m
    return reply, worst, sweeps


def expected_threat(board, unseen, words, samples=3000, seed=1, cap=2, bag_after=0):
    """Opponent's reply, weighted by the chance they hold the tiles (exact if <= 7 unseen).
    Returns (expected score, worst reply, chance of worst, exact?, chance they empty the bag,
    (example premium sweep, chance their best reply is one))."""
    import random
    pool = "".join(unseen.elements())
    if len(pool) <= 7:
        m, _ = opponent_threat(board, unseen, words)
        return (m["score"] if m else 0), m, 1.0, True, 0.0, (None, 0.0)
    reply, worst, sweeps = reply_model(board, unseen, words, cap, bag_after)
    if worst is None:
        return 0, None, 0.0, False, 0.0, (None, 0.0)
    rng = random.Random(seed)
    tiles = list(pool)
    total, worst_hits, empties, sweep_hits, seen = 0, 0, 0, 0, Counter()
    for _ in range(samples):
        score, sw, can_empty = reply(Counter(rng.sample(tiles, 7)))
        total += score
        worst_hits += (score == worst["score"])
        empties += can_empty
        if sw:
            sweep_hits += 1
            seen[(sw["word"], sw["row"], sw["col"], sw["dir"])] += 1
    p_empty = empties / samples if bag_after else 0.0
    example = max(sweeps, key=lambda m: m["score"]) if sweeps else None
    if seen:
        k = seen.most_common(1)[0][0]
        example = next(m for m in sweeps if (m["word"], m["row"], m["col"], m["dir"]) == k)
    return (round(total / samples, 1), worst, worst_hits / samples, False, p_empty,
            (example, sweep_hits / samples))


def swap_moves(rack, bag, best=3):
    """The `best` swaps by leave value of the tiles kept."""
    from itertools import combinations
    rack = rack.upper()
    keeps = set()
    for k in range(len(rack)):  # keep 0..6 tiles; at least one tile is swapped
        for c in combinations(rack, k):
            keeps.add("".join(sorted(c)))
    out = []
    for keep in keeps:
        swapped = "".join(sorted((Counter(rack) - Counter(keep)).elements()))
        out.append(dict(word="SWAP", swap=swapped, row=0, col=0, tiles={}, dir="", score=0,
                        main=0, cross=[], leave=keep))
    for m in out:
        m["lv"] = leave_value(m["leave"], bag)
    out.sort(key=lambda m: (-m["lv"], len(m["leave"])))
    return out[:best]


def where(m):
    if "swap" in m:
        return f"swap {m['swap']:<7}"
    return f"row {m['row']:>2} col {m['col']:>2} {m['dir']:<6}"


def fmt(m):
    if "swap" in m:
        return f"{m['score']:4d}  {'SWAP':<10} {where(m)}             keep  {m['leave'] or '-':<7}"
    extras = ", ".join(f"{w} {s}" for w, s in m["cross"])
    return (f"{m['score']:4d}  {m['word']:<10} row {m['row']:>2} col {m['col']:>2} {m['dir']:<6}"
            f"  leave {m['leave'] or '-':<7} {('+ ' + extras) if extras else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("board")
    ap.add_argument("rack")
    ap.add_argument("--words", default="NWL2023.txt", help="word list; downloaded if missing")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--defend", type=int, default=0,
                    help="for the top N moves (by score + leave), find the opponent's likely reply "
                         "and rank by equity; with the bag at 7 or fewer, the best bag-emptying "
                         "moves are checked too")
    ap.add_argument("--no-swap", action="store_true", help="leave swaps out of the ranking")
    ap.add_argument("--swap-min", type=int, default=7,
                    help="smallest bag that allows a swap (default 7, as in Scrabble)")
    ap.add_argument("--cap", type=int, default=None,
                    help="most tiles in an opponent reply the defense check lists (default by "
                         "bag: 4 if over 40, 5 if 21-40, 7 = every reply if 20 or fewer)")
    ap.add_argument("--last-move", action="store_true",
                    help="the bag is empty and this play ends the game (you emptied the bag and the "
                         "opponent has had their final turn): rank by score alone, check no reply")
    a = ap.parse_args()
    global TRIE_CACHE
    TRIE_CACHE = a.words + ".trie.pkl"
    words = load_words(a.words)
    board = load_board(a.board)
    unseen = unseen_tiles(board, a.rack)
    n = sum(unseen.values())
    bag = max(0, n - 7)
    if a.cap is None:
        a.cap = 4 if bag > 40 else 5 if bag > 20 else 7
    print(f"unseen {n} (bag {bag}, opponent rack {min(7, n)}): "
          + " ".join(k * v for k, v in sorted(unseen.items())))
    moves = solve(board, a.rack, words)
    if a.last_move and bag > 0:
        print(f"ignoring --last-move: the bag still holds {bag}")
        a.last_move = False
    if a.last_move:
        print("last move of the game: ranked by score, no reply to check")
        for m in moves[:a.top]:
            print(fmt(m))
        return
    for m in moves:
        m["lv"] = leave_value(m["leave"], bag)
        m["val"] = m["score"] + m["lv"]
    if not a.no_swap and bag >= a.swap_min:
        for m in swap_moves(a.rack, bag):
            m["val"] = m["lv"]
            moves.append(m)
    moves.sort(key=lambda m: -m["val"])
    print("leave value = move value | score word ...   (ranked by value = score + leave value)")
    for m in moves[:a.top]:
        tag = "  [empties bag]" if 0 < bag <= len(m["tiles"]) else ""
        print(f"{m['lv']:+6.1f} = {m['val']:6.1f} | " + fmt(m) + tag)
    if a.defend:
        pool = moves[:a.defend]
        best_swap = next((m for m in moves if "swap" in m), None)
        if best_swap and best_swap not in pool:
            pool.append(best_swap)  # always compare against the best swap
        if 0 < bag <= 7:  # also check the best plays that empty the bag (turn-count rule)
            pool += [m for m in moves if len(m["tiles"]) >= bag and m not in pool][:3]
        print(f"\ndefense check, {len(pool)} moves, replies up to {a.cap} tiles: "
              "equity = score + leave value - expected reply - turn cost")
        if bag == 0:
            print("bag is empty: this assumes the opponent still has a final turn. If this play "
                  "ends the game, rerun with --last-move.")
        rows = []
        cache = None  # every swap leaves the board unchanged, so its reply is computed once
        for m in pool:
            b_after = max(0, bag - len(m["tiles"]))
            if "swap" in m and cache:
                exp, worst, p, exact, p_empty, sweep = cache
            else:
                exp, worst, p, exact, p_empty, sweep = expected_threat(
                    apply_move(board, m), unseen, words, samples=500, cap=a.cap, bag_after=b_after)
                if "swap" in m:
                    cache = (exp, worst, p, exact, p_empty, sweep)
            cost = round(p_empty * exp, 1)
            rows.append((m["val"] - exp - cost, exp, m, worst, p, exact, b_after, p_empty, cost,
                         sweep))

        def mv(x):
            return f"{x['word']} {x['score']} (row {x['row']} col {x['col']} {x['dir']})"
        for eq, exp, m, worst, p, exact, b_after, p_empty, cost, sweep in sorted(
                rows, key=lambda x: -x[0]):
            if worst:
                wt = mv(worst) + ("" if exact else f", they can play it {p:.0%} of the time")
            else:
                wt = "none"
            if bag == 0:
                turns = ""
            elif b_after == 0:
                turns = " | empties bag"
            elif cost:
                turns = (f" | leaves bag {b_after}, they can empty it {p_empty:.0%}: "
                         f"turn cost {cost:.1f}")
            else:
                turns = ""
            sw = ""
            if not exact:
                sx, sp = sweep
                sw = f" | 7-tile onto 2W/3W: {sp:.0%}" + (f", e.g. {mv(sx)}" if sx and sp else "")
            print(f"equity {eq:6.1f}  {m['word']:<10} {m['score']:3d} {where(m)} "
                  f"leave {m['leave'] or '-':<7} ({m['lv']:+.1f}) | "
                  f"{'exact reply' if exact else 'expected reply'} {exp:.1f}{turns} | "
                  f"worst: {wt}{sw}")


if __name__ == "__main__":
    main()
