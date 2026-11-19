# crossplay-solver

Move finder for NYT Crossplay.

```
curl -sSLO https://raw.githubusercontent.com/mookiemc/crossplay-solver/main/crossplay_solver.py
python3 crossplay_solver.py board.txt RACK --top 10 [--defend 8] [--final 80] [--last-move]
```

`board.txt` is 15 lines of 15 characters: `.` empty, CAPITAL tile, lowercase blank.
See `example_board.txt`. The NWL2023 word list downloads on first run.

`--final N`: with `--defend`, a move that empties the bag is ranked with your own final play
counted. The unseen tiles are split into the opponent's rack and your draw, they make their
top-scoring reply, and you make your top-scoring play. Every split is tried when there are N or
fewer (default 80), otherwise a fixed sample of N. `--final 0` prices the leave instead.

`--last-move`: the bag is empty and your play ends the game (you emptied the bag and the
opponent has had their final turn), so moves rank by score alone.
