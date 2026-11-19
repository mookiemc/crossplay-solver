# crossplay-solver

Move finder for NYT Crossplay.

```
curl -sSLO https://raw.githubusercontent.com/mookiemc/crossplay-solver/main/crossplay_solver.py
python3 crossplay_solver.py board.txt RACK --top 10 [--defend 8]
```

`board.txt` is 15 lines of 15 characters: `.` empty, CAPITAL tile, lowercase blank.
See `example_board.txt`. The NWL2023 word list downloads on first run.
