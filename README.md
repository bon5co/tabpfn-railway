# tabpfn-railway

Tabular prediction API for [Railway](https://railway.com), CPU only. Serves [TabPFN v2](https://github.com/PriorLabs/TabPFN) (Prior Labs, Nature 2025), a pretrained transformer that classifies or regresses a small table in one forward pass with no training loop, behind one authenticated HTTP API.

Built with PriorLabs-TabPFN.

- `POST /v1/predict` (JSON): `task` (`classification` or `regression`), `X_train` (rows of numbers, strings for categorical columns, or null), `y_train`, `X_test`, optional `quantiles` (regression), `n_estimators` (1 to 16, default 4), `seed`. Classification returns `predictions`, `classes` and `probabilities`; regression returns `predictions` and, when asked, `quantiles`.
- `GET /v1/models`: what is served and the limits in force.
- `GET /health` (503 until both checkpoints are loaded and warm, then 200).
- Auth: `Authorization: Bearer $API_KEY` on every route except `/health`. An empty `API_KEY` refuses to start.
- Env: see `env.md`; `API_KEY` is the only required one.
- Limits (CPU): 1,000 training rows, 1,000 rows to score, 100 columns, 120,000 cells, 10 classes.

Weights are baked into the image from two ungated Hugging Face repositories pinned to commit SHAs (`Prior-Labs/TabPFN-v2-clf`, `Prior-Labs/TabPFN-v2-reg`), the TabPFN-2 checkpoints, the only TabPFN weights licensed for commercial use. The package's newer weights (2.5 to 3.5) are non-commercial and gated and are never loaded. Data is held in memory for one request and never logged or stored. Wrapper code is Apache-2.0 (`LICENSE`); weights and the licence's attribution requirement are in `NOTICE` and `LICENSE-priorlabs`.

Image: `ghcr.io/bon5co/tabpfn-railway`.
