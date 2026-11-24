# TabPFN v2 tabular prediction API - environment

Paste into Railway's variables panel. Everything else lives in the image, because a template variable holding a literal is dropped at generate time and republishes as a blank required field.

```
API_KEY="${{secret(32)}}"
```

## Optional

| Variable | Default | Values |
|---|---|---|
| `MAX_TRAIN_ROWS` | `1000` | Most training rows accepted (413 above it); TabPFN-2's CPU limit |
| `MAX_TEST_ROWS` | `1000` | Most rows to score per request |
| `MAX_FEATURES` | `100` | Most columns |
| `MAX_CELLS` | `120000` | Cap on (training + scored rows) x columns |
| `MAX_SECONDS` | `240` | Wall-clock budget from request arrival; past it the request answers 504 (Railway's edge cuts a connection at about 5 minutes) |
| `MAX_PENDING` | `3` | Requests running or waiting; another answers 429 with `Retry-After: 30` |

## Set by the entrypoint at boot

`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `THREADS` (torch's thread count) come from `/sys/fs/cgroup/cpu.max`, because the runtimes size their thread pools from the host's cores and oversubscribe the container.

`PORT` is left to Railway: the healthcheck dials the injected port, so the app honours it.
