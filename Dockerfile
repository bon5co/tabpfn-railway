# TabPFN v2 tabular prediction API (PyTorch, CPU), Bearer API_KEY, POST /v1/predict.
# Source: https://github.com/bon5co/tabpfn-railway
# CPU only, weights baked in (about 75 MB), stateless (no volume needed).
# upstream: PriorLabs/TabPFN (the `tabpfn` package, pinned to v10.2.0, the newest release, checked 2026-11-24). The
# weights are the TabPFN-2 checkpoints Prior-Labs/TabPFN-v2-clf@275f19d2 and Prior-Labs/TabPFN-v2-reg@02a44705,
# Prior Labs License 1.1 (Apache 2.0 plus "Built with PriorLabs-TabPFN" attribution).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
    && pip list 2>/dev/null | grep -iE "^(tabpfn|torch|numpy|scikit-learn|fastapi|uvicorn) "

# Weights: two ungated Hugging Face repositories, each pinned to a commit SHA (see fetch_weights.py).
ENV MODEL_ROOT=/models
COPY fetch_weights.py /tmp/fetch_weights.py
RUN python /tmp/fetch_weights.py && find /models -name .cache -type d -prune -exec rm -rf {} + && rm -rf /root/.cache && du -sh /models

# The Prior Labs License (section 10 requires a copy to travel with the weights and the service), the Apache-2.0
# licence of the tabpfn package and this repository's notice.
COPY LICENSE-priorlabs /licenses/PriorLabs-TabPFN-v2-LICENSE
COPY LICENSE-tabpfn-code /licenses/tabpfn-package-LICENSE
COPY NOTICE /licenses/NOTICE

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin tabpfn \
    && chmod -R a+rX /models /licenses
WORKDIR /app
COPY app.py entrypoint.sh NOTICE LICENSE /app/
RUN chmod +x /app/entrypoint.sh

# HF_HUB_OFFLINE: the checkpoints are loaded by path, nothing is fetched at runtime. TABPFN_NO_BROWSER stops the
# package opening a browser login flow; the cache dir is a path the non-root service user can write.
ENV HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TABPFN_NO_BROWSER=1 TABPFN_MODEL_CACHE_DIR=/tmp/tabpfn-cache
USER 1000
EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "/app/app.py"]
