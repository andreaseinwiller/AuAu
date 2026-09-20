# syntax=docker/dockerfile:1

FROM python:3.12-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        gfortran \
        libblas-dev \
        liblapack-dev \
        r-base \
    && rm -rf /var/lib/apt/lists/*

# R packages for eval/reliability_psych.r + eval/cfa_sem.r.
ENV CRAN_SNAPSHOT_DATE=2026-06-01
RUN Rscript -e "options(repos = c(CRAN = paste0('https://packagemanager.posit.co/cran/', Sys.getenv('CRAN_SNAPSHOT_DATE')))); install.packages('remotes')" \
    && Rscript -e "remotes::install_version('psych', version = '2.5.6', upgrade = 'never')" \
    && Rscript -e "remotes::install_version('lavaan', version = '0.6-21', upgrade = 'never')" \
    && Rscript -e "remotes::install_version('semTools', version = '0.5-7', upgrade = 'never')" \
    && Rscript -e "pkgs <- c(psych='2.5.6', lavaan='0.6-21', semTools='0.5-7'); ip <- installed.packages(); for (p in names(pkgs)) { if (!(p %in% rownames(ip))) stop(paste(p, 'not installed')); got <- as.character(ip[p, 'Version']); if (got != pkgs[[p]]) stop(sprintf('%s: expected %s, got %s', p, pkgs[[p]], got)) }; cat('All pinned R package versions verified.\n')"

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

ENV PATH="/opt/venv/bin:${PATH}"

RUN useradd --create-home --uid 1000 repro
USER repro

CMD ["/bin/bash"]
