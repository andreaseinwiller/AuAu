#!/usr/bin/env bash
#
# run.sh - single entry point for the eval reproducibility pipeline.
#
# Usage:
#   ./run.sh <stage> [stage ...]
#   ./run.sh all
#
# Stages (this is also the order "all" runs them in):
#   build       Build the Docker image (uv-managed Python + R); pulls the full pyproject.toml
#   download    Download data from Zenodo -> resources/zenodo_archive/
#   extract     Unpack into resources/output/, copy tidy CSVs into eval/data/tidy/
#   shell       Open an interactive shell in the container (debugging)
#   clean       Remove extracted/derived data (resources/output, eval/data/tidy)
#   clean-all   Clean + also remove downloaded archives (resources/zenodo_archive)
#
# Examples:
#   ./run.sh all                      # Build the image + fetch + unpack data
#   ./run.sh download extract         # Just refresh the data
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="auau-eval-reprod"
IMAGE_TAG="v0.0.1"
CONTAINER_WORKDIR="/workspace"
LOG_DIR="${REPO_ROOT}/resources/logs"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="${LOG_DIR}/run_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

# Mirror everything to a per-run logfile as well as stdout, for provenance.
exec > >(tee -a "${RUN_LOG}") 2>&1

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required but was not found on PATH. Install Docker and re-run." >&2
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
docker_run() {
  docker run --rm \
    -v "${REPO_ROOT}:${CONTAINER_WORKDIR}" \
    -w "${CONTAINER_WORKDIR}" \
    -e PYTHONHASHSEED=0 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e HOME=/tmp \
    --user "$(id -u):$(id -g)" \
    "${IMAGE_NAME}:${IMAGE_TAG}" "$@"
}

stage_build() {
  require_docker
  log "Building docker image ${IMAGE_NAME}:${IMAGE_TAG}"
  docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" "${REPO_ROOT}"
  log "Image ID: $(docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" --format='{{.Id}}')"
  log "Recording environment fingerprint"
  docker_run python scripts/env_report.py --output "results/logs/environment_${TIMESTAMP}.json"
}

stage_download() {
  log "Downloading data from Zenodo (see resources/config/data_manifest.yaml)"
  docker_run python scripts/download_zenodo_data.py --manifest resources/config/data_manifest.yaml
}

stage_extract() {
  log "Extracting into resources/output/ and eval/data/tidy/"
  docker_run python scripts/extract_zenodo_data.py --manifest resources/config/data_manifest.yaml
}

stage_shell() {
  require_docker
  log "Opening interactive shell in container"
  docker run --rm -it \
    -v "${REPO_ROOT}:${CONTAINER_WORKDIR}" \
    -w "${CONTAINER_WORKDIR}" \
    "${IMAGE_NAME}:${IMAGE_TAG}" bash
}

stage_clean() {
  log "Removing extracted/derived data (resources/output, eval/data/tidy)"
  rm -rf "${REPO_ROOT}/resources/output" "${REPO_ROOT}/eval/data/tidy"
  if [[ "${1:-}" == "--all" ]]; then
    log "Also removing raw downloads (resources/zenodo_archive)"
    rm -rf "${REPO_ROOT}/resources/zenodo_archive"
  fi
}

usage() {
  sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

log "Run started. Git commit: $(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo 'not a git repo / no commits yet')"

for stage in "$@"; do
  case "${stage}" in
    build)      stage_build ;;
    download)   stage_download ;;
    extract)    stage_extract ;;
    shell)      stage_shell ;;
    clean)      stage_clean ;;
    clean-all)  stage_clean --all ;;
    all)
      stage_build
      stage_download
      stage_extract
      ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown stage: ${stage}" >&2
      usage
      exit 1
      ;;
  esac
done

log "Done. Full log: ${RUN_LOG}"