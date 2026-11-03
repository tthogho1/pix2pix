#!/usr/bin/env bash
# Deploy the inference app to Google Cloud Run, from the same container the
# Hugging Face Space would run (space/Dockerfile).
#
# Cloud Build builds the image remotely (x86, so TensorFlow gets the AVX it
# needs), pushes it to Artifact Registry, and Cloud Run serves it publicly.
#
# Usage (from anywhere; needs `gcloud auth login` and a project with billing):
#   space/deploy_cloudrun.sh <project-id> [region]
set -euo pipefail

PROJECT="${1:?usage: $0 <project-id> [region]}"
REGION="${2:-asia-northeast1}"
SERVICE="${SERVICE:-face2anime}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# Same layout as the Space: infer.py imports pix2pix.py and cyclegan.py from
# its parent directory.
mkdir -p "$STAGE/application/static"
cp "$REPO_ROOT"/space/{Dockerfile,requirements.txt} "$STAGE/"
cp "$REPO_ROOT"/{pix2pix.py,cyclegan.py} "$STAGE/"
cp "$REPO_ROOT/application/infer.py" "$STAGE/application/"
cp "$REPO_ROOT/application/static/index.html" "$STAGE/application/static/"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com --project "$PROJECT"

# Source deploys build as the default compute service account, which new
# projects no longer grant enough to read the uploaded source or push the
# image. roles/run.builder is the role Google documents for exactly this.
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format 'value(projectNumber)')"
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member "serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role roles/run.builder \
  --condition None \
  --quiet > /dev/null

# The GANs alone use ~1.4 GiB; the fp32 diffusion pipeline adds ~6 GB, and
# Cloud Run needs 4 CPUs for more than 8 GiB of memory. Diffusion on CPU takes
# a minute or more per request, hence the long timeout. --cpu-boost speeds the
# cold start, where TensorFlow, torch and four models load. The container
# listens on 7860, the port the Space expects, so Cloud Run is told to use it.
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --source "$STAGE" \
  --port 7860 \
  --memory 12Gi \
  --cpu 4 \
  --cpu-boost \
  --min-instances 0 \
  --max-instances 2 \
  --timeout 900 \
  --allow-unauthenticated \
  --quiet

gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --format 'value(status.url)'
