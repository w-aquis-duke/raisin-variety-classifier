#!/usr/bin/env bash
set -euo pipefail

project="${1:-raisin-variety-classifier}"
region=us-central1
repository=raisin-images
service=raisin-classifier
registry="${region}-docker.pkg.dev"
image_path="$registry/$project/$repository/$service"

if [[ ! -s artifacts/model.joblib ]]; then
    printf '%s\n' 'Run python3 train.py to create artifacts/model.joblib.' >&2
    exit 1
fi

commit=$(git rev-parse --short HEAD)
model_hash=$(python3 -c 'import hashlib; from pathlib import Path; print(hashlib.sha256(Path("artifacts/model.joblib").read_bytes()).hexdigest()[:12])')
image="$image_path:$commit-$model_hash"

gcloud services enable artifactregistry.googleapis.com run.googleapis.com --project "$project"
if ! gcloud artifacts repositories describe "$repository" \
    --project "$project" --location "$region" >/dev/null 2>&1; then
    gcloud artifacts repositories create "$repository" --repository-format docker \
        --project "$project" --location "$region"
fi

gcloud auth configure-docker "$registry" --quiet
docker build --platform linux/amd64 -t "$image" .
docker push "$image"
digest=$(gcloud artifacts docker images describe "$image" \
    --project "$project" --format='value(image_summary.digest)')
test -n "$digest"

gcloud run deploy "$service" --image "$image_path@$digest" --project "$project" \
    --region "$region" --port 8080 --max-instances 1 --allow-unauthenticated --quiet
