#!/usr/bin/env bash
# Render a job manifest: substitute only our infra placeholders, leave the inline
# bash in the job ($B, $f, $(date), ${qrc}) untouched. Source infra/env.sh first.
# Usage: tools/render-job.sh prune/glm46-prune-job.yaml | kubectl apply -f -
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$here/infra/env.sh"
exec envsubst '${S3_BUCKET_NAME} ${K8S_NS} ${K8S_SA} ${AWS_REGION} ${AWS_ACCOUNT} ${EKS_CLUSTER} ${KARPENTER_NODE_ROLE} ${AWS_PROFILE}' < "$1"
