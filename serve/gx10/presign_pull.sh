#!/usr/bin/env bash
# Print a pull script for gx10: S3 objects via presigned URLs (12 h), aria2c -c (resumes after link drops),
# then sha256sum -c against the <key>.sha256 written next to every object by the build jobs.
# aria2c retries without limit (--max-tries=0): on 2026-10-03 a DNS failure on gx10 ended a pull at 70% after
# the default 5 tries. The script writes its PID to /tmp/pull.pid; check it with kill -0, not pgrep -f
# (pgrep -f matches the ssh command line that runs it).
# gx10 has no AWS credentials; the presigned URLs carry temporary ones, so the output stays out of git.
# Usage:
#   source infra/env.sh
#   serve/gx10/presign_pull.sh '~/models' glm46-full-gguf/glm46-abl-mtp-IQ2_XXS-Q5K.gguf > "$SCRATCH/pull.sh"
#   cat "$SCRATCH/pull.sh" | ssht gx10 'cat > /tmp/pull.sh'
#   ssht gx10 'setsid bash /tmp/pull.sh > /tmp/pull.log 2>&1 < /dev/null &'
set -euo pipefail
DEST=${1:?destination dir on gx10}; shift
[ "$#" -ge 1 ] || { echo "usage: $0 <dest dir> <s3 key> [<s3 key> ...]" >&2; exit 1; }
: "${S3_BUCKET:?source infra/env.sh first}"
echo '#!/usr/bin/env bash'
echo 'set -euo pipefail'
echo 'echo $$ > /tmp/pull.pid'
echo "mkdir -p $DEST && cd $DEST"
for key in "$@"; do
  name=$(basename "$key")
  for k in "$key.sha256" "$key"; do
    url=$(aws s3 presign "$S3_BUCKET/$k" --expires-in 43200)
    echo "aria2c -c -x 16 -s 16 --max-tries=0 --retry-wait=10 --file-allocation=none --summary-interval=60 -o '$(basename "$k")' '$url'"
  done
  echo "sha256sum -c '$name.sha256'"
done
echo 'echo PULL_OK $(date -u); rm -f /tmp/pull.pid'
