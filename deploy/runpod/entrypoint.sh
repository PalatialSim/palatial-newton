#!/usr/bin/env bash
set -euo pipefail

# Keeping the default pod process alive makes this an interactive simulator
# workbench. Run Newton through SSH and write recordings, OVRTX shader cache,
# and PNG output under the mounted /workspace network volume.
exec "$@"
