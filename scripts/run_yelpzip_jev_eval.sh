#!/bin/bash
# Run one documented stage. The Python runner reads TYPESAFE_API_KEY from the
# process environment or the ignored local .env file; this wrapper never echoes it.
set -euo pipefail
exec python -m scripts.yelpzip_jev_eval "$@"
