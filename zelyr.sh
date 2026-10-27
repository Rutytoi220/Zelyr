#!/usr/bin/env bash
cd "$(dirname "$0")"
QT_QPA_PLATFORM=xcb PYTHONPATH=$(pwd) uv run python -m zelyr.launcher "$@"
