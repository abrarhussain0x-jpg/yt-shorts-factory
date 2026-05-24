#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# scripts/batch_run.sh — Batch process URLs from a file
# Reads URLs from a text file (one per line, # = comment),
# runs the pipeline for each, and prints a summary.
# ──────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

URL_FILE="${1:-urls.txt}"

# ── Validate ──────────────────────────────────────────────
if [[ ! -f "$URL_FILE" ]]; then
    echo "❌ URL file not found: $URL_FILE"
    echo "Usage: $0 <url_file>"
    echo "  Create a file with one YouTube URL per line."
    echo "  Lines starting with # are ignored."
    exit 1
fi

# ── Activate venv ─────────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    source "$VENV_DIR/bin/activate"
else
    echo "⚠ Virtual environment not found. Run 'make install' first."
    exit 1
fi

# ── Read URLs ─────────────────────────────────────────────
URLS=()
while IFS= read -r line; do
    line=$(echo "$line" | xargs)  # trim whitespace
    if [[ -n "$line" ]] && [[ ! "$line" =~ ^# ]]; then
        URLS+=("$line")
    fi
done < "$URL_FILE"

TOTAL=${#URLS[@]}

if [[ "$TOTAL" -eq 0 ]]; then
    echo "❌ No valid URLs found in $URL_FILE"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🎬 YT Shorts Factory — Batch Run"
echo "  📄 File: $URL_FILE"
echo "  🔢 URLs: $TOTAL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Process each URL ──────────────────────────────────────
SUCCESS=0
FAILED=0
FAILURES=()

for i in "${!URLS[@]}"; do
    URL="${URLS[$i]}"
    NUM=$((i + 1))
    echo ""
    echo "▶ [$NUM/$TOTAL] Processing: $URL"

    if python "$PROJECT_DIR/main.py" run --url "$URL"; then
        SUCCESS=$((SUCCESS + 1))
        echo "✅ [$NUM/$TOTAL] Success"
    else
        FAILED=$((FAILED + 1))
        FAILURES+=("$URL")
        echo "❌ [$NUM/$TOTAL] Failed"
    fi
done

# ── Summary ───────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📊 Batch Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Total:   $TOTAL"
echo "  Success: $SUCCESS"
echo "  Failed:  $FAILED"

if [[ "$FAILED" -gt 0 ]]; then
    echo ""
    echo "  Failed URLs:"
    for url in "${FAILURES[@]}"; do
        echo "    - $url"
    done
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $FAILED
