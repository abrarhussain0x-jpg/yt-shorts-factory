#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# scripts/watch_folder.sh — Watch a folder for new URL files
# When a .txt file is dropped in the watched folder, reads
# URLs from it, runs the batch pipeline, then moves the file
# to a processed/ subdirectory.
# ──────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

WATCH_DIR="${1:-$HOME/Desktop/shorts-queue}"
PROCESSED_DIR="$WATCH_DIR/processed"

# ── Setup directories ─────────────────────────────────────
mkdir -p "$WATCH_DIR"
mkdir -p "$PROCESSED_DIR"

# ── Activate venv ─────────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    source "$VENV_DIR/bin/activate"
else
    echo "⚠ Virtual environment not found. Run 'make install' first."
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📂 Watching: $WATCH_DIR"
echo "  ✅ Processed: $PROCESSED_DIR"
echo "  Drop .txt files with YouTube URLs (one per line)"
echo "  Press Ctrl+C to stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Detect file watcher tool ──────────────────────────────
WATCHER=""
if command -v fswatch &>/dev/null; then
    WATCHER="fswatch"
elif command -v inotifywait &>/dev/null; then
    WATCHER="inotifywait"
else
    echo "⚠ No file watcher found (fswatch or inotifywait)."
    echo "  Installing polling fallback..."
    WATCHER="poll"
fi

process_file() {
    local filepath="$1"
    local filename=$(basename "$filepath")

    echo "📄 New file detected: $filename"

    # Wait for file to be fully written
    sleep 2

    # Run batch processing
    bash "$SCRIPT_DIR/batch_run.sh" "$filepath"

    # Move to processed
    mv "$filepath" "$PROCESSED_DIR/$filename"
    echo "📦 Moved to: $PROCESSED_DIR/$filename"
}

# ── Watch loop ────────────────────────────────────────────
case "$WATCHER" in
    fswatch)
        fswatch -o --event Created --event MovedTo "$WATCH_DIR" | while read -r _; do
            for f in "$WATCH_DIR"/*.txt; do
                if [[ -f "$f" ]]; then
                    process_file "$f"
                fi
            done
        done
        ;;
    inotifywait)
        inotifywait -m -e create -e moved_to --format '%w%f' "$WATCH_DIR" | while read -r filepath; do
            if [[ "$filepath" == *.txt ]]; then
                process_file "$filepath"
            fi
        done
        ;;
    poll)
        echo "🔄 Using polling mode (checking every 10 seconds)..."
        while true; do
            for f in "$WATCH_DIR"/*.txt; do
                if [[ -f "$f" ]]; then
                    process_file "$f"
                fi
            done
            sleep 10
        done
        ;;
esac
