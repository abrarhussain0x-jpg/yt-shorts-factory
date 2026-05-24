#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# scripts/install_deps.sh — One-shot system dependency installer
# Detects OS and installs ffmpeg, Python 3.11+, creates venv,
# installs pip dependencies, and downloads the Whisper base model.
# ──────────────────────────────────────────────────────────
set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Detect OS ─────────────────────────────────────────────
detect_os() {
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "macos"
    elif [[ -f /etc/debian_version ]] || [[ -f /etc/lsb-release ]]; then
        echo "ubuntu"
    elif [[ -f /etc/redhat-release ]]; then
        echo "fedora"
    elif grep -qi microsoft /proc/version 2>/dev/null; then
        echo "wsl"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
info "Detected OS: $OS"

# ── Check Python version ──────────────────────────────────
PYTHON_CMD=""
for cmd in python3.11 python3.12 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PY_VERSION=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
        if [[ "$PY_MAJOR" -ge 3 ]] && [[ "$PY_MINOR" -ge 10 ]]; then
            PYTHON_CMD="$cmd"
            info "Found Python $PY_VERSION at $(command -v $cmd)"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    warn "Python 3.10+ not found. Installing..."
    case "$OS" in
        macos)
            if ! command -v brew &>/dev/null; then
                error "Homebrew not found. Install it from https://brew.sh"
            fi
            brew install python@3.11
            PYTHON_CMD="python3.11"
            ;;
        ubuntu|wsl)
            sudo apt update
            sudo apt install -y python3.11 python3.11-venv python3-pip
            PYTHON_CMD="python3.11"
            ;;
        fedora)
            sudo dnf install -y python3.11 python3.11-pip
            PYTHON_CMD="python3.11"
            ;;
        *)
            error "Cannot auto-install Python on this OS. Please install Python 3.10+ manually."
            ;;
    esac
fi

# ── Check / Install FFmpeg ────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
    warn "FFmpeg not found. Installing..."
    case "$OS" in
        macos)
            brew install ffmpeg
            ;;
        ubuntu|wsl)
            sudo apt update
            sudo apt install -y ffmpeg
            ;;
        fedora)
            sudo dnf install -y ffmpeg ffmpeg-devel
            ;;
        *)
            error "Cannot auto-install FFmpeg on this OS. Install it from https://ffmpeg.org/download.html"
            ;;
    esac
else
    FF_VERSION=$(ffmpeg -version 2>/dev/null | head -1 | grep -oP '\d+\.\d+' | head -1)
    info "FFmpeg found: version $FF_VERSION"
fi

# ── Create virtual environment ────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment at $VENV_DIR..."
    $PYTHON_CMD -m venv "$VENV_DIR"
else
    info "Virtual environment already exists at $VENV_DIR"
fi

# ── Activate venv and install pip dependencies ────────────
info "Installing pip dependencies..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel
pip install -r "$PROJECT_DIR/requirements.txt"
pip install -e "$PROJECT_DIR"

# ── Download Whisper base model ───────────────────────────
info "Downloading Whisper 'base' model (this may take a few minutes)..."
$PYTHON_CMD -c "
import whisper
print('Loading Whisper base model...')
model = whisper.load_model('base')
print('Whisper base model downloaded and cached successfully.')
"

# ── Copy .env.example if .env doesn't exist ───────────────
if [[ ! -f "$PROJECT_DIR/.env" ]] && [[ -f "$PROJECT_DIR/.env.example" ]]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    info "Copied .env.example → .env (edit it to customise settings)"
fi

# ── Create assets directory ───────────────────────────────
mkdir -p "$PROJECT_DIR/assets"

# ── Completion checklist ──────────────────────────────────
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  ✅ Installation Complete!${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "  Checklist:"
echo "    ✅ Python 3.10+ installed"
echo "    ✅ FFmpeg installed"
echo "    ✅ Virtual environment created"
echo "    ✅ pip dependencies installed"
echo "    ✅ Whisper base model cached"
echo "    ✅ .env configuration file created"
echo ""
echo "  Next steps:"
echo "    1. Place your logo at: assets/logo.png"
echo "    2. Edit .env to customise settings"
echo "    3. Activate venv: source .venv/bin/activate"
echo "    4. Run: python main.py run --url https://www.youtube.com/watch?v=VIDEO_ID"
echo ""
