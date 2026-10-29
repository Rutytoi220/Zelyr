#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}[INFO]${NC} Starting Zelyr Pro AppImage Build Process..."

# Ensure running from project root
if [ ! -f "main.py" ] || [ ! -d "scripts" ]; then
    echo -e "${RED}[ERROR]${NC} Please run this script from the project root directory."
    exit 1
fi

# Resolve the release version: $1, then $VERSION, then pyproject.toml.
# A leading "v" is stripped so both "v1.2.0-rc1" and "1.2.0-rc1" are accepted.
VERSION="${1:-${VERSION:-}}"
if [ -z "$VERSION" ]; then
    VERSION="$(python3 - <<'PY' 2>/dev/null || true
import re
try:
    import tomllib
    with open("pyproject.toml", "rb") as fh:
        print(tomllib.load(fh)["project"]["version"])
except Exception:
    with open("pyproject.toml", encoding="utf-8") as fh:
        match = re.search(r'^version\s*=\s*"([^"]+)"', fh.read(), re.M)
        if match:
            print(match.group(1))
PY
)"
fi
VERSION="${VERSION#v}"
if [ -z "$VERSION" ]; then
    echo -e "${RED}[ERROR]${NC} Could not determine version; pass it as \$1 or set VERSION."
    exit 1
fi
APPIMAGE_NAME="Zelyr-v${VERSION}-x86_64.AppImage"
echo -e "${BLUE}[INFO]${NC} Building ${APPIMAGE_NAME}"

# 1. Setup & Compilation (PyInstaller)
echo -e "${BLUE}[INFO]${NC} Creating venv and installing PyInstaller..."
uv venv --python 3.11 --clear
uv pip install pyinstaller

echo -e "${BLUE}[INFO]${NC} Locking and syncing dependencies..."
uv sync --python 3.11

echo -e "${BLUE}[INFO]${NC} Compiling with PyInstaller..."
uv run --python 3.11 pyinstaller zelyr.spec --clean --noconfirm

# 2. AppDir Construction
echo -e "${BLUE}[INFO]${NC} Constructing Zelyr.AppDir..."
APPDIR="Zelyr.AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
mkdir -p "$APPDIR/usr/share/applications"

echo -e "${BLUE}[INFO]${NC} Copying binaries to AppDir..."
# With the COLLECT step in zelyr.spec, PyInstaller generates a directory at dist/Zelyr/
cp -r dist/Zelyr/* "$APPDIR/usr/bin/"

echo -e "${BLUE}[INFO]${NC} Generating Zelyr.desktop..."
cat <<EOF > "$APPDIR/Zelyr.desktop"
[Desktop Entry]
Name=Zelyr Pro
Exec=AppRun
Icon=zelyr-logo
Type=Application
Categories=Utility;Development;
EOF

# Make sure it's also in usr/share/applications/ per AppImage best practices
cp "$APPDIR/Zelyr.desktop" "$APPDIR/usr/share/applications/"

echo -e "${BLUE}[INFO]${NC} Setting up icon..."
if [ -f "zelyr/gui/assets/logo.png" ]; then
    cp "zelyr/gui/assets/logo.png" "$APPDIR/zelyr-logo.png"
else
    echo -e "${BLUE}[INFO]${NC} 'zelyr/gui/assets/logo.png' not found. Creating a default dummy icon..."
    # Generate a tiny 1x1 transparent PNG file as a fallback
    echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" | base64 -d > "$APPDIR/zelyr-logo.png"
fi

echo -e "${BLUE}[INFO]${NC} Creating AppRun symlink..."
# AppRun must point to the executable relative to the AppDir root
cat <<'EOF' > "$APPDIR/AppRun"
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"

# Dependency Check Function
check_lib() {
    if ! /sbin/ldconfig -p | grep -q "$1"; then
        return 1
    fi
    return 0
}

MISSING_LIBS=""
if ! check_lib "libxcb.so.1"; then
    MISSING_LIBS="$MISSING_LIBS\n- libxcb.so.1 (sudo apt install libxcb-cursor0 libxcb1)"
fi
if ! check_lib "libGL.so.1"; then
    MISSING_LIBS="$MISSING_LIBS\n- libGL.so.1 (sudo apt install libgl1)"
fi

if [ -n "$MISSING_LIBS" ]; then
    MSG="Zelyr cannot start because critical host libraries are missing:\n$MISSING_LIBS\n\nPlease install them using your package manager."
    echo -e "\033[0;31m[CRITICAL ERROR]\033[0m $MSG"
    
    if command -v zenity &> /dev/null; then
        zenity --error --title="Zelyr: Missing Dependencies" --text="$MSG"
    elif command -v kdialog &> /dev/null; then
        kdialog --error "$MSG" --title "Zelyr: Missing Dependencies"
    fi
    exit 1
fi

exec "${HERE}/usr/bin/Zelyr" "$@"
EOF
chmod +x "$APPDIR/AppRun"

# 3. AppImage Packaging
echo -e "${BLUE}[INFO]${NC} Checking for appimagetool..."
if [ ! -f "appimagetool-x86_64.AppImage" ]; then
    echo -e "${BLUE}[INFO]${NC} Downloading appimagetool..."
    wget -q https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x appimagetool-x86_64.AppImage
fi

echo -e "${BLUE}[INFO]${NC} Packaging AppImage..."
# Run appimagetool silently to avoid noisy stdout, but set -e ensures we catch exit codes
./appimagetool-x86_64.AppImage "$APPDIR" "$APPIMAGE_NAME" > /dev/null

echo -e "${BLUE}[INFO]${NC} Cleaning up temporary AppDir..."
rm -rf "$APPDIR"

echo -e "${GREEN}[SUCCESS]${NC} Build complete! Generated ${APPIMAGE_NAME}"
