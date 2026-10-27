# Contributing to Zelyr

Welcome! We are thrilled that you want to help evolve Zelyr into the most powerful sovereign AI operating system layer.

## 🛠️ Setting up your Development Environment

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/rutytoi/zelyr.git
   ```
2. **Environment Setup**:
   Zelyr requires Python 3.11+. We recommend using a virtual environment.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. **Core Dependencies**:
   - **Ollama**: Ensure Ollama is running locally (`ollama serve`).
   - **Bubblewrap**: Required for standard sandboxing (`sudo apt install bubblewrap`).
   - **FUSE**: Required for ZelyrFS (`sudo apt install fuse3`).

---

## 🧪 Testing & Verification

We maintain a rigorous regression suite. Never submit a PR without passing all tests.

- **Unit Tests**:
  ```bash
  pytest tests/
  ```
- **Coverage**:
  ```bash
  pytest --cov=zelyr tests/
  ```
- **Type Checking**:
  ```bash
  mypy zelyr/
  ```

---

## 📦 Packaging & Distribution

Before a release, we use the automated package builder to generate artifacts for all platforms:

```bash
python3 -m zelyr.api.cli package --target all
```

This generates `.rpm`, `.deb`, `.AppImage`, and Windows `.exe` installers in the `dist/` directory.

---

## 🤝 Pull Request Guidelines

1. **Branching**: Create a feature branch from `main` (e.g., `feat/zelyr-fs-encryption`).
2. **Atomic Commits**: Keep your commits focused and descriptive.
3. **Documentation**: If you add a new subsystem or tool, you **must** update `ARCHITECTURE.md` and provide a docstring in the code.
4. **Verification**: Link your PR to a relevant issue and include the `pytest` output in your PR description.

---

## 🛡️ Security First

If you discover a security vulnerability (especially in the sandbox or kernel execution layer), please **do not** open a public issue. Email the core maintainers directly at `security@zelyr-ai.io`.

---

## ⚖️ Code of Conduct

Be respectful, be technical, and keep Zelyr sovereign. We follow the standard Contributor Covenant.
