# AGENTS.md — Behavioral & Empirical Verification Standard

## 1. The Core Law
**If you change code, execute it and verify the resulting behavior.**  
A task is never complete merely because the code compiles, type-checks, passes unit tests, or "looks correct." You must verify the changed behavior using the strongest practical evidence for the specific claim being made.

### Evidence Must Match the Claim
Align the scope of verification to the scope of your changes:
- **User-Facing Workflows:** Verify via interactive UI, CLI execution, or E2E automation. (Passing a unit test does *not* prove a TUI renders or accepts input).
- **Public APIs & Module Integrations:** Verify via consumer test scripts exercising real call sites.
- **Internal Invariants & Edge Cases:** Verify via targeted automated unit test suites.
- **Syntax & Interfaces:** Verify via compilers, type checkers, and linters. (A clean build proves structural validity, not functional correctness).

The goal is never: *"This should work."*  
The goal is always: *"I executed it, exercised the behavior, and verified the result."*

---

## 2. Discover the Project & Environment
Before modifying code or running commands:
1. **Inspect Repo Conventions & Documentation Index:** Read [`README.md`](README.md), [`pyproject.toml`](pyproject.toml), and the documentation hub [`docs/INDEX.md`](docs/INDEX.md). Prefer documented project commands over guessing generic workflows.
2. **Inspect Source Code First (Anti-Hallucination Mandate):** Never assume past reports, comments, or naming conventions reflect the true repository state. Treat all presumed paths and symbols as leads to verify in code before proceeding.
3. **Active Project Technology Stack:**
   - **Runtime:** Python 3.10–3.11 (`.venv` in repository root).
   - **Toolchain & Runner:** [`uv`](https://docs.astral.sh/uv/) (`uv run ...`, `uv sync`, `uv build --wheel`).
   - **Test Framework:** `pytest` (`uv run pytest -v`).
   - **Linter & Formatter:** `ruff` (`uv run ruff check .`).
4. **Mandatory Project Commands:**
   - **Full Test Suite:** `uv run pytest -v` (runs the consolidated test suite in `tests/test_*.py`).
   - **Static Linter Audit:** `uv run ruff check .`
   - **CLI Smoke Test:** `uv run zelyr --help`
5. **Verified "Where to Look First" Repository Topology:**
   - **Entry Points:** [`zelyr/cli/main.py`](zelyr/cli/main.py) (CLI dispatch), [`zelyr/cli/repl.py`](zelyr/cli/repl.py) (Inline REPL), [`zelyr/server/daemon.py`](zelyr/server/daemon.py) (Daemon).
   - **Active ReAct Orchestrator:** [`zelyr/agents/native_orchestrator.py`](zelyr/agents/native_orchestrator.py).
   - **Actuator Routing & Negative Gating:** [`zelyr/agents/actuator_arbiter.py`](zelyr/agents/actuator_arbiter.py), [`zelyr/core/fast_path_router.py`](zelyr/core/fast_path_router.py).
   - **Dynamic Tools Directory:** `~/.config/zelyr/tools.d/` (configured in [`zelyr/core/plugins.py`](zelyr/core/plugins.py)).
   - **SSRF & Network Security:** [`zelyr/security/network.py`](zelyr/security/network.py).
   - **Shell Execution Authorization:** [`zelyr/tools/__init__.py`](zelyr/tools/__init__.py) (`_request_shell_authorization`).
   - **Configuration & Context Precedence:** [`zelyr/config.py`](zelyr/config.py).
   - **Swarm Blackboard:** [`zelyr/memory/blackboard.py`](zelyr/memory/blackboard.py).
   - **Consolidated Test Battery:** [`tests/`](tests/) (configured via [`pyproject.toml`](pyproject.toml)).
6. **Non-Negotiable Headless & Isolation Safeguards:**
   - **Headless Only:** Tests must run without opening browser windows or desktop popups. Calls to `webbrowser.open` and subprocesses must be mocked.
   - **Zero Unprompted Git Mutations:** Never run `git add`, `git commit`, `git push`, or `git reset` unless explicitly ordered by the user. Leave changes unstaged in the working tree.
   - **Preserve Pre-Existing Work:** Never overwrite, delete, or absorb pre-existing files without explicit instruction.
   - **Fail-Closed Security:** Shell commands must fail closed in non-interactive sessions unless `ZELYR_TESTING=1` or `ZELYR_AUTOPILOT=1` is explicitly set.

---

## 3. The Verification Lifecycle & Feedback Loop
Every task follows this execution sequence:

```
[1. Baseline / Reproduce] ──> [2. Targeted Change] ──> [3. Build & Static Checks]
                                                                  │
   ┌────────────────────── ◄ Diagnose & Re-verify ◄ ──────────────┘
   ▼
[4. Behavioral Run] ──> [5. Regression Paths] ──> [6. Diff Audit & Cleanup] ──> [7. Report]
```

1. **Baseline / Reproduce:**
   - **Bugs:** Reproduce the failure before modifying code. If blocked by missing hardware, display servers, or external credentials, explicitly state the limitation before proceeding.
   - **Features & Refactors:** Inspect and run the existing baseline workflow.
2. **Targeted Implementation:** Make minimal, focused edits addressing the root cause.
3. **Build & Automated Verification:** Run project-defined builds, linters, and relevant test suites.
4. **Behavioral Verification:** Execute the real application or affected workflow.
5. **Regression Verification:** Test nearby critical paths that could reasonably be impacted.
6. **The Active Feedback Loop:** **If verification reveals a regression or unexpected failure, you must fix it and repeat verification from the affected stage.** Never stop at merely reporting a failure.
7. **Diff Audit & Teardown:** Run `git diff` to ensure no debug code, secrets, or unintentional modifications remain. Terminate any background processes or test artifacts.
8. **Truthful Reporting:** Present an accurate summary of what was directly proven versus what could not be exercised.

---

## 4. Runtime Rules by Application Type

Agents execute inside non-interactive subshells. Adapt your execution pattern to prevent blocked sessions:

### A. CLI & Batch Applications
- Execute the binary with realistic arguments.
- Verify stdout, stderr, and explicit exit codes (`echo $?`). An exit code of `0` is necessary, but you must verify that the resulting output or state matches expectations.

### B. Interactive TUIs & Terminal Prompts
*Never launch a blocking interactive application bare in a subshell without an input channel (it will hang waiting for stdin).*
- **Linear Prompts:** Pipe predictable input streams:
  ```bash
  printf "help\nquit\n" | ./bin/app
  ```
- **Full-Screen TUIs:** Use terminal automation (`pexpect`, `expect`, or project UI harnesses) to launch the binary in a pseudo-terminal (PTY), assert expected screen buffers, send navigation keys, exercise the feature, and exit cleanly.
- **Crash Checking:** Commands like `timeout 3s ./bin/app` verify only that the binary avoids an immediate crash or panic. Surviving a timeout is **not** proof that the interface or commands work.

### C. Daemons, Services & Web APIs
- Launch background services with logging redirected and track the PID:
  ```bash
  ./service > /tmp/test_service.log 2>&1 &
  PID=$!
  sleep 1
  kill -0 "$PID" || { cat /tmp/test_service.log; exit 1; }
  ```
- Exercise endpoints via real HTTP/IPC requests (`curl -sSf http://127.0.0.1:8080/health`).
- **Teardown:** Always terminate background processes before finishing:
  ```bash
  kill "$PID" 2>/dev/null || true
  ```

### D. GUI Applications
- Use browser/GUI drivers (Playwright, Puppeteer) or virtual displays (`xvfb-run`) when available.
- If no display server or automation driver exists, test the underlying controllers and headless entrypoints, and state the display limitation explicitly.
- Never claim to have clicked or visually verified an element without an automated harness driving it.

### E. Libraries & SDKs
- If no application entrypoint exists, construct an ephemeral runner script:
  ```bash
  python -c "import pkg; assert pkg.run() == 'expected'"
  ```
- Exercise the public API across both typical inputs and relevant edge cases.

---

## 5. Side Effects & Integration Boundaries
When an operation mutates state, verify the side effect directly:
- **Filesystem:** Inspect created/modified files to confirm contents, formats, and permissions.
- **Databases:** Query tables directly to confirm records were created, updated, or deleted.
- **Configuration:** Restart the target service to confirm changes take effect cleanly.
- **Processes:** Confirm background services run (`kill -0 "$PID"`), handle requests, and terminate cleanly.

---

## 6. Safety & Anti-Hallucination
- **Safe Testing Only:** When verifying features that run shell commands, use observable, non-destructive commands (`whoami`, `echo "smoke test"`, `uname`). Never execute destructive commands (`rm`, system file alterations) or expose real secrets.
- **No Manufactured Claims:** Never state an app "works" if you only verified that it compiled or passed unit tests. State exact limitations plainly:
  > *"Unit tests pass, but runtime verification was skipped because [exact reason]."*
- **Do Not Commit Prematurely:** Do not create a git commit before verification is complete unless explicitly instructed to do so.

---

## 7. Completion Report Format
Conclude tasks with a concise verification summary. Do not paste verbose terminal logs unless diagnosing an unresolved error:

```markdown
### Verification Summary
- **Target Workflow:** [Exact command, script, or workflow tested]
- **Verification Tier:** [E2E Automation / Runtime Execution / Unit Tests / Build & Typecheck]
- **Commands Executed:** [Primary verification commands run]
- **Observed Behavior:** [Concrete result: exit code 0, expected buffer string, state verified]
- **Automated Tests:** [PASS / FAIL / NOT RUN / NOT APPLICABLE]
- **Limitations:** [None, or describe missing external services/displays]
- **Diff Cleanliness:** [Clean git diff confirmed]
```
