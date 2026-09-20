# Cross-Platform Setup Guide (Windows / macOS / Linux)

The Python code in this project (`app.py`, `retriever.py`, `generator.py`,
`build_corpus.py`) is 100% pure Python — no OS-specific calls, no hardcoded
path separators, every file read specifies `encoding="utf-8"` explicitly.
It runs identically on Windows, macOS, and Linux. **The only thing that
actually differs by platform is shell syntax** — how you create/activate a
virtual environment and how you set an environment variable. This guide
gives you the exact command for each.

Jump to your platform: [Windows (cmd.exe)](#windows-cmdexe) ·
[Windows (PowerShell)](#windows-powershell) · [macOS](#macos) ·
[Linux](#linux)

---

## Windows (cmd.exe)

```cmd
:: 1. Navigate to the project folder
cd C:\path\to\your\project

:: 2. Create a virtual environment
python -m venv venv

:: 3. Activate it (prompt should show "(venv)")
venv\Scripts\activate.bat

:: 4. Install dependencies
pip install -r requirements.txt

:: 5. Set your provider and API key (pick ONE block)
set LLM_PROVIDER=anthropic
set ANTHROPIC_API_KEY=sk-ant-your-key-here

:: OR
set LLM_PROVIDER=openai
set OPENAI_API_KEY=sk-your-key-here

:: OR
set LLM_PROVIDER=kimi
set MOONSHOT_API_KEY=sk-your-key-here

:: OR
set LLM_PROVIDER=openrouter
set OPENROUTER_API_KEY=sk-or-v1-your-key-here

:: 6. (Optional) rebuild chunks.json if you edited papers/*.txt
python build_corpus.py

:: 7. Run the tests
python smoke_test.py
python test_generator_providers.py

:: 8. Run the app
python app.py
:: Then open http://127.0.0.1:5000/ in a browser

:: 9. When done
deactivate
```

**cmd.exe-specific gotchas:**
- `set VAR=value` — **no quotes**, no space around `=`. `set VAR =value`
  (space before `=`) creates a variable literally named `VAR ` with a
  trailing space, not `VAR` — a real, easy-to-make mistake.
- `set` only applies to the *current* window, for the current session. Close
  the window and it's gone — you'll need to `set` it again in a new window.
- To check what a variable is currently set to: `echo %VAR_NAME%`. If it
  prints back the literal text `%VAR_NAME%` instead of a value, it was
  never set in this window.
- Use `python`, not `python3`, unless you specifically installed a
  `python3.exe`. If `python` isn't recognized at all, try `py -3` instead.

---

## Windows (PowerShell)

```powershell
# 1. Navigate to the project folder
cd C:\path\to\your\project

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it (prompt should show "(venv)")
venv\Scripts\Activate.ps1

# If you get an error about execution policies being disabled, run this
# once (only affects this window), then retry the line above:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set your provider and API key (pick ONE block)
$env:LLM_PROVIDER = "anthropic"
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"

# OR
$env:LLM_PROVIDER = "openai"
$env:OPENAI_API_KEY = "sk-your-key-here"

# OR
$env:LLM_PROVIDER = "kimi"
$env:MOONSHOT_API_KEY = "sk-your-key-here"

# OR
$env:LLM_PROVIDER = "openrouter"
$env:OPENROUTER_API_KEY = "sk-or-v1-your-key-here"

# 6. (Optional) rebuild chunks.json if you edited papers/*.txt
python build_corpus.py

# 7. Run the tests
python smoke_test.py
python test_generator_providers.py

# 8. Run the app
python app.py
# Then open http://127.0.0.1:5000/ in a browser

# 9. When done
deactivate
```

**PowerShell-specific gotchas (the #1 source of confusion on Windows):**
- PowerShell's `set` is an **alias for `Set-Variable`**, which creates a
  PowerShell *session* variable — **not** an environment variable. Python's
  `os.environ` never sees it. You **must** use `$env:VAR = "value"` syntax
  instead — the colon and quotes are required.
- To check what a variable is currently set to: `echo $env:VAR_NAME` (or
  just `$env:VAR_NAME`). Blank output means it isn't set.
- If `Activate.ps1` refuses to run with a script-execution error, that's
  Windows' execution policy blocking unsigned scripts — the
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` line above
  fixes it for the current window only (doesn't weaken security
  permanently).

---

## macOS

```bash
# 1. Navigate to the project folder
cd /path/to/your/project

# 2. Create a virtual environment
python3 -m venv venv

# 3. Activate it (prompt should show "(venv)")
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set your provider and API key (pick ONE block)
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-your-key-here

# OR
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-your-key-here

# OR
export LLM_PROVIDER=kimi
export MOONSHOT_API_KEY=sk-your-key-here

# OR
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-v1-your-key-here

# 6. (Optional) rebuild chunks.json if you edited papers/*.txt
python3 build_corpus.py

# 7. Run the tests
python3 smoke_test.py
python3 test_generator_providers.py

# 8. Run the app
python3 app.py
# Then open http://127.0.0.1:5000/ in a browser

# 9. When done
deactivate
```

**macOS-specific notes:**
- Use `python3`/`pip3` explicitly — plain `python` may not exist or may
  point at an old system Python 2 on some macOS versions.
- `export VAR=value` — no quotes needed for a simple value, but quote it if
  the value contains spaces or special shell characters:
  `export SOME_VAR="value with spaces"`.
- Like the other shells, `export` only lasts for the current terminal
  session/tab. To check a value: `echo $VAR_NAME`.
- To make a variable persist across every new terminal session, add the
  `export` line to `~/.zshrc` (default shell on modern macOS) or `~/.bash_
  profile` (if you're using bash), then restart the terminal or run `source
  ~/.zshrc`.

---

## Linux

```bash
# 1. Navigate to the project folder
cd /path/to/your/project

# 2. Create a virtual environment
python3 -m venv venv

# 3. Activate it (prompt should show "(venv)")
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set your provider and API key (pick ONE block)
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-your-key-here

# OR
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-your-key-here

# OR
export LLM_PROVIDER=kimi
export MOONSHOT_API_KEY=sk-your-key-here

# OR
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-v1-your-key-here

# 6. (Optional) rebuild chunks.json if you edited papers/*.txt
python3 build_corpus.py

# 7. Run the tests
python3 smoke_test.py
python3 test_generator_providers.py

# 8. Run the app
python3 app.py
# Then open http://127.0.0.1:5000/ in a browser

# 9. When done
deactivate
```

**Linux-specific notes:**
- Some distros' `python3-venv` isn't installed by default. If step 2 fails
  with a message about `ensurepip`, install it first:
  `sudo apt install python3-venv` (Debian/Ubuntu) or the equivalent for
  your distro (`dnf install python3-venv` on Fedora, etc.).
- Everything else is identical to macOS — both use bash/zsh-family shells
  and the same `export` syntax.
- To persist a variable across sessions, add the `export` line to
  `~/.bashrc` (or `~/.zshrc` if you use zsh), then `source ~/.bashrc`.

---

## One command that's identical on all three platforms

Once the venv is activated and dependencies installed, these commands are
**typed exactly the same way** regardless of OS — only `python` vs
`python3` differs (Windows uses `python`; macOS/Linux use `python3` unless
you've aliased it):

| Task | Windows | macOS / Linux |
|---|---|---|
| Rebuild the corpus | `python build_corpus.py` | `python3 build_corpus.py` |
| Run the Flask smoke tests | `python smoke_test.py` | `python3 smoke_test.py` |
| Run the provider tests | `python test_generator_providers.py` | `python3 test_generator_providers.py` |
| Run the app | `python app.py` | `python3 app.py` |

## Quick self-check: "is my environment variable actually set?"

Run this right before `python app.py`, in the exact same window:

| Shell | Command |
|---|---|
| cmd.exe | `echo %LLM_PROVIDER%` |
| PowerShell | `echo $env:LLM_PROVIDER` |
| bash / zsh (macOS/Linux) | `echo $LLM_PROVIDER` |

If it prints your value back, you're good. If it prints nothing (or, in
cmd.exe, prints the literal unexpanded `%LLM_PROVIDER%`), the variable
was never set in this window — the single most common cause of the
"ANTHROPIC_API_KEY is not set" error even after you *did* run a `set`/
`export` command, somewhere.

## Folder layout (identical on every OS)

```
your-project-folder/
├── app.py
├── retriever.py
├── generator.py
├── build_corpus.py
├── chunks.json
├── requirements.txt
├── README.md
├── PROJECT_DOCUMENTATION.md
├── CROSS_PLATFORM_SETUP.md      (this file)
├── INTERVIEW_PREP.md
├── smoke_test.py
├── test_generator_providers.py
├── papers/
│   ├── transformer.txt
│   ├── rag.txt
│   └── gpt3.txt
└── templates/
    └── index.html
```

Forward slashes above are just this document's convention for showing a
folder tree — on Windows, this same structure appears with backslashes in
File Explorer (`your-project-folder\papers\transformer.txt`), but you never
need to type a path with a separator yourself for any command in this
guide; `cd` into the top folder and every command above works as written.
