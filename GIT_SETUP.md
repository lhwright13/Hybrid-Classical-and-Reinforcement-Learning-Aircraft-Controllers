# Git Setup & Workflow Guide

## Repository Status ✅

**Repository initialized**: `/Users/lhwri/controls`
**Branch**: `main`
**Initial commit**: `c9730b9`
**Files tracked**: 33

---

## Initial Commit Summary

### Commit: Initial commit: Multi-level aircraft control system design

**Includes**:
- ✅ 14 design documents (~12,000 lines)
- ✅ Docker configuration (Dockerfile, docker-compose.yml)
- ✅ Python package structure (setup.py, requirements.txt)
- ✅ Configuration files
- ✅ Comprehensive .gitignore and .gitattributes

**Total**: 13,338 insertions across 33 files

---

## Git Configuration

### .gitignore

Ignores:
- Python bytecode and cache (`__pycache__/`, `*.pyc`)
- Build artifacts (`build/`, `dist/`, `*.so`)
- Virtual environments (`venv/`, `.venv/`)
- Logs and data (`logs/`, `*.hdf5`)
- Model checkpoints (`models/*.zip`, `models/*.onnx`)
- Experiment outputs (`wandb/`, `runs/`)
- Media files (`*.mp4`, `screenshots/`)
- IDE configs (`.vscode/`, `.idea/`)
- OS files (`.DS_Store`, `Thumbs.db`)
- Secrets (`*.key`, `credentials.json`)
- **External repos** (`dRehmFlight/`)

### .gitattributes

- Text files: LF line endings (Unix-style)
- Binary files: No text conversion
- Archive exclusions: `.github/`, `tests/`, `docs/`

---

## Recommended Workflow

### 1. Check Status

```bash
git status
```

### 2. Stage Changes

```bash
# Stage all changes
git add .

# Stage specific files
git add design_docs/14_NEW_DOC.md

# Stage by pattern
git add controllers/*.py
```

### 3. Commit Changes

```bash
# Commit with message
git commit -m "Add feature X"

# Commit with detailed message
git commit -m "Add feature X" -m "Detailed description here"

# Commit with editor (for long messages)
git commit
```

### 4. View History

```bash
# One-line format
git log --oneline

# Detailed log
git log

# Graph view
git log --graph --oneline --all

# Show files changed
git log --stat
```

### 5. View Changes

```bash
# Unstaged changes
git diff

# Staged changes
git diff --staged

# Changes in specific file
git diff controllers/types.py
```

---

## Branching Strategy

### Feature Branches

For implementing new features:

```bash
# Create and switch to feature branch
git checkout -b feature/level3-controller

# Make changes and commit
git add controllers/level3.py
git commit -m "Implement Level 3 attitude controller"

# Switch back to main
git checkout main

# Merge feature (if ready)
git merge feature/level3-controller

# Delete feature branch
git branch -d feature/level3-controller
```

### Development Branches

```bash
# For ongoing development
git checkout -b dev

# For specific phases
git checkout -b phase1-foundation
git checkout -b phase2-simulation
```

---

## Recommended Branch Structure

```
main                    # Stable, working code
├── dev                 # Development integration
│   ├── feature/xyz     # Feature branches
│   ├── bugfix/abc      # Bug fix branches
│   └── experiment/qrs  # Experimental features
└── release/v1.0        # Release branches
```

---

## Tagging Milestones

### Phase Completion Tags

```bash
# Tag design completion
git tag -a v0.1-design -m "Design documentation complete"

# Tag Phase 1 completion
git tag -a v0.2-phase1 -m "Phase 1: Foundation complete"

# Tag Phase 8 completion
git tag -a v1.0 -m "Version 1.0: Full system implementation"

# Push tags
git push origin --tags
```

---

## Commit Message Guidelines

### Format

```
<type>: <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Examples

**Simple**:
```bash
git commit -m "feat: Implement Level 3 PID controller"
```

**Detailed**:
```bash
git commit -m "feat: Implement Level 3 PID controller

- Add attitude PID controller in C++
- Add Pybind11 bindings
- Add Python wrapper class
- Add unit tests

Closes #15"
```

---

## Working with Remote (GitHub/GitLab)

### Add Remote

```bash
# GitHub
git remote add origin git@github.com:username/aircraft-control.git

# GitLab
git remote add origin git@gitlab.com:username/aircraft-control.git

# Verify
git remote -v
```

### Push to Remote

```bash
# First push
git push -u origin main

# Subsequent pushes
git push

# Push all branches
git push --all

# Push tags
git push --tags
```

### Pull from Remote

```bash
# Fetch and merge
git pull

# Fetch only
git fetch

# Rebase instead of merge
git pull --rebase
```

---

## Useful Commands

### Undo Changes

```bash
# Unstage file
git restore --staged controllers/types.py

# Discard changes in working directory
git restore controllers/types.py

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```

### Stashing

```bash
# Save work in progress
git stash

# List stashes
git stash list

# Apply stash
git stash apply

# Apply and remove stash
git stash pop
```

### Search

```bash
# Search in files
git grep "AircraftState"

# Search in commit messages
git log --grep="controller"

# Find who changed line
git blame controllers/types.py
```

---

## .gitignore Patterns

### Already Configured

Current .gitignore covers:
- ✅ Python artifacts
- ✅ C++ build files
- ✅ Logs and data
- ✅ Models and checkpoints
- ✅ IDE configs
- ✅ OS-specific files
- ✅ Secrets

### Adding Custom Patterns

Edit `.gitignore`:
```bash
# Ignore specific file
my_local_config.yaml

# Ignore directory
my_experiments/

# Ignore pattern
*.tmp

# Don't ignore specific file
!important.tmp
```

---

## Pre-commit Hooks (Optional)

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Run tests before commit

pytest tests/
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

---

## Git LFS (Large File Storage)

For large model files:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.onnx"
git lfs track "*.hdf5"
git lfs track "models/*.zip"

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS"
```

---

## Backup Strategy

### Local Backup

```bash
# Create bundle
git bundle create ../aircraft-control-backup.bundle --all

# Restore from bundle
git clone aircraft-control-backup.bundle aircraft-control-restored
```

### Remote Backup

```bash
# Add multiple remotes
git remote add github git@github.com:username/aircraft-control.git
git remote add gitlab git@gitlab.com:username/aircraft-control.git

# Push to all
git push github main
git push gitlab main
```

---

## Common Issues

### Large Files

If you accidentally commit large files:

```bash
# Remove from last commit
git rm --cached models/huge_model.zip
git commit --amend

# Remove from history (use BFG or git-filter-repo)
# WARNING: Rewrites history
git filter-repo --path models/huge_model.zip --invert-paths
```

### Merge Conflicts

```bash
# See conflicted files
git status

# Edit files to resolve conflicts
# Look for <<<<<<< HEAD markers

# Mark as resolved
git add conflicted_file.py

# Complete merge
git commit
```

---

## Project-Specific Workflows

### Implementing New Feature

```bash
# 1. Create feature branch
git checkout -b feature/multi-agent-formation

# 2. Implement feature
# ... make changes ...

# 3. Test
pytest tests/test_multi_agent.py

# 4. Commit
git add multi_agent/
git commit -m "feat: Add formation flying for multi-agent

- Implement FormationFlying coordinator
- Add V-formation, line, diamond patterns
- Add unit tests
- Update documentation"

# 5. Merge to dev
git checkout dev
git merge feature/multi-agent-formation

# 6. Delete feature branch
git branch -d feature/multi-agent-formation
```

### Phase Completion

```bash
# Tag phase completion
git tag -a v0.3-phase3 -m "Phase 3: Classical controllers complete

Implemented:
- Level 1-4 controllers
- C++ PID controllers
- Pybind11 bindings
- All unit tests passing"

# Push tag
git push origin v0.3-phase3
```

---

## Quick Reference

```bash
# Status
git status                 # Check status
git log --oneline         # View history
git diff                  # Show changes

# Staging
git add .                 # Stage all
git add <file>            # Stage file
git reset <file>          # Unstage file

# Committing
git commit -m "message"   # Commit with message
git commit --amend        # Amend last commit

# Branching
git branch                # List branches
git checkout -b <branch>  # Create and switch
git merge <branch>        # Merge branch
git branch -d <branch>    # Delete branch

# Remote
git push                  # Push to remote
git pull                  # Pull from remote
git fetch                 # Fetch without merge

# Undo
git restore <file>        # Discard changes
git reset HEAD~1          # Undo last commit
git stash                 # Save work in progress
```

---

## Next Steps

1. **Create GitHub/GitLab repository** (if using remote)
2. **Add remote**: `git remote add origin <url>`
3. **Push initial commit**: `git push -u origin main`
4. **Create dev branch**: `git checkout -b dev`
5. **Start Phase 1 implementation**

---

**Current Status**: Repository initialized, ready for development
**Total commits**: 2
**Branch**: main
**Files tracked**: 33
