# Eclipse Detection PR Finalization Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate the current eclipse detection branch with tests/black and prepare clean commits for PR submission.

**Architecture:** No code changes planned beyond formatting or fixes required by tests. Focus is on running targeted and full test suites, formatting with Black, and producing clean commits plus PR-ready notes.

**Tech Stack:** Python 3.9+, pytest, black.

### Task 1: Baseline status and scope confirmation

**Files:**
- Verify: `eclipsebin/binning.py`
- Verify: `tests/test_eclipsing_binary_binner.py`
- Verify: `test_edge_detection_integration.py`

**Step 1: Check git status**

Run: `git status -sb`
Expected: working tree shows current changes and branch name

**Step 2: Review diff against main**

Run: `git diff --stat main..HEAD`
Expected: summary of modified files in this branch

**Step 3: Decide commit boundaries**

If changes are mixed, plan separate commits by topic (e.g., edge detection logic, tests, docs). If already clean, proceed to testing.

### Task 2: Run targeted tests for eclipse detection

**Files:**
- Test: `tests/test_eclipsing_binary_binner.py`
- Test: `test_edge_detection_integration.py` (if still present)

**Step 1: Run edge detection-focused tests**

Run: `pytest tests/test_eclipsing_binary_binner.py -v`
Expected: PASS

**Step 2: Run any additional edge-detection test script (if present)**

Run: `python test_edge_detection_integration.py`
Expected: completes without errors

### Task 3: Full test suite and linting

**Files:**
- Verify: repo-wide

**Step 1: Run full tests**

Run: `pytest --cov`
Expected: PASS

**Step 2: Run Black check**

Run: `black --check --verbose .`
Expected: no formatting changes required

**Step 3: If Black fails, format and re-run**

Run: `black .`
Expected: formatted files updated

Run: `black --check --verbose .`
Expected: PASS

### Task 4: Commit and PR-ready notes

**Files:**
- Commit: all changed files in branch

**Step 1: Stage changes**

Run: `git add -A`
Expected: all intended changes staged

**Step 2: Commit**

If single commit:

```bash
git commit -m "feat: improve eclipse boundary detection"
```

If multiple commits, commit by topic:

```bash
git commit -m "feat: add edge-based eclipse boundaries"
```

```bash
git commit -m "test: cover edge detection scenarios"
```

```bash
git commit -m "docs: describe edge detection option"
```

**Step 3: Capture PR notes**

Run: `git log --oneline main..HEAD`
Expected: list of commits for PR summary

Prepare PR text:
- Summary: "Improve eclipse boundary detection with edge-based method and supporting tests/docs"
- Testing: `pytest tests/test_eclipsing_binary_binner.py -v`, `pytest --cov`, `black --check --verbose .`
