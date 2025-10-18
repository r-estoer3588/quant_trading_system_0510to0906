# precommit_loop_and_commit.ps1
# Usage: pwsh -NoProfile -ExecutionPolicy Bypass -File tools\precommit_loop_and_commit.ps1
# This script runs a converge sequence (ruff -> isort -> black), runs the enhanced tests,
# then runs pre-commit up to $MaxAttempts times. If a hook modifies files, it stages them
# and retries. On success, it makes a commit with the provided message.
param(
    [string]$CommitMessage = "style: apply auto-formatters (black/isort/ruff) to converge repo formatting",
    [int]$MaxAttempts = 6
)

Write-Host "Starting repo converge: ruff format -> ruff --fix -> isort -> black"
python -m ruff format .
python -m ruff check --fix .
python -m isort . --profile black --line-length 88
python -m black . --line-length 88

Write-Host "Running enhanced tests (same as run-enhanced-tests hook)"
python -m pytest tests/test_final_allocation_enhanced.py -q --tb=short

# Stage any changes produced by formatters/tests
git add -A

$attempt = 1
while ($attempt -le $MaxAttempts) {
    Write-Host "pre-commit iteration $attempt/$MaxAttempts"
    pre-commit run --all-files
    $exit = $LASTEXITCODE
    if ($exit -eq 0) {
        Write-Host "All pre-commit hooks passed. Creating commit..."
        git add -A
        git commit -m $CommitMessage
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Commit created successfully."
            exit 0
        }
        else {
            Write-Host "Commit failed (git exit code: $LASTEXITCODE). Aborting."
            exit $LASTEXITCODE
        }
    }
    else {
        Write-Host "Some hooks modified files or failed (exit $exit). Staging changes and retrying..."
        git add -A
        $attempt++
    }
}

Write-Host "Reached max attempts ($MaxAttempts) without success. Leaving repository in current state."
exit 1
