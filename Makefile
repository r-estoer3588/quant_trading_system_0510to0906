PYTHON ?= python
PYTEST ?= pytest -q
IMGDIR ?= results_images
SNAPSHOT_SOURCES ?= results_csv results_csv_test logs $(IMGDIR)

.PHONY: test snapshot imgdiff verify auto-refine run-and-snapshot clean fmt lint fmt-check

test:
	$(PYTHON) -m $(PYTEST)

snapshot:
	@for dir in $(SNAPSHOT_SOURCES); do \
		$(PYTHON) tools/snapshot.py --source $$dir; \
	done

imgdiff:
	$(PYTHON) tools/imgdiff.py --src-dir $(IMGDIR)

verify: test snapshot imgdiff
	@echo "Tests, snapshot, and image diff finished."

auto-refine:
	$(PYTHON) tools/auto_refine_loop.py --src-dir $(IMGDIR) $(foreach dir,$(SNAPSHOT_SOURCES),--snapshot-source $(dir))

run-and-snapshot:
	@echo "Running today signals and capturing screenshot..."
	$(PYTHON) tools/capture_ui_screenshot.py \
		--url http://localhost:8501 \
		--output results_images/today_signals_complete.png \
		--click-button "▶ 本日のシグナル実行" \
		--wait-after-click 30
	@echo "Creating snapshot..."
	$(PYTHON) tools/snapshot.py --source results_csv --source logs --source results_images
	@echo "✅ Complete! Check results_images/today_signals_complete.png"

clean:
	@echo "Removing snapshots directory..."
	@rm -rf snapshots

# --- Formatting & Linting (manual step before commit) ---
fmt:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m black .
	$(PYTHON) -m isort .

fmt-check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .
	$(PYTHON) -m isort --check-only .

lint: fmt-check
	@echo "Lint/format checks passed."
