.DEFAULT_GOAL := help

SHELL := /usr/bin/env bash

CONTAINER_COMMAND ?= podman
PYTHON_VERSION ?= 3.13
VENV_DIR ?= .venv

GITLEAKS         ?= gitleaks
LEFTHOOK_VERSION ?= 1.7.10
LEFTHOOK_DIR     ?= $(CURDIR)/.bin
LEFTHOOK_BIN     ?= $(LEFTHOOK_DIR)/lefthook

PREFIX ?= ffreis
IMAGE_PROVIDER ?=
IMAGE_TAG ?= api-grpc-smoke
SMOKE_TIMEOUT ?= 20m
BASE_DIR ?= .
CONTAINER_DIR ?= container

IMAGE_PREFIX := $(if $(IMAGE_PROVIDER),$(IMAGE_PROVIDER)/,)$(PREFIX)
IMAGE_ROOT := $(IMAGE_PREFIX)
BASE_IMAGE ?= $(IMAGE_PREFIX)/base
BASE_RUNNER_IMAGE ?= $(IMAGE_PREFIX)/base-runner
UV_VENV_IMAGE ?= $(IMAGE_PREFIX)/onnx-converter-uv-venv
PACKAGE_IMAGE ?= $(IMAGE_PREFIX)/onnx-converter-package
CLI_IMAGE ?= $(IMAGE_PREFIX)/onnx-converter-cli
EXTRAS ?= all
CONTAINER_BUILD_FLAGS ?=

BASE_IMAGE_VALUE := $(shell grep '^BASE_IMAGE=' $(CONTAINER_DIR)/digests.env | cut -d= -f2)
BASE_DIGEST_VALUE := $(shell grep '^BASE_DIGEST=' $(CONTAINER_DIR)/digests.env | cut -d= -f2)

.PHONY: help
help: ## Show help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: all
all: check ## Run lint and tests

# ------------------------------------------------------------------------------
# Local development
# ------------------------------------------------------------------------------

.PHONY: env
env: ## Create virtual environment
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment already exists at $(VENV_DIR)"; \
	else \
		python$(PYTHON_VERSION) -m venv $(VENV_DIR); \
	fi
	@echo "Activate with: . $(VENV_DIR)/bin/activate"

.PHONY: install
install: ## Install runtime dependencies
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -e "./[${EXTRAS}]"

.PHONY: install-dev
install-dev: ## Install dev tooling
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -e "./[all,dev]"

.PHONY: fmt
fmt: ## Format code in place (alias for format)
	$(VENV_DIR)/bin/ruff format src tests
	$(VENV_DIR)/bin/black src tests
	$(VENV_DIR)/bin/isort src tests

.PHONY: format
format: ## Format code (ruff + black + isort)
	$(VENV_DIR)/bin/ruff format src tests
	$(VENV_DIR)/bin/black src tests
	$(VENV_DIR)/bin/isort src tests

.PHONY: lint
lint: ## Lint code (ruff + flake8 + mypy)
	$(VENV_DIR)/bin/ruff check src tests
	$(VENV_DIR)/bin/flake8 src tests
	$(VENV_DIR)/bin/mypy src

.PHONY: validate
validate: ## Static type checking (mypy)
	$(VENV_DIR)/bin/mypy src

.PHONY: plan
plan: ## Not applicable — use 'make validate' or 'make test' for Python repos
	@echo "INFO: 'plan' is Terraform-specific and does not apply to Python repos."
	@echo "      To type-check: make validate"
	@echo "      To run tests: make test"

.PHONY: test
test: ## Run tests
	$(VENV_DIR)/bin/pytest -q

.PHONY: test-unit
test-unit: ## Run unit tests only
	$(VENV_DIR)/bin/pytest -q tests/unit_tests

.PHONY: test-integration
test-integration: ## Run integration tests only
	$(VENV_DIR)/bin/pytest -q tests/integration_tests

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests only
	$(VENV_DIR)/bin/pytest -q tests/e2e_tests

.PHONY: grpc-generate
grpc-generate: ## Regenerate gRPC protobuf stubs
	./scripts/generate_grpc_stubs.sh

.PHONY: grpc-check
grpc-check: ## Verify gRPC protobuf stubs are in sync
	./scripts/check_grpc_stubs.sh

.PHONY: grpc-clean
grpc-clean: ## Remove generated gRPC protobuf stubs
	rm -f src/converter_grpc/converter_pb2.py src/converter_grpc/converter_pb2_grpc.py

.PHONY: smoke-api-grpc
smoke-api-grpc: ## Run docker-compose HTTP + gRPC smoke test
	@set -euo pipefail; \
	cleanup() { \
		IMAGE_ROOT="$(IMAGE_ROOT)" IMAGE_TAG="$(IMAGE_TAG)" docker compose -f examples/docker-compose.api-grpc.yml down --remove-orphans || true; \
	}; \
	trap cleanup EXIT; \
	IMAGE_ROOT="$(IMAGE_ROOT)" IMAGE_TAG="$(IMAGE_TAG)" timeout --foreground "$(SMOKE_TIMEOUT)" docker compose -f examples/docker-compose.api-grpc.yml up --build --abort-on-container-exit --exit-code-from smoke

.PHONY: test-grpc-parity
test-grpc-parity: ## Run gRPC/API parity tests
	$(VENV_DIR)/bin/pytest -q tests/integration_tests/test_grpc_parity.py

.PHONY: test-grpc-parity-property
test-grpc-parity-property: ## Run gRPC/API parity property tests (Hypothesis)
	@set +e; \
	$(VENV_DIR)/bin/pytest -q tests/integration_tests/test_grpc_parity.py -m property; \
	rc=$$?; \
	if [ $$rc -eq 5 ]; then \
		echo "No property tests collected; treating as success."; \
		exit 0; \
	fi; \
	exit $$rc

.PHONY: openapi-check
openapi-check: ## Validate OpenAPI contract and verify runtime drift
	env -u VIRTUAL_ENV uv run --project . --extra server --with openapi-spec-validator --with pyyaml python scripts/check_openapi.py

.PHONY: check
check: grpc-check lint test-unit ## Run lint and fast tests

.PHONY: coverage
coverage: ## Generate coverage report
	mkdir -p coverage
	$(VENV_DIR)/bin/pytest tests/unit_tests --cov=onnx_converter --cov-report=xml:coverage.xml --cov-report=html:coverage/html --cov-report=term

.PHONY: architecture-check
architecture-check: ## Run architecture and complexity checks
	$(VENV_DIR)/bin/python scripts/check_architecture.py
	$(VENV_DIR)/bin/python scripts/check_orchestrator_complexity.py
	$(VENV_DIR)/bin/mypy src/onnx_converter/application

.PHONY: ci-local
ci-local: architecture-check lint test-unit coverage ## Approximate default CI checks locally

.PHONY: clean
clean: ## Remove caches and venv
	rm -rf $(VENV_DIR) .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov coverage.xml coverage
	find . -type d -name '__pycache__' -exec rm -r {} +
	find . -type f -name '*.py[cod]' -delete

# ------------------------------------------------------------------------------
# Container builds
# ------------------------------------------------------------------------------

.PHONY: build-base
build-base: ## Build base image (pinned by digest env)
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base \
		-t $(BASE_IMAGE) \
		$(BASE_DIR) \
		--build-arg BASE_IMAGE="$(BASE_IMAGE_VALUE)" \
		--build-arg BASE_DIGEST="$(BASE_DIGEST_VALUE)"

.PHONY: build-base-runner
build-base-runner: build-base ## Build base-runner image
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base-runner -t $(BASE_RUNNER_IMAGE) $(BASE_DIR) \
		--build-arg BASE_IMAGE="$(BASE_IMAGE)"

.PHONY: build-uv-venv
build-uv-venv: build-base ## Build uv venv image
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.uv-builder -t $(UV_VENV_IMAGE) $(BASE_DIR) \
		--build-arg BASE_IMAGE="$(BASE_IMAGE)" \
		--build-arg PYTHON_VERSION="$(PYTHON_VERSION)"

.PHONY: build-package
build-package: build-uv-venv ## Build package image (installs converter)
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.package -t $(PACKAGE_IMAGE) $(BASE_DIR) \
		--build-arg UV_VENV_IMAGE="$(UV_VENV_IMAGE)" \
		--build-arg EXTRAS="$(EXTRAS)"

.PHONY: build-cli
build-cli: build-base-runner build-package ## Build CLI image
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.cli -t $(CLI_IMAGE) $(BASE_DIR) \
		--build-arg PACKAGE_IMAGE="$(PACKAGE_IMAGE)" \
		--build-arg BASE_RUNNER_IMAGE="$(BASE_RUNNER_IMAGE)"

.PHONY: build build-converter-images
build: build-converter-images ## Build all converter images (alias used by CI)
build-converter-images: build-uv-venv build-package build-cli ## Build all converter images
.PHONY: run-cli
run-cli: ## Run converter CLI image (use RUN_ARGS=...)
	$(CONTAINER_COMMAND) run --rm $(CLI_IMAGE) $(RUN_ARGS)

.PHONY: clean-images
clean-images: ## Remove converter images
	$(CONTAINER_COMMAND) rmi $(UV_VENV_IMAGE) $(PACKAGE_IMAGE) $(CLI_IMAGE) || true

.PHONY: secrets-scan-staged lefthook-bootstrap lefthook-install lefthook-run lefthook

secrets-scan-staged: ## Scan staged diff for secrets
	@command -v $(GITLEAKS) >/dev/null 2>&1 || (echo "Missing tool: $(GITLEAKS). Install: https://github.com/gitleaks/gitleaks#installing" && exit 1)
	$(GITLEAKS) protect --staged --redact

lefthook-bootstrap: ## Download lefthook binary into ./.bin
	LEFTHOOK_VERSION="$(LEFTHOOK_VERSION)" BIN_DIR="$(LEFTHOOK_DIR)" bash ./scripts/bootstrap_lefthook.sh

lefthook-install: lefthook-bootstrap ## Install git hooks (runs bootstrap first)
	@if [ -x "$(LEFTHOOK_BIN)" ] && [ -x ".git/hooks/pre-commit" ] && [ -x ".git/hooks/pre-push" ] && [ -x ".git/hooks/commit-msg" ]; then \
		echo "lefthook hooks already installed"; \
		exit 0; \
	fi
	LEFTHOOK="$(LEFTHOOK_BIN)" "$(LEFTHOOK_BIN)" install

lefthook-run: lefthook-bootstrap ## Run all hooks locally (pre-commit + commit-msg + pre-push)
	LEFTHOOK="$(LEFTHOOK_BIN)" "$(LEFTHOOK_BIN)" run pre-commit
	@tmp_msg="$$(mktemp)"; \
	echo "chore(hooks): validate commit-msg hook" > "$$tmp_msg"; \
	LEFTHOOK="$(LEFTHOOK_BIN)" "$(LEFTHOOK_BIN)" run commit-msg -- "$$tmp_msg"; \
	rm -f "$$tmp_msg"
	LEFTHOOK="$(LEFTHOOK_BIN)" "$(LEFTHOOK_BIN)" run pre-push

lefthook: lefthook-bootstrap lefthook-install lefthook-run ## Install hooks and run them

.PHONY: ci-grpc
ci-grpc: grpc-check openapi-check lint test-grpc-parity ## Run gRPC sync + parity quality gate

# ── Standard quality-system targets (uv-based) ────────────────────────────────
UV      ?= uv
SRC_DIR ?= src
TEST_DIR ?= tests/unit_tests

.PHONY: typecheck
typecheck: ## Type-check with mypy (uv run)
	$(UV) run mypy $(SRC_DIR)

.PHONY: test-all
test-all: ## Run full test suite
	$(UV) run pytest tests/

.PHONY: test-property
test-property: ## Run Hypothesis property-based tests
	$(UV) run pytest -q tests/hypothesis_tests/ 2>/dev/null || \
	  $(UV) run pytest -q -k "hypothesis or property" tests/ 2>/dev/null || true

.PHONY: mutation-test
mutation-test: ## Run mutation testing with mutmut (slow — run in CI)
	$(UV) run mutmut run --paths-to-mutate=$(SRC_DIR) --tests-dir=$(TEST_DIR) || true
	$(UV) run mutmut results

PLATFORM_STANDARDS_SHA ?= 3c787edb4e96ddea2e86b2add2c32139685e8db7  # v1.2.1
PLATFORM_STANDARDS_RAW ?= https://raw.githubusercontent.com/FelipeFuhr/ffreis-platform-standards

install-act: ## Download pinned act binary into .bin/
	@mkdir -p scripts
	@curl -fsSL "$(PLATFORM_STANDARDS_RAW)/$(PLATFORM_STANDARDS_SHA)/scripts/install_act.sh" \
		-o scripts/install_act.sh && chmod +x scripts/install_act.sh
	@bash ./scripts/install_act.sh

ci-local-act: ## Run workflows locally via act (GH Actions quota fallback). Args via ARGS=...
	@mkdir -p scripts
	@curl -fsSL "$(PLATFORM_STANDARDS_RAW)/$(PLATFORM_STANDARDS_SHA)/scripts/run-ci-local.sh" \
		-o scripts/run-ci-local.sh && chmod +x scripts/run-ci-local.sh
	@PATH="$(CURDIR)/.bin:$(PATH)" bash ./scripts/run-ci-local.sh $(ARGS)
