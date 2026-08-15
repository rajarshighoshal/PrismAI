"""prism-core (in-progress): PrismAI's channel-neutral, provider-agnostic core.

Currently nested under ``orchestrator/`` so it ships with the existing image without a
build-context change. It will be promoted to a top-level ``prism_core`` package in the
adapter/packaging step, once the Docker build contexts move to the repo root.

Rule for everything in this subpackage: no imports of OpenWebUI, of a specific model
provider, or of orchestrator runtime glue. Model calls are INJECTED by the host.
"""
