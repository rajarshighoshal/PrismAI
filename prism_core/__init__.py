"""prism-core: PrismAI's channel-neutral, provider-agnostic core.

No imports of OpenWebUI, of a specific model provider, or of orchestrator runtime
glue. Model calls are INJECTED by the host service (the orchestrator today), so this
package stays publish-ready: standard library only.

Ships inside the orchestrator image via the repo-root Docker build context (see
orchestrator/Dockerfile); the tool-server does not depend on it yet.
"""
