// Fugu test proxy — Cloudflare Worker.
//
// TESTING ONLY. Do not use this to avoid provider regional or contractual
// restrictions. Production PrismAI should keep ENABLE_FUGU=false until Sakana
// provides an explicit EU/GDPR-supported endpoint for this account.
//
// Required Worker secrets:
//   SAKANA_API_KEY
//   RELAY_TOKEN       random shared token; PrismAI sends X-Relay-Token
//
// This is not an open proxy: it only forwards /v1/chat/completions and injects
// the Sakana API key server-side.

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method !== "POST" || url.pathname !== "/v1/chat/completions") {
      return new Response("not found", { status: 404 });
    }
    if (!env.RELAY_TOKEN || request.headers.get("x-relay-token") !== env.RELAY_TOKEN) {
      return new Response("unauthorized", { status: 401 });
    }
    if (!env.SAKANA_API_KEY) {
      return new Response("missing upstream key", { status: 500 });
    }

    const upstream = new URL(request.url);
    upstream.protocol = "https:";
    upstream.hostname = "api.sakana.ai";
    upstream.port = "";

    const headers = new Headers(request.headers);
    headers.delete("authorization");
    headers.delete("x-relay-token");
    headers.set("authorization", `Bearer ${env.SAKANA_API_KEY}`);
    headers.set("content-type", "application/json");

    const clean = new Request(upstream, {
      method: request.method,
      headers: headers,
      body: request.body,
    });

    return fetch(clean);
  }
}
