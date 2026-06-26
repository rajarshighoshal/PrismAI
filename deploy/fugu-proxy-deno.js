// Fugu test proxy — Deno Deploy.
//
// TESTING ONLY. Do not use this to avoid provider regional or contractual
// restrictions. Production PrismAI should keep ENABLE_FUGU=false until Sakana
// provides an explicit EU/GDPR-supported endpoint for this account.
//
// Required environment variables:
//   SAKANA_API_KEY
//   RELAY_TOKEN       random shared token; PrismAI sends X-Relay-Token
//
// This is not an open proxy: it only forwards /v1/chat/completions and injects
// the Sakana API key server-side.

Deno.serve(async (req) => {
  const url = new URL(req.url);
  if (req.method !== "POST" || url.pathname !== "/v1/chat/completions") {
    return new Response("not found", { status: 404 });
  }
  if (!Deno.env.get("RELAY_TOKEN") || req.headers.get("x-relay-token") !== Deno.env.get("RELAY_TOKEN")) {
    return new Response("unauthorized", { status: 401 });
  }
  if (!Deno.env.get("SAKANA_API_KEY")) {
    return new Response("missing upstream key", { status: 500 });
  }

  const upstream = new URL(req.url);
  upstream.protocol = "https:";
  upstream.hostname = "api.sakana.ai";
  upstream.port = "";

  const headers = new Headers(req.headers);
  headers.delete("authorization");
  headers.delete("x-relay-token");
  headers.set("authorization", `Bearer ${Deno.env.get("SAKANA_API_KEY")}`);
  headers.set("content-type", "application/json");

  return fetch(new Request(upstream, {
    method: req.method,
    headers,
    body: req.body,
  }));
});
