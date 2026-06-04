// CF Pages Functions middleware — wraps every request with caches.default
// so HTML responses are cached at the CF Edge without zone-level Cache
// Rules. Fleet pattern; matches the Workers worker.mjs wrapper in spirit
// but uses the Pages Functions API instead.
//
// Without this, Pages returns `cf-cache-status: DYNAMIC` on HTML
// regardless of the Cache-Control headers in _headers — the edge
// refuses to cache HTML by default.

interface Env {
  ASSETS?: { fetch: (req: Request) => Promise<Response> };
}

export const onRequest: PagesFunction<Env> = async (context) => {
  const { request } = context;

  if (request.method !== "GET") {
    return context.next();
  }

  const url = new URL(request.url);
  // Only cache HTML routes. Skip assets, API, etc.
  if (
    url.pathname.startsWith("/_astro/") ||
    url.pathname.startsWith("/_next/") ||
    url.pathname.startsWith("/api/") ||
    url.pathname.includes(".")
  ) {
    // Asset paths — let Pages handle directly (already cached).
    return context.next();
  }

  const cache = caches.default;
  const cached = await cache.match(request);
  if (cached) {
    const hit = new Response(cached.body, cached);
    hit.headers.set("x-edge-cache", "HIT");
    return hit;
  }

  const response = await context.next();
  const contentType = response.headers.get("content-type") ?? "";
  if (response.status !== 200 || !contentType.includes("text/html")) {
    return response;
  }

  const body = await response.arrayBuffer();
  const headers = new Headers(response.headers);
  headers.set(
    "Cache-Control",
    "public, max-age=3600, s-maxage=86400, stale-while-revalidate=604800",
  );

  const cacheable = new Response(body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
  context.waitUntil(cache.put(request, cacheable.clone()));

  const clientResponse = new Response(body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
  clientResponse.headers.set("x-edge-cache", "MISS");
  return clientResponse;
};
