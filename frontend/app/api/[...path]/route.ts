import { type NextRequest } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type ProxyContext = {
  params: {
    path: string[];
  };
};

async function proxyRequest(request: NextRequest, { params }: ProxyContext) {
  const backendUrl = process.env.BACKEND_INTERNAL_URL || "http://localhost:8000";
  const target = new URL(params.path.join("/"), `${backendUrl.replace(/\/+$/, "")}/`);
  target.search = request.nextUrl.search;

  const headers = new Headers(request.headers);
  headers.delete("connection");
  headers.delete("content-length");
  headers.delete("host");

  const method = request.method.toUpperCase();
  const response = await fetch(target, {
    method,
    headers,
    body: method === "GET" || method === "HEAD" ? undefined : await request.arrayBuffer(),
    cache: "no-store",
    signal: request.signal,
  });

  const responseHeaders = new Headers(response.headers);
  responseHeaders.set("Cache-Control", "no-cache, no-transform");
  responseHeaders.set("X-Accel-Buffering", "no");

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: responseHeaders,
  });
}

export { proxyRequest as DELETE, proxyRequest as GET, proxyRequest as PATCH, proxyRequest as POST, proxyRequest as PUT };
