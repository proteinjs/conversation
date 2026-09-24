import type {
  ImageTransport,
  ImageTransportBody,
  ImageTransportRequest,
  ImageTransportResponse,
} from './ImageProviderAdapter';

type FetchFunction = (url: string, init: RequestInit) => Promise<Response>;

/**
 * The real wire: the runtime's own `fetch`. The only place in the picture path that opens a
 * connection. Headers are never logged or echoed (they carry the credential); a vendor error is
 * returned as an answer, and the caller's signal is handed to `fetch` untouched.
 */
export class FetchImageTransport implements ImageTransport {
  private readonly fetchFunction: FetchFunction;

  /** `fetchFunction` defaults to the global `fetch`; a test hands in its own. */
  constructor(fetchFunction?: FetchFunction) {
    this.fetchFunction = fetchFunction ?? ((url, init) => fetch(url, init));
  }

  async post(request: ImageTransportRequest): Promise<ImageTransportResponse> {
    const { headers, body } = this.encode(request.body, request.headers);
    const response = await this.fetchFunction(request.url, {
      method: 'POST',
      headers,
      body,
      signal: request.signal,
    });
    const text = await response.text();
    const requestId = response.headers.get(request.requestIdHeader ?? 'x-request-id') ?? undefined;
    return { status: response.status, json: this.parseJson(text), ...(requestId ? { requestId } : {}) };
  }

  /** JSON gets its content type; multipart leaves it to `fetch`, which writes the boundary. */
  private encode(
    body: ImageTransportBody,
    headers: Record<string, string>
  ): { headers: Record<string, string>; body: string | FormData } {
    if (body.kind === 'json') {
      return { headers: { ...headers, 'Content-Type': 'application/json' }, body: JSON.stringify(body.json) };
    }
    const form = new FormData();
    for (const part of body.parts) {
      if ('bytes' in part) {
        form.append(part.name, new Blob([part.bytes], { type: part.mimeType }), part.filename);
      } else {
        form.append(part.name, part.value);
      }
    }
    return { headers: { ...headers }, body: form };
  }

  private parseJson(text: string): unknown {
    try {
      return JSON.parse(text);
    } catch {
      return undefined;
    }
  }
}
