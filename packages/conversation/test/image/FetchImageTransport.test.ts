import { FetchImageTransport } from '../../src/image/FetchImageTransport';
import { TINY_PNG_BYTES } from './openAiImageFixtures';

/**
 * The real wire, over a `fetch` handed in by the test: what it would have put on the network,
 * and what it hands back. No network is touched.
 */

type SeenCall = { url: string; init: RequestInit };

const fetchAnswering = (answer: { status: number; body: string; headers?: Record<string, string> }) => {
  const calls: SeenCall[] = [];
  const fetchFunction = async (url: string, init: RequestInit): Promise<Response> => {
    calls.push({ url, init });
    return new Response(answer.body, { status: answer.status, headers: answer.headers });
  };
  return { calls, fetchFunction };
};

test('a JSON body is posted as JSON, with the caller’s headers and signal', async () => {
  const { calls, fetchFunction } = fetchAnswering({
    status: 200,
    body: JSON.stringify({ data: [] }),
    headers: { 'x-request-id': 'req_1' },
  });
  const controller = new AbortController();

  const response = await new FetchImageTransport(fetchFunction).post({
    url: 'https://vendor.example/v1/images/generations',
    headers: { Authorization: 'Bearer k' },
    body: { kind: 'json', json: { model: 'm', n: 1 } },
    signal: controller.signal,
  });

  expect(calls).toHaveLength(1);
  expect(calls[0].url).toBe('https://vendor.example/v1/images/generations');
  expect(calls[0].init.method).toBe('POST');
  expect(calls[0].init.headers).toEqual({ Authorization: 'Bearer k', 'Content-Type': 'application/json' });
  expect(calls[0].init.body).toBe('{"model":"m","n":1}');
  expect(calls[0].init.signal).toBe(controller.signal);
  expect(response).toEqual({ status: 200, json: { data: [] }, requestId: 'req_1' });
});

test('a multipart body becomes a form: fields as text, pictures as named files, no content type of our own', async () => {
  const { calls, fetchFunction } = fetchAnswering({ status: 200, body: '{}' });

  await new FetchImageTransport(fetchFunction).post({
    url: 'https://vendor.example/v1/images/edits',
    headers: { Authorization: 'Bearer k' },
    body: {
      kind: 'multipart',
      parts: [
        { name: 'model', value: 'm' },
        { name: 'image[]', bytes: TINY_PNG_BYTES, mimeType: 'image/png', filename: 'one.png' },
        { name: 'image[]', bytes: TINY_PNG_BYTES, mimeType: 'image/jpeg', filename: 'two.jpg' },
      ],
    },
  });

  expect(calls[0].init.headers).toEqual({ Authorization: 'Bearer k' });
  const form = calls[0].init.body as FormData;
  expect(form).toBeInstanceOf(FormData);
  expect(form.get('model')).toBe('m');
  const files = form.getAll('image[]') as File[];
  expect(files.map((file) => [file.name, file.type, file.size])).toEqual([
    ['one.png', 'image/png', TINY_PNG_BYTES.length],
    ['two.jpg', 'image/jpeg', TINY_PNG_BYTES.length],
  ]);
  expect(Buffer.from(await files[0].arrayBuffer()).equals(TINY_PNG_BYTES)).toBe(true);
});

test('a vendor error is an answer, and a body that is not JSON reads as no body', async () => {
  const vendorError = fetchAnswering({ status: 429, body: JSON.stringify({ error: { message: 'slow down' } }) });
  expect(
    await new FetchImageTransport(vendorError.fetchFunction).post({
      url: 'https://vendor.example/x',
      headers: {},
      body: { kind: 'json', json: {} },
    })
  ).toEqual({ status: 429, json: { error: { message: 'slow down' } } });

  const notJson = fetchAnswering({ status: 502, body: '<html>bad gateway</html>' });
  expect(
    await new FetchImageTransport(notJson.fetchFunction).post({
      url: 'https://vendor.example/x',
      headers: {},
      body: { kind: 'json', json: {} },
    })
  ).toEqual({ status: 502, json: undefined });
});

test('a connection that fails rejects', async () => {
  const transport = new FetchImageTransport(async () => {
    throw new TypeError('fetch failed');
  });
  await expect(
    transport.post({ url: 'https://vendor.example/x', headers: {}, body: { kind: 'json', json: {} } })
  ).rejects.toThrow('fetch failed');
});
