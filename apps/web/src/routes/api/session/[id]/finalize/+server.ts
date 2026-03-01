import { json } from '@sveltejs/kit';

export async function POST({ params, fetch }) {
  const sidecarBase = process.env.ASR_SIDECAR_URL || 'http://127.0.0.1:8000';
  const sessionId = params.id;

  const res = await fetch(`${sidecarBase}/transcribe/finalize`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json'
    },
    body: JSON.stringify({ session_id: sessionId })
  });

  if (!res.ok) {
    const err = await res.text();
    return json({ error: err }, { status: 502 });
  }

  const payload = (await res.json()) as { raw_tamil: string };

  await fetch(`${sidecarBase}/session/close`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json'
    },
    body: JSON.stringify({ session_id: sessionId })
  }).catch(() => null);

  return json({ rawTamil: payload.raw_tamil, sessionId });
}
