import { json } from '@sveltejs/kit';

export async function POST({ params, fetch }) {
  const sidecarBase = process.env.ASR_SIDECAR_URL || 'http://127.0.0.1:8000';
  const sessionId = params.id;

  const res = await fetch(`${sidecarBase}/session/close`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json'
    },
    body: JSON.stringify({ session_id: sessionId })
  });

  if (!res.ok) {
    return json({ ok: false }, { status: 502 });
  }

  const payload = await res.json();
  return json({ ok: true, ...payload });
}
