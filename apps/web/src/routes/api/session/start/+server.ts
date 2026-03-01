import { json } from '@sveltejs/kit';

export async function POST({ fetch }) {
  const sidecarBase = process.env.ASR_SIDECAR_URL || 'http://127.0.0.1:8000';

  const res = await fetch(`${sidecarBase}/session/start`, {
    method: 'POST'
  });

  if (!res.ok) {
    return json({ error: 'Failed to start session' }, { status: 502 });
  }

  const payload = (await res.json()) as { session_id: string; ws_url: string };
  return json({ sessionId: payload.session_id, wsUrl: payload.ws_url });
}
