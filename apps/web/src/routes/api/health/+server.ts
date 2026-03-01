import { json } from '@sveltejs/kit';

export async function GET({ fetch }) {
  const sidecarBase = process.env.ASR_SIDECAR_URL || 'http://127.0.0.1:8000';
  try {
    const res = await fetch(`${sidecarBase}/healthz`);
    if (!res.ok) {
      return json({ ok: false }, { status: 503 });
    }
    const payload = await res.json();
    return json({ ok: true, sidecar: payload });
  } catch {
    return json({ ok: false }, { status: 503 });
  }
}
