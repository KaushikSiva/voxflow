import { json } from '@sveltejs/kit';

export async function POST({ params, request, fetch }) {
  const sidecarBase = process.env.ASR_SIDECAR_URL || 'http://127.0.0.1:8000';
  const sessionId = params.id;

  const arrayBuffer = await request.arrayBuffer();
  const contentType = request.headers.get('content-type') || 'audio/webm';
  const chunkIndex = Number(request.headers.get('x-chunk-index') || '0');
  const isFinalChunk = request.headers.get('x-final-chunk') === '1';

  const form = new FormData();
  const extension = contentType.includes('wav') ? 'wav' : 'webm';
  const file = new File([arrayBuffer], `chunk-${chunkIndex}.${extension}`, { type: contentType });
  form.set('audio_file', file);
  form.set('session_id', sessionId);
  form.set('chunk_index', String(chunkIndex));
  form.set('is_final_chunk', isFinalChunk ? '1' : '0');

  const res = await fetch(`${sidecarBase}/transcribe/chunk`, {
    method: 'POST',
    body: form
  });

  if (!res.ok) {
    const err = await res.text();
    return json({ error: err }, { status: 502 });
  }

  const payload = (await res.json()) as {
    partial_tamil: string;
    chunk_index: number;
    warning?: string;
  };
  return json({
    partialTamil: payload.partial_tamil,
    chunkIndex: payload.chunk_index,
    isFinalChunk,
    warning: payload.warning ?? null
  });
}
