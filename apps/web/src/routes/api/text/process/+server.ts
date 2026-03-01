import { json } from '@sveltejs/kit';
import { generateTanglish } from '$lib/server/openai';

export async function POST({ request, fetch }) {
  let rawTamil = '';
  try {
    const body = (await request.json()) as { rawTamil?: string };
    rawTamil = (body.rawTamil || '').trim();

    if (!rawTamil) {
      return json({ error: 'rawTamil is required' }, { status: 400 });
    }

    const output = await generateTanglish(rawTamil, fetch);
    return json(output);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unable to process text';
    return json({ error: message }, { status: 500 });
  }
}
