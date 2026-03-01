import { env } from '$env/dynamic/private';

const DEFAULT_MODEL = 'gpt-4.1-mini';

export type TanglishResult = {
  cleanTanglish: string;
  rawEnglish: string;
  notes: string[];
};

type ResponsesPayload = {
  output_text?: string;
  output?: Array<{
    content?: Array<{ type?: string; text?: string }>;
  }>;
};

function extractJsonObject(text: string): string {
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start === -1 || end === -1 || end <= start) {
    throw new Error('No JSON object found in model response.');
  }
  return text.slice(start, end + 1);
}

function applyTanglishStyle(text: string): string {
  return text
    .replace(/\s+/g, ' ')
    .replace(/\s+([,.!?;:])/g, '$1')
    .replace(/\s+'\s+/g, "'")
    .trim();
}

function applyCleanupStyle(text: string): string {
  const normalized = applyTanglishStyle(text)
    .replace(/\bnaal\s*ki\b/gi, 'nalaiki')
    .replace(/\bnaal\s*kku\b/gi, 'nalaikku')
    .replace(/\bnaal\s*ku\b/gi, 'nalaiku')
    .replace(/\bkadaiyil\b/gi, 'kadaiku')
    .replace(/\bkadaiyila\b/gi, 'kadaila')
    .replace(/\bpogalamaa\b/gi, 'pogalam')
    .replace(/\bvaralamaa\b/gi, 'varalam')
    .trim();
  if (!normalized) return normalized;
  return normalized[0].toUpperCase() + normalized.slice(1);
}

function parseOutput(rawText: string): TanglishResult {
  const parsed = JSON.parse(extractJsonObject(rawText)) as {
    clean_tanglish?: string;
    raw_english?: string;
    uncertain_segments?: string[];
  };

  if (!parsed.clean_tanglish || !parsed.raw_english) {
    throw new Error('Model response is missing required fields.');
  }

  return {
    cleanTanglish: applyCleanupStyle(parsed.clean_tanglish),
    rawEnglish: applyTanglishStyle(parsed.raw_english),
    notes: (parsed.uncertain_segments ?? []).map((x) => String(x))
  };
}

function getOutputText(payload: ResponsesPayload): string {
  if (payload.output_text && payload.output_text.trim()) {
    return payload.output_text;
  }

  const fromOutput =
    payload.output
      ?.flatMap((entry) => entry.content ?? [])
      .filter((x) => x.type === 'output_text' || x.type === 'text')
      .map((x) => x.text ?? '')
      .join('\n') ?? '';

  return fromOutput;
}

export async function generateTanglish(rawTamil: string, fetchFn: typeof fetch): Promise<TanglishResult> {
  const apiKey = env.OPENAI_API_KEY;
  const model = env.OPENAI_MODEL || DEFAULT_MODEL;

  if (!apiKey) {
    throw new Error('OPENAI_API_KEY is not configured.');
  }

  const instruction = [
    'You are a Tamil dictation post-processor.',
    'Given raw Tamil ASR text, produce two outputs: cleaned tanglish and raw English translation.',
    'The cleaned tanglish must be natural, readable, and grammatically correct in Roman Tamil.',
    'Do not do rigid letter-by-letter transliteration when it sounds unnatural.',
    'Prefer idiomatic spoken Tanglish phrasing while preserving meaning exactly.',
    'Raw English should be direct and faithful translation, simple and literal.',
    'Keep English technical words unchanged.',
    'Do not add details not present in source.',
    'Cleaned tanglish style rules (English Mixed output):',
    '1) Keep it concise, natural, and uniform spelling.',
    '2) Fix grammar, tense consistency, and awkward morphology.',
    '3) Normalize awkward inflections to common spoken form when meaning is unchanged.',
    '4) Remove accidental repetitions/fillers unless intentionally spoken.',
    '5) Use clean punctuation and sentence casing.',
    '6) Example: "Naalaikku kadaiyil pogalamaa?" -> "Naalaiku kadaiku pogalam".',
    '7) Example: "Innikki meeting la join aagalaama?" -> "Iniki meeting la join aagalam".',
    'Respond as strict JSON with this schema only:',
    '{"clean_tanglish":"...","raw_english":"...","uncertain_segments":["..."]}'
  ].join(' ');

  const response = await fetchFn('https://api.openai.com/v1/responses', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'content-type': 'application/json'
    },
    body: JSON.stringify({
      model,
      temperature: 0.1,
      input: [
        { role: 'system', content: [{ type: 'input_text', text: instruction }] },
        {
          role: 'user',
          content: [{ type: 'input_text', text: `Raw Tamil:\n${rawTamil}` }]
        }
      ]
    })
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`OpenAI request failed (${response.status}): ${errText.slice(0, 250)}`);
  }

  const payload = (await response.json()) as ResponsesPayload;
  const outputText = getOutputText(payload);
  if (!outputText.trim()) {
    throw new Error('Model returned empty output.');
  }

  return parseOutput(outputText);
}

export const _test = {
  parseOutput,
  extractJsonObject,
  getOutputText
};
