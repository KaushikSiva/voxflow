import { describe, expect, it } from 'vitest';
import { _test } from '../openai';

describe('openai parser', () => {
  it('extracts json even with extra text', () => {
    const payload = 'Here you go\n{"clean_tanglish":"Vanakkam.","raw_english":"Hello.","uncertain_segments":[]}';
    expect(_test.extractJsonObject(payload)).toContain('"clean_tanglish"');
  });

  it('maps valid output fields', () => {
    const parsed = _test.parseOutput(
      '{"clean_tanglish":"naal ki meeting ku pogalama?","raw_english":"Can we go to the meeting tomorrow?","uncertain_segments":["x"]}'
    );
    expect(parsed.cleanTanglish).toBe('Nalaiki meeting ku pogalam?');
    expect(parsed.rawEnglish).toBe('Can we go to the meeting tomorrow?');
    expect(parsed.notes).toEqual(['x']);
  });

  it('reads output_text fallback', () => {
    const text = _test.getOutputText({
      output: [
        {
          content: [
            { type: 'output_text', text: '{"clean_tanglish":"b","raw_english":"c","uncertain_segments":[]}' }
          ]
        }
      ]
    });
    expect(text).toContain('clean_tanglish');
  });
});
