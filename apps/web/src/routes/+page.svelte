<script lang="ts">
  import Badge from '$lib/components/ui/badge/badge.svelte';
  import Button from '$lib/components/ui/button/button.svelte';
  import Card from '$lib/components/ui/card/card.svelte';
  import Textarea from '$lib/components/ui/textarea/textarea.svelte';
  import type { FinalizeResponse, ProcessTextResponse, SessionStartResponse } from '$lib/types';
  import { copyText, msToClock } from '$lib/utils';

  type UiStatus = 'idle' | 'recording' | 'processing' | 'error';

  let status: UiStatus = 'idle';
  let micStatus: 'ready' | 'denied' | 'busy' = 'ready';
  let asrStatus: 'unknown' | 'ok' | 'down' = 'unknown';
  let llmStatus: 'idle' | 'running' | 'ok' | 'error' = 'idle';

  let rawTamil = '';
  let cleanTanglish = '';
  let rawEnglish = '';
  let noteText = '';

  let sessionId = '';
  let ws: WebSocket | null = null;
  let mediaStream: MediaStream | null = null;
  let audioContext: AudioContext | null = null;
  let sourceNode: MediaStreamAudioSourceNode | null = null;
  let processorNode: ScriptProcessorNode | null = null;
  let muteGainNode: GainNode | null = null;
  let pendingChunkTasks: Promise<unknown>[] = [];
  let chunkUploadChain: Promise<void> = Promise.resolve();
  let pcmWindowChunks: Float32Array[] = [];
  let pcmAllChunks: Float32Array[] = [];
  let totalPcmFrames = 0;
  let micWatchdog: ReturnType<typeof setInterval> | null = null;
  let nextChunkIndex = 0;

  let recordStartMs = 0;
  let elapsedMs = 0;
  let timer: ReturnType<typeof setInterval> | null = null;

  let errorMessage = '';
  let isStopping = false;

  $: statusTone =
    status === 'error' ? 'error' : status === 'recording' ? 'warn' : status === 'processing' ? 'warn' : 'ok';

  async function pingHealth() {
    try {
      const res = await fetch('/api/health');
      asrStatus = res.ok ? 'ok' : 'down';
    } catch {
      asrStatus = 'down';
    }
  }

  pingHealth();

  function resetTimer() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    elapsedMs = 0;
  }

  function stopTimerOnly() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
  }

  function openSocket(url: string) {
    if (ws) {
      ws.close();
    }
    ws = new WebSocket(url);

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload?.partialTamil) {
          rawTamil = payload.partialTamil;
        }
      } catch {
        // Ignore malformed websocket packets.
      }
    };

    ws.onerror = () => {
      asrStatus = 'down';
    };

    ws.onclose = () => {
      ws = null;
    };
  }

  async function startSession() {
    if (sessionId) {
      await fetch(`/api/session/${sessionId}/close`, { method: 'POST' }).catch(() => null);
      sessionId = '';
      if (ws) {
        ws.close();
        ws = null;
      }
    }
    const startRes = await fetch('/api/session/start', { method: 'POST' });
    if (!startRes.ok) {
      throw new Error('Unable to start ASR session.');
    }
    const payload = (await startRes.json()) as SessionStartResponse;
    sessionId = payload.sessionId;
    openSocket(payload.wsUrl);
  }

  async function sendChunk(blob: Blob, index: number, finalChunk = false) {
    if (!sessionId) return;
    const chunkRes = await fetch(`/api/session/${sessionId}/chunk`, {
      method: 'POST',
      headers: {
        'content-type': blob.type || 'audio/wav',
        'x-chunk-index': String(index),
        'x-final-chunk': finalChunk ? '1' : '0'
      },
      body: blob
    });

    const data = await chunkRes.json().catch(() => ({}));
    if (!chunkRes.ok) {
      const detail =
        typeof data?.error === 'string' && data.error.trim()
          ? data.error
          : `Chunk upload failed with status ${chunkRes.status}`;
      if (detail.includes('Session not found')) {
        noteText = [noteText, 'ASR session expired. Click Start again.'].filter(Boolean).join('\n');
        sessionId = '';
        if (ws) {
          ws.close();
          ws = null;
        }
      }
      throw new Error(detail);
    }

    if (data?.partialTamil) {
      rawTamil = data.partialTamil;
    }
    if (data?.warning) {
      noteText = [noteText, String(data.warning)].filter(Boolean).join('\n');
    }
  }

  function mergeFloat32(chunks: Float32Array[]): Float32Array {
    const totalLength = chunks.reduce((acc, arr) => acc + arr.length, 0);
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    return merged;
  }

  function floatTo16BitPCM(view: DataView, offset: number, input: Float32Array) {
    for (let i = 0; i < input.length; i += 1, offset += 2) {
      const s = Math.max(-1, Math.min(1, input[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
  }

  function writeString(view: DataView, offset: number, value: string) {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  }

  function encodeWav(samples: Float32Array, sampleRate: number): Blob {
    const bytesPerSample = 2;
    const numChannels = 1;
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = samples.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);
    floatTo16BitPCM(view, 44, samples);

    return new Blob([buffer], { type: 'audio/wav' });
  }

  function buildWavBlob(chunks: Float32Array[]): Blob | null {
    if (!audioContext || chunks.length === 0) return null;
    const merged = mergeFloat32(chunks);
    if (!merged.length) return null;
    return encodeWav(merged, audioContext.sampleRate);
  }

  function enqueueSnapshotChunk(finalChunk: boolean, mode: 'window' | 'full' = 'window') {
    const source = mode === 'full' ? pcmAllChunks : pcmWindowChunks;
    if (mode === 'window') {
      pcmWindowChunks = [];
    }

    const blob = buildWavBlob(source);
    if (!blob) return;

    const chunkIndex = nextChunkIndex;
    nextChunkIndex += 1;
    const task = (chunkUploadChain = chunkUploadChain
      .then(() => sendChunk(blob, chunkIndex, finalChunk))
      .then(() => {
        asrStatus = 'ok';
      })
      .catch((err) => {
        asrStatus = 'down';
        status = 'error';
        errorMessage = err instanceof Error ? err.message : 'Realtime chunk failed.';
      })) as Promise<void>;
    pendingChunkTasks.push(task);
  }

  async function processText(raw: string) {
    llmStatus = 'running';
    const res = await fetch('/api/text/process', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ rawTamil: raw })
    });

    if (!res.ok) {
      llmStatus = 'error';
      const data = await res.json().catch(() => ({}));
      const detail =
        typeof data?.error === 'string' && data.error.trim()
          ? data.error
          : `Text processing failed (${res.status}).`;
      if (detail.includes('OPENAI_API_KEY')) {
        noteText = [noteText, 'OpenAI key missing. Set OPENAI_API_KEY in apps/web/.env and restart web server.']
          .filter(Boolean)
          .join('\n');
        llmStatus = 'idle';
        return;
      }
      throw new Error(detail);
    }

    const payload = (await res.json()) as ProcessTextResponse;
    cleanTanglish = payload.cleanTanglish;
    rawEnglish = payload.rawEnglish;
    noteText = payload.notes.join('\n');
    llmStatus = 'ok';
  }

  async function finalizeSession() {
    if (!sessionId) return;
    const res = await fetch(`/api/session/${sessionId}/finalize`, { method: 'POST' });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      const detail =
        typeof data?.error === 'string' && data.error.trim()
          ? data.error
          : `Unable to finalize session (${res.status}).`;
      if (detail.includes('Session not found')) {
        noteText = [noteText, 'ASR session expired before finalize. Please record again.']
          .filter(Boolean)
          .join('\n');
        sessionId = '';
        if (ws) {
          ws.close();
          ws = null;
        }
        return;
      }
      throw new Error(detail);
    }
    const payload = (await res.json()) as FinalizeResponse;
    const finalRaw = (payload.rawTamil || '').trim();
    rawTamil = finalRaw;
    if (!finalRaw) {
      noteText = [noteText, 'No speech detected. Click Start and speak for 2-4 seconds.']
        .filter(Boolean)
        .join('\n');
      return;
    }
    await processText(finalRaw);
  }

  async function startRecording() {
    if (status === 'recording') return;
    errorMessage = '';
    noteText = '';
    cleanTanglish = '';
    rawEnglish = '';
    rawTamil = '';
    llmStatus = 'idle';

    try {
      status = 'processing';
      micStatus = 'busy';
      await startSession();

      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContext = new AudioContext();
      await audioContext.resume();
      sourceNode = audioContext.createMediaStreamSource(mediaStream);
      processorNode = audioContext.createScriptProcessor(4096, 1, 1);
      muteGainNode = audioContext.createGain();
      muteGainNode.gain.value = 0;

      sourceNode.connect(processorNode);
      processorNode.connect(muteGainNode);
      muteGainNode.connect(audioContext.destination);

      pendingChunkTasks = [];
      chunkUploadChain = Promise.resolve();
      pcmWindowChunks = [];
      pcmAllChunks = [];
      totalPcmFrames = 0;
      nextChunkIndex = 0;

      processorNode.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        const copy = new Float32Array(input.length);
        copy.set(input);
        pcmWindowChunks.push(copy);
        pcmAllChunks.push(copy);
        totalPcmFrames += copy.length;
      };

      micWatchdog = setInterval(() => {
        if (status === 'recording' && totalPcmFrames === 0) {
          noteText = [noteText, 'No microphone frames captured yet. Check browser mic permission/input device.']
            .filter(Boolean)
            .join('\n');
        }
      }, 2500);

      status = 'recording';
      micStatus = 'ready';
      recordStartMs = Date.now();
      timer = setInterval(() => {
        elapsedMs = Date.now() - recordStartMs;
      }, 200);
    } catch (err) {
      micStatus = 'denied';
      status = 'error';
      errorMessage = err instanceof Error ? err.message : 'Microphone access failed.';
      if (sessionId) {
        await fetch(`/api/session/${sessionId}/close`, { method: 'POST' }).catch(() => null);
        sessionId = '';
      }
      cleanupMedia();
    }
  }

  function cleanupMedia() {
    if (micWatchdog) {
      clearInterval(micWatchdog);
      micWatchdog = null;
    }

    if (processorNode) {
      processorNode.disconnect();
    }
    processorNode = null;

    if (sourceNode) {
      sourceNode.disconnect();
    }
    sourceNode = null;

    if (muteGainNode) {
      muteGainNode.disconnect();
    }
    muteGainNode = null;

    if (audioContext) {
      audioContext.close().catch(() => null);
    }
    audioContext = null;

    if (mediaStream) {
      for (const track of mediaStream.getTracks()) {
        track.stop();
      }
    }
    mediaStream = null;
  }

  async function stopRecording() {
    if (status !== 'recording' || isStopping) return;
    isStopping = true;
    status = 'processing';
    stopTimerOnly();

    try {
      enqueueSnapshotChunk(true, 'full');

      if (pendingChunkTasks.length) {
        await Promise.allSettled(pendingChunkTasks);
        pendingChunkTasks = [];
      }

      cleanupMedia();
      await finalizeSession();
      status = 'idle';
      micStatus = 'ready';
      if (sessionId) {
        sessionId = '';
      }
      if (ws) {
        ws.close();
        ws = null;
      }
    } catch (err) {
      status = 'error';
      errorMessage = err instanceof Error ? err.message : 'Finalize failed.';
    } finally {
      elapsedMs = 0;
      pcmWindowChunks = [];
      pcmAllChunks = [];
      totalPcmFrames = 0;
      isStopping = false;
    }
  }

  async function toggleRecording() {
    if (status === 'processing' || isStopping) return;
    if (status === 'recording') {
      await stopRecording();
      return;
    }
    await startRecording();
  }

  async function clearAll() {
    rawTamil = '';
    cleanTanglish = '';
    rawEnglish = '';
    noteText = '';
    errorMessage = '';
    llmStatus = 'idle';

    if (sessionId) {
      await fetch(`/api/session/${sessionId}/close`, { method: 'POST' }).catch(() => null);
    }
    sessionId = '';
    pcmWindowChunks = [];
    pcmAllChunks = [];
    totalPcmFrames = 0;
    if (ws) {
      ws.close();
      ws = null;
    }
  }

  async function copy(field: 'clean' | 'english') {
    const value = field === 'clean' ? cleanTanglish : rawEnglish;
    const ok = await copyText(value);
    if (!ok) {
      errorMessage = 'Copy failed. Browser clipboard permission is required.';
      status = 'error';
    }
  }

  $: hasRaw = rawTamil.trim().length > 0;
  $: hasEnglishMixed = cleanTanglish.trim().length > 0;
  $: hasRawEnglish = rawEnglish.trim().length > 0;
</script>

<svelte:head>
  <title>VoxFlow</title>
</svelte:head>

<main class="container page">
  <header class="topbar">
    <div>
      <h1>VoxFlow</h1>
      <p>ASR -> English Mixed -> Raw English</p>
    </div>
    <div class="badges">
      <Badge variant={micStatus === 'denied' ? 'error' : micStatus === 'busy' ? 'warn' : 'ok'}>
        Mic: {micStatus}
      </Badge>
      <Badge variant={asrStatus === 'ok' ? 'ok' : asrStatus === 'down' ? 'error' : 'idle'}>
        ASR: {asrStatus}
      </Badge>
      <Badge variant={llmStatus === 'ok' ? 'ok' : llmStatus === 'error' ? 'error' : llmStatus === 'running' ? 'warn' : 'idle'}>
        LLM: {llmStatus}
      </Badge>
      <Badge variant={statusTone}>State: {status}</Badge>
    </div>
  </header>

  <Card className="hero-card">
    <div class="hero-content">
      <button
        class={`mic-button ${status === 'recording' ? 'stop' : 'start'}`}
        on:click={toggleRecording}
        disabled={status === 'processing' || isStopping}
        aria-label={status === 'recording' ? 'Click to stop recording' : 'Click to start recording'}
      >
        {status === 'recording' ? 'Click to Stop' : status === 'processing' ? 'Processing...' : 'Click to Start'}
      </button>

      <p class="hint">Click once to record, click again to transcribe + clean after you stop</p>

      <div class="wave-row" aria-hidden="true">
        {#each Array(8) as _, i}
          <span
            class="wave"
            style={`height: ${status === 'recording' ? 10 + ((i * 17 + elapsedMs / 50) % 26) : 8}px`}
          />
        {/each}
      </div>
      <div class="clock">{msToClock(elapsedMs)}</div>
    </div>
  </Card>

  {#if errorMessage}
    <Card className="error-card">
      <strong>Error</strong>
      <span>{errorMessage}</span>
    </Card>
  {/if}

  <section class="flow-stack">
    {#if hasRaw}
    <Card className="flow-card delay-1">
      <h2>Raw</h2>
      <Textarea value={rawTamil} rows={8} readonly placeholder="ASR output appears after stop..." />
    </Card>
    {/if}

    {#if hasEnglishMixed}
    <Card className="flow-card delay-2">
      <h2>English Mixed</h2>
      <Textarea value={cleanTanglish} rows={8} readonly placeholder="Clean mixed output appears here..." />
      <div class="actions-row">
        <Button variant="success" on:click={() => copy('clean')} disabled={!cleanTanglish}>Copy English Mixed</Button>
      </div>
    </Card>
    {/if}

    {#if hasRawEnglish}
    <Card className="flow-card delay-3">
      <h2>Raw English</h2>
      <Textarea value={rawEnglish} rows={8} readonly placeholder="Direct English translation appears here..." />
      <div class="actions-row">
        <Button variant="success" on:click={() => copy('english')} disabled={!rawEnglish}>Copy Raw English</Button>
      </div>
    </Card>
    {/if}
  </section>

  <div class="actions-row page-actions">
    <Button variant="destructive" on:click={clearAll}>Clear</Button>
    <Button variant="secondary" on:click={pingHealth}>Refresh ASR Health</Button>
  </div>

  {#if noteText}
    <Card>
      <h3>Model notes</h3>
      <pre>{noteText}</pre>
    </Card>
  {/if}
</main>

<style>
  .page {
    padding-top: 2rem;
    padding-bottom: 2rem;
    display: grid;
    gap: 1rem;
  }

  .topbar {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
    flex-wrap: wrap;
  }

  h1 {
    margin: 0;
    font-size: clamp(1.7rem, 2.2vw, 2.35rem);
  }

  p {
    margin: 0.35rem 0 0;
    color: hsl(var(--muted-foreground));
    font-size: 0.93rem;
  }

  .badges {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
  }

  .hero-card {
    background: linear-gradient(140deg, #ffffff, #f5f5f5);
    border-color: #d9d9d9;
  }

  .hero-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.8rem;
    text-align: center;
    padding: 0.7rem 0;
  }

  .mic-button {
    height: 2.8rem;
    min-width: 10.5rem;
    border-radius: 999px;
    border: 1px solid #1f1f1f;
    color: #ffffff;
    font-weight: 800;
    font-size: 0.92rem;
    cursor: pointer;
    box-shadow: 0 10px 22px -16px rgba(0, 0, 0, 0.4);
    transition: transform 0.1s ease, filter 0.15s ease;
    padding: 0 1.1rem;
  }

  .mic-button.start {
    background: #166534;
    border-color: #14532d;
  }

  .mic-button.stop {
    background: #b42318;
    border-color: #912018;
  }

  .mic-button:active {
    transform: scale(0.96);
  }

  .mic-button:hover:not(:disabled) {
    filter: brightness(1.03);
  }

  .hint {
    margin: 0;
    font-size: 0.85rem;
  }

  .wave-row {
    display: flex;
    gap: 0.26rem;
    align-items: flex-end;
  }

  .wave {
    width: 6px;
    border-radius: 999px;
    background: linear-gradient(180deg, #3c3c3c, #000000);
    transition: height 120ms linear;
  }

  .clock {
    font-weight: 700;
    font-size: 0.92rem;
  }

  .error-card {
    border-color: #f0c6ca;
    background: #fff3f4;
    color: #9e1a22;
    display: grid;
    gap: 0.35rem;
  }

  .flow-stack {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .flow-card {
    opacity: 0;
    transform: translateY(12px);
    animation: flowReveal 340ms ease forwards;
  }

  .delay-1 {
    animation-delay: 40ms;
  }

  .delay-2 {
    animation-delay: 120ms;
  }

  .delay-3 {
    animation-delay: 190ms;
  }

  @keyframes flowReveal {
    from {
      opacity: 0;
      transform: translateY(12px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  h2 {
    margin-top: 0;
    margin-bottom: 0.7rem;
    font-size: 1.1rem;
  }

  h3 {
    margin: 0 0 0.6rem;
    color: hsl(var(--muted-foreground));
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  pre {
    margin: 0;
    white-space: pre-wrap;
    font-family: 'DM Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.84rem;
  }

  .actions-row {
    margin-top: 0.7rem;
    display: flex;
    gap: 0.55rem;
    flex-wrap: wrap;
  }

  .page-actions {
    margin-top: 0;
  }

</style>
