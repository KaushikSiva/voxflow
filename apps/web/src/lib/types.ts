export type SessionStartResponse = {
  sessionId: string;
  wsUrl: string;
};

export type ChunkResponse = {
  partialTamil: string;
  chunkIndex: number;
  isFinalChunk: boolean;
};

export type FinalizeResponse = {
  rawTamil: string;
  sessionId: string;
};

export type ProcessTextResponse = {
  cleanTanglish: string;
  rawEnglish: string;
  notes: string[];
};
