import { describe, expect, test } from "vitest";

import { getBatchProvider } from "./useRunBatch";

describe("getBatchProvider", () => {
  test("maps local Whisper models to whispercpp", () => {
    expect(getBatchProvider("hyprnote", "QuantizedLargeTurbo")).toBe(
      "whispercpp",
    );
  });

  test("maps local Cactus models to cactus", () => {
    expect(
      getBatchProvider("hyprnote", "cactus-parakeet-tdt-0.6b-v3-int8"),
    ).toBe("cactus");
  });

  test("maps pyannote to the batch transcription provider", () => {
    expect(getBatchProvider("pyannote", "parakeet-tdt-0.6b-v3")).toBe(
      "pyannote",
    );
  });

  test("keeps openai mapped to the batch transcription provider", () => {
    expect(getBatchProvider("openai", "gpt-4o-transcribe")).toBe("openai");
  });
});
