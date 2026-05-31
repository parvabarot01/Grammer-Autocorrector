"use client";

import { useState } from "react";
import CorrectionResult from "@/components/CorrectionResult";
import StatusPill from "@/components/StatusPill";
import { correctText } from "@/lib/api";
import type { PublicCorrectionResponse } from "@/lib/types";

const MAX_CHARACTERS = 1000;
const SAMPLE_TEXT = "She go to school every day.";

export default function TextCorrectionForm() {
  const [text, setText] = useState(SAMPLE_TEXT);
  const [result, setResult] = useState<PublicCorrectionResponse | null>(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const isEmpty = text.trim().length === 0;
  const isOverLimit = text.length > MAX_CHARACTERS;

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isEmpty || isOverLimit || isLoading) {
      return;
    }

    setIsLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await correctText(text));
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "We could not correct your text. Please try again.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  function clearForm() {
    setText("");
    setResult(null);
    setError("");
  }

  return (
    <div className="space-y-6">
      <form
        onSubmit={handleSubmit}
        className="rounded-3xl border border-cyan-300/20 bg-white/[0.055] p-5 shadow-[0_0_44px_rgba(56,189,248,0.1)] backdrop-blur-2xl sm:p-7"
      >
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-xs font-bold tracking-[0.2em] text-cyan-200 uppercase">
              Writing console
            </p>
            <h1 className="mt-2 text-2xl font-bold text-white sm:text-3xl">
              Refine your sentence
            </h1>
            <p className="mt-2 max-w-xl text-sm leading-6 text-slate-400">
              Paste a sentence or short paragraph. You will get a cleaner version
              and a simple explanation of every change.
            </p>
          </div>
          <StatusPill status={isLoading ? "Correcting" : "Ready"} />
        </div>

        <label
          htmlFor="correction-input"
          className="mt-7 block text-sm font-bold text-slate-200"
        >
          Your text
        </label>
        <textarea
          id="correction-input"
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={9}
          maxLength={MAX_CHARACTERS + 1}
          placeholder="Type or paste your text here..."
          className="mt-3 w-full resize-y rounded-2xl border border-white/10 bg-[#071126]/80 p-4 text-base leading-7 text-slate-100 outline-none transition-all duration-300 placeholder:text-slate-600 focus:border-cyan-300/50 focus:shadow-[0_0_26px_rgba(34,211,238,0.09)]"
        />

        <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
          <p
            className={`text-xs font-bold tracking-wide ${
              isOverLimit ? "text-rose-300" : "text-slate-500"
            }`}
          >
            {text.length} / {MAX_CHARACTERS} characters
          </p>
          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={clearForm}
              className="rounded-xl border border-white/10 px-4 py-2.5 text-sm font-bold text-slate-300 transition-all duration-300 hover:border-white/20 hover:bg-white/5 hover:text-white"
            >
              Clear
            </button>
            <button
              type="submit"
              disabled={isEmpty || isOverLimit || isLoading}
              className="rounded-xl border border-cyan-200/30 bg-gradient-to-r from-cyan-300 via-blue-400 to-violet-400 px-5 py-2.5 text-sm font-bold text-[#071126] shadow-[0_0_24px_rgba(56,189,248,0.2)] transition-all duration-300 hover:-translate-y-0.5 hover:shadow-[0_0_34px_rgba(56,189,248,0.34)] disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:translate-y-0"
            >
              {isLoading ? "Correcting..." : "Correct text"}
            </button>
          </div>
        </div>

        {error ? (
          <p className="mt-5 rounded-2xl border border-rose-300/25 bg-rose-300/10 px-4 py-3 text-sm leading-6 text-rose-100">
            {error}
          </p>
        ) : null}
      </form>

      {result ? <CorrectionResult result={result} /> : null}
    </div>
  );
}
