"use client";

import { useState } from "react";
import ChangeList from "@/components/ChangeList";
import StatusPill from "@/components/StatusPill";
import type { PublicCorrectionResponse } from "@/lib/types";

export default function CorrectionResult({
  result,
}: {
  result: PublicCorrectionResponse;
}) {
  const [copied, setCopied] = useState(false);

  async function copyCorrectedText() {
    await navigator.clipboard.writeText(result.corrected_text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  return (
    <section className="reveal rounded-3xl border border-emerald-300/20 bg-white/[0.055] p-5 shadow-[0_0_44px_rgba(52,211,153,0.08)] backdrop-blur-2xl sm:p-7">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs font-bold tracking-[0.2em] text-emerald-200 uppercase">
            Polished result
          </p>
          <h2 className="mt-2 text-xl font-bold text-white">Your improved text</h2>
        </div>
        <StatusPill status="Corrected" />
      </div>

      <div className="mt-6 rounded-2xl border border-white/10 bg-[#071126]/70 p-4 sm:p-5">
        <p className="text-base leading-8 text-slate-100 sm:text-lg">
          {result.corrected_text}
        </p>
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-slate-300">{result.summary}</p>
        <button
          type="button"
          onClick={copyCorrectedText}
          className="rounded-xl border border-cyan-300/25 bg-cyan-300/10 px-4 py-2 text-sm font-bold text-cyan-100 transition-all duration-300 hover:-translate-y-0.5 hover:border-cyan-200/50 hover:bg-cyan-300/15"
        >
          {copied ? "Copied" : "Copy corrected text"}
        </button>
      </div>

      <div className="mt-7 border-t border-white/10 pt-6">
        <h3 className="text-sm font-bold tracking-[0.16em] text-slate-200 uppercase">
          What changed
        </h3>
        <div className="mt-4">
          <ChangeList changes={result.changes} />
        </div>
      </div>
    </section>
  );
}
