"use client";

import { useEffect, useState, type Dispatch, type SetStateAction } from "react";

import { checkAPIHealth } from "@/lib/api";

type HealthStatus = "checking" | "online" | "offline";

const healthCopy: Record<
  HealthStatus,
  { label: string; dot: string; text: string }
> = {
  checking: {
    label: "Checking",
    dot: "bg-amber-300 shadow-[0_0_12px_rgba(252,211,77,0.85)]",
    text: "Contacting endpoint",
  },
  online: {
    label: "API online",
    dot: "bg-emerald-300 shadow-[0_0_12px_rgba(110,231,183,0.85)]",
    text: "Health endpoint reached",
  },
  offline: {
    label: "API offline",
    dot: "bg-rose-300 shadow-[0_0_12px_rgba(253,164,175,0.85)]",
    text: "Endpoint unreachable",
  },
};

async function refreshHealth(
  setStatus: Dispatch<SetStateAction<HealthStatus>>,
) {
  setStatus("checking");
  setStatus((await checkAPIHealth()) ? "online" : "offline");
}

export default function APIHealthCheck() {
  const [status, setStatus] = useState<HealthStatus>("checking");
  const copy = healthCopy[status];

  useEffect(() => {
    void refreshHealth(setStatus);
    const interval = window.setInterval(() => {
      void refreshHealth(setStatus);
    }, 30_000);

    return () => window.clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-3 py-2 shadow-[0_0_24px_rgba(15,23,42,0.22)]">
      <span
        aria-hidden="true"
        className={`h-2.5 w-2.5 rounded-full ${copy.dot} ${
          status === "checking" ? "animate-pulse" : ""
        }`}
      />
      <div className="hidden min-w-28 xl:block">
        <p className="text-[10px] font-bold tracking-[0.18em] text-slate-400 uppercase">
          API health
        </p>
        <p className="text-xs font-semibold text-slate-100">{copy.label}</p>
      </div>
      <span className="text-xs font-semibold text-slate-100 xl:hidden">
        {copy.label}
      </span>
      <button
        type="button"
        onClick={() => void refreshHealth(setStatus)}
        className="rounded-full border border-cyan-300/20 px-2.5 py-1 text-[10px] font-bold tracking-wide text-cyan-100 uppercase transition-colors hover:bg-cyan-300/10"
        aria-label={`Check API health again. Current status: ${copy.text}.`}
      >
        Check
      </button>
    </div>
  );
}
