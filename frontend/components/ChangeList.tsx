import type { PublicChange } from "@/lib/types";

export default function ChangeList({ changes }: { changes: PublicChange[] }) {
  if (changes.length === 0) {
    return (
      <p className="rounded-2xl border border-emerald-300/20 bg-emerald-300/8 px-4 py-3 text-sm text-emerald-100">
        Your text already reads clearly. No changes were needed.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      {changes.map((change, index) => (
        <div
          key={`${change.before}-${change.after}-${index}`}
          className="rounded-2xl border border-white/10 bg-white/[0.035] p-4 transition-all duration-300 hover:border-cyan-300/25 hover:bg-cyan-300/[0.045]"
        >
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="rounded-lg border border-rose-300/20 bg-rose-300/10 px-2.5 py-1 font-bold text-rose-100 line-through decoration-rose-300/70">
              {change.before || "Remove"}
            </span>
            <span className="text-cyan-200">to</span>
            <span className="rounded-lg border border-emerald-300/25 bg-emerald-300/10 px-2.5 py-1 font-bold text-emerald-100">
              {change.after || "Removed"}
            </span>
          </div>
          <p className="mt-3 text-sm leading-6 text-slate-400">
            {change.explanation}
          </p>
        </div>
      ))}
    </div>
  );
}
