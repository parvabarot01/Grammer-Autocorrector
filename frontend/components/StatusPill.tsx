type Status = "Ready" | "Correcting" | "Corrected";

const styles: Record<Status, string> = {
  Ready: "border-cyan-300/30 bg-cyan-300/10 text-cyan-100",
  Correcting: "border-violet-300/30 bg-violet-300/10 text-violet-100",
  Corrected: "border-emerald-300/30 bg-emerald-300/10 text-emerald-100",
};

export default function StatusPill({ status }: { status: Status }) {
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-bold tracking-[0.16em] uppercase ${styles[status]}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current shadow-[0_0_12px_currentColor]" />
      {status}
    </span>
  );
}
