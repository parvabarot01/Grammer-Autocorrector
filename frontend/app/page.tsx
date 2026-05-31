import Link from "next/link";
import FuturisticBackground from "@/components/FuturisticBackground";
import Navbar from "@/components/Navbar";
import StatusPill from "@/components/StatusPill";

const features = [
  {
    index: "01",
    title: "Correct grammar",
    body: "Improve sentence structure and everyday grammar in a single pass.",
  },
  {
    index: "02",
    title: "Highlight changes",
    body: "See exactly what was adjusted without hunting through your paragraph.",
  },
  {
    index: "03",
    title: "Explain mistakes",
    body: "Understand each improvement through simple, useful explanations.",
  },
  {
    index: "04",
    title: "Copy improved text",
    body: "Move the polished result into your next message, document, or draft.",
  },
];

export default function Home() {
  return (
    <main className="min-h-screen overflow-hidden bg-[#050816] text-white">
      <FuturisticBackground />
      <Navbar />

      <section className="mx-auto grid max-w-7xl gap-14 px-5 py-16 sm:px-8 sm:py-24 lg:grid-cols-[1.05fr_0.95fr] lg:items-center lg:gap-12 lg:py-28">
        <div className="reveal">
          <p className="inline-flex items-center gap-2 rounded-full border border-cyan-300/25 bg-cyan-300/8 px-3 py-1.5 text-xs font-bold tracking-[0.2em] text-cyan-100 uppercase">
            <span className="h-1.5 w-1.5 rounded-full bg-cyan-200 shadow-[0_0_12px_rgba(103,232,249,1)]" />
            Precision writing console
          </p>
          <h1 className="mt-7 max-w-3xl text-5xl leading-[0.98] font-black tracking-[-0.065em] sm:text-7xl">
            AI Grammar{" "}
            <span className="bg-gradient-to-r from-cyan-300 via-violet-300 to-emerald-300 bg-clip-text text-transparent">
              Autocorrector
            </span>
          </h1>
          <p className="mt-7 max-w-2xl text-base leading-7 text-slate-300 sm:text-lg sm:leading-8">
            Clean, intelligent grammar correction powered by an AI correction
            pipeline. Write with more confidence, without losing your voice.
          </p>
          <div className="mt-9 flex flex-wrap items-center gap-4">
            <Link
              href="/correct"
              className="rounded-xl border border-cyan-200/30 bg-gradient-to-r from-cyan-300 via-blue-400 to-violet-400 px-6 py-3 text-sm font-black text-[#071126] shadow-[0_0_30px_rgba(56,189,248,0.24)] transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_0_42px_rgba(56,189,248,0.38)]"
            >
              Try correction
            </Link>
            <span className="text-xs font-bold tracking-[0.18em] text-slate-500 uppercase">
              No account required
            </span>
          </div>
        </div>

        <div className="reveal reveal-delay-1 relative">
          <div className="absolute -inset-4 rounded-[2rem] bg-gradient-to-br from-cyan-400/20 via-violet-400/10 to-emerald-400/10 blur-2xl" />
          <div className="relative overflow-hidden rounded-3xl border border-white/12 bg-white/[0.065] p-5 shadow-[0_0_52px_rgba(56,189,248,0.1)] backdrop-blur-2xl sm:p-7">
            <div className="flex items-center justify-between gap-3">
              <p className="text-xs font-bold tracking-[0.2em] text-slate-400 uppercase">
                Live correction preview
              </p>
              <StatusPill status="Corrected" />
            </div>
            <div className="mt-7 space-y-4">
              <div className="rounded-2xl border border-rose-300/15 bg-rose-300/[0.055] p-4">
                <p className="text-xs font-bold tracking-[0.2em] text-rose-200 uppercase">
                  Before
                </p>
                <p className="mt-3 text-lg leading-8 text-slate-300">
                  She <span className="text-rose-200 line-through">go</span> to
                  school every day.
                </p>
              </div>
              <div className="rounded-2xl border border-emerald-300/20 bg-emerald-300/[0.065] p-4">
                <p className="text-xs font-bold tracking-[0.2em] text-emerald-200 uppercase">
                  After
                </p>
                <p className="mt-3 text-lg leading-8 text-slate-100">
                  She{" "}
                  <span className="rounded-md bg-emerald-300/15 px-1.5 py-0.5 text-emerald-100">
                    goes
                  </span>{" "}
                  to school every day.
                </p>
              </div>
            </div>
            <div className="signal-line mt-6 h-px bg-gradient-to-r from-transparent via-cyan-200/80 to-transparent" />
            <p className="mt-5 text-sm leading-6 text-slate-400">
              Use the correct verb form for the subject.
            </p>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-5 pb-20 sm:px-8 sm:pb-28">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((feature, index) => (
            <article
              key={feature.index}
              className={`reveal reveal-delay-${Math.min(index + 1, 3)} rounded-2xl border border-white/10 bg-white/[0.045] p-5 backdrop-blur-xl transition-all duration-300 hover:-translate-y-1 hover:border-cyan-300/25 hover:bg-cyan-300/[0.055]`}
            >
              <p className="text-xs font-bold tracking-[0.2em] text-cyan-200 uppercase">
                {feature.index}
              </p>
              <h2 className="mt-5 text-lg font-bold text-slate-100">
                {feature.title}
              </h2>
              <p className="mt-3 text-sm leading-6 text-slate-400">
                {feature.body}
              </p>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
