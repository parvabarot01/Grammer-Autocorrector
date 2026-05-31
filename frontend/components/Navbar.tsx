import Link from "next/link";

import APIHealthCheck from "@/components/APIHealthCheck";

const navigation = [
  { href: "/", label: "Home" },
  { href: "/correct", label: "Correct Text" },
];

export default function Navbar() {
  return (
    <header className="sticky top-0 z-40 border-b border-white/8 bg-[#050816]/70 backdrop-blur-2xl">
      <nav className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-5 py-4 sm:px-8">
        <Link href="/" className="group flex items-center gap-3">
          <span className="relative flex h-9 w-9 items-center justify-center rounded-xl border border-cyan-300/30 bg-cyan-300/10 shadow-[0_0_24px_rgba(34,211,238,0.18)]">
            <span className="h-2.5 w-2.5 rounded-full bg-cyan-200 shadow-[0_0_14px_rgba(103,232,249,0.95)]" />
          </span>
          <span className="text-sm font-bold tracking-[0.18em] text-slate-100 uppercase sm:text-base">
            Grammar Autocorrector
          </span>
        </Link>
        <div className="flex flex-wrap items-center justify-end gap-3">
          <APIHealthCheck />
          <div className="flex items-center gap-1 rounded-full border border-white/10 bg-white/5 p-1">
            {navigation.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="rounded-full px-3 py-2 text-xs font-bold tracking-wide text-slate-300 transition-all duration-300 hover:bg-white/10 hover:text-cyan-100 sm:px-4 sm:text-sm"
              >
                {item.label}
              </Link>
            ))}
          </div>
        </div>
      </nav>
    </header>
  );
}
