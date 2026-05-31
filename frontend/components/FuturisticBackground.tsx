export default function FuturisticBackground() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-[#050816]"
    >
      <div className="tech-grid absolute inset-0 opacity-70" />
      <div className="scan-lines absolute inset-0 opacity-50" />
      <div className="aurora absolute -left-36 -top-48 h-[34rem] w-[34rem] rounded-full bg-cyan-400/15 blur-[110px]" />
      <div className="aurora absolute -right-48 top-20 h-[40rem] w-[40rem] rounded-full bg-violet-500/14 blur-[130px] [animation-delay:-6s]" />
      <div className="aurora absolute bottom-[-15rem] left-1/3 h-[30rem] w-[30rem] rounded-full bg-emerald-400/10 blur-[120px] [animation-delay:-11s]" />
      <div className="absolute inset-x-0 top-[18%] h-px bg-gradient-to-r from-transparent via-cyan-300/20 to-transparent" />
    </div>
  );
}
