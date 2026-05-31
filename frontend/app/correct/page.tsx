import FuturisticBackground from "@/components/FuturisticBackground";
import Navbar from "@/components/Navbar";
import TextCorrectionForm from "@/components/TextCorrectionForm";

export default function CorrectPage() {
  return (
    <main className="min-h-screen overflow-hidden bg-[#050816] text-white">
      <FuturisticBackground />
      <Navbar />
      <div className="mx-auto max-w-5xl px-5 py-12 sm:px-8 sm:py-16">
        <div className="reveal mb-8">
          <p className="text-xs font-bold tracking-[0.25em] text-violet-200 uppercase">
            AI writing assistant
          </p>
          <h2 className="mt-3 max-w-3xl text-4xl font-black tracking-[-0.04em] text-white sm:text-5xl">
            Turn rough sentences into{" "}
            <span className="bg-gradient-to-r from-cyan-300 via-violet-300 to-emerald-300 bg-clip-text text-transparent">
              clear writing.
            </span>
          </h2>
        </div>
        <TextCorrectionForm />
      </div>
    </main>
  );
}
