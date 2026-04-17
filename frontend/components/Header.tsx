"use client";

export default function Header() {
  return (
    <header className="px-4 sm:px-6 lg:px-8 py-5 border-b border-border flex items-center gap-3 sm:gap-4">
      <h1 className="text-lg sm:text-xl font-semibold tracking-widest text-text">DECIMUS</h1>
      <span className="text-[10px] sm:text-xs text-dim hidden sm:inline">AI Orchestration Engine &mdash; Piano to Full Score</span>
    </header>
  );
}
