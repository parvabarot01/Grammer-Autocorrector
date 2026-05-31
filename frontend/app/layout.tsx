import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Grammar Autocorrector",
  description: "Clean, intelligent grammar correction for everyday writing.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
