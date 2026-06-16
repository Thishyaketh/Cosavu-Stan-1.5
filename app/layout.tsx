import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Cosavu Side-by-Side",
  description: "Clean Cosavu comparison demo",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
