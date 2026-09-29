import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Keep Turbopack scoped to this app when it lives inside a larger workspace.
  turbopack: {
    root: process.cwd(),
  },
};

export default nextConfig;
