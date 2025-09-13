import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Ensure stable webpack build on Vercel to avoid Turbopack bundling issues
  experimental: {
    turbo: {
      // Disable Turbopack for production builds
      resolveAlias: {},
    },
  },
};

export default nextConfig;
