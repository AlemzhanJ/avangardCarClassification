import type { NextConfig } from "next";

// Migrate from deprecated experimental.turbo to config.turbopack (Next 15+)
const nextConfig: NextConfig = {
  turbopack: {
    // Keep explicit object to mirror previous config; adjust if needed
    resolveAlias: {},
  },
  async headers() {
    return [
      {
        source: "/ort/:path*\.wasm",
        headers: [
          { key: "Content-Type", value: "application/wasm" },
          { key: "Cross-Origin-Resource-Policy", value: "same-origin" },
        ],
      },
    ];
  },
};

export default nextConfig;
