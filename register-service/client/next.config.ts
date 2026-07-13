import type { NextConfig } from "next";

const apiBaseUrl =
  process.env.API_BASE_URL ??
  `${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost"}:${process.env.NEXT_PUBLIC_API_PORT ?? "3000"}`;

const securityHeaders = [
  {
    key: "X-Content-Type-Options",
    value: "nosniff",
  },
  {
    key: "X-Frame-Options",
    value: "DENY",
  },
  {
    key: "Referrer-Policy",
    value: "strict-origin-when-cross-origin",
  },
  {
    key: "Permissions-Policy",
    value: "camera=(), microphone=(), geolocation=(), payment=()",
  },
  {
    key: "Strict-Transport-Security",
    value: "max-age=63072000; includeSubDomains; preload",
  },
];

const nextConfig: NextConfig = {
  cacheComponents: true,
  turbopack: {
    rules: {
      "*.svg": [
        {
          condition: { query: /[?&]url(?=&|$)/ },
          type: "asset",
        },
        {
          condition: { not: { query: /[?&]url(?=&|$)/ } },
          loaders: ["@svgr/webpack"],
          as: "*.js",
        },
      ],
    },
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: securityHeaders,
      },
    ];
  },
  async rewrites() {
    return [
      {
        source: "/chat-realtime",
        destination: `${apiBaseUrl}/chat-realtime`,
      },
      {
        source: "/chat-realtime/:path*",
        destination: `${apiBaseUrl}/chat-realtime/:path*`,
      },
    ];
  },
  output: "standalone",
  reactCompiler: true,
};

export default nextConfig;
