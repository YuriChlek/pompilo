#!/usr/bin/env node
import { setTimeout as sleep } from 'node:timers/promises';

const DEFAULT_SCENARIO = [
  {
    name: 'readiness',
    method: 'GET',
    path: '/health/readiness',
    expected: [200, 503],
  },
  {
    name: 'mail-metrics',
    method: 'GET',
    path: '/metrics',
    expected: [200, 401, 403, 503],
  },
  {
    name: 'login-invalid',
    method: 'POST',
    path: '/auth/login',
    body: { login: 'load-probe@example.invalid', password: 'InvalidPassword1' },
    expected: [401, 429, 503],
  },
  {
    name: 'register-validation-probe',
    method: 'POST',
    path: '/auth/register',
    body: { email: 'not-an-email', name: '', password: 'short' },
    expected: [400, 429, 503],
  },
  {
    name: 'refresh-without-cookie',
    method: 'POST',
    path: '/auth/refresh',
    expected: [400, 401, 403, 500, 503],
  },
  {
    name: 'password-reset-probe',
    method: 'POST',
    path: '/auth/password/forgot',
    body: { email: 'load-probe@example.invalid' },
    expected: [200, 201, 202, 204, 400, 404, 429, 503],
  },
];

const options = parseArgs(process.argv.slice(2));
const baseUrl = options.baseUrl.replace(/\/$/, '');
const concurrency = Number(options.concurrency);
const durationMs = Number(options.durationSeconds) * 1000;
const startedAt = Date.now();
const results = new Map();

if (options.dryRun) {
  console.log(
    JSON.stringify(
      {
        baseUrl,
        concurrency,
        durationSeconds: Number(options.durationSeconds),
        scenario: DEFAULT_SCENARIO.map(({ name, method, path, expected }) => ({
          name,
          method,
          path,
          expected,
        })),
      },
      null,
      2,
    ),
  );
  process.exit(0);
}

await Promise.all(
  Array.from({ length: concurrency }, async (_value, workerIndex) => {
    let iteration = 0;
    while (Date.now() - startedAt < durationMs) {
      const step = DEFAULT_SCENARIO[(workerIndex + iteration) % DEFAULT_SCENARIO.length];
      await runStep(step);
      iteration += 1;
      if (options.delayMs > 0) {
        await sleep(options.delayMs);
      }
    }
  }),
);

const summary = summarize();
console.log(JSON.stringify(summary, null, 2));

if (summary.failedRequests > 0) {
  process.exit(1);
}

async function runStep(step) {
  const requestStartedAt = Date.now();
  let status = 0;
  let ok = false;
  let error = null;

  try {
    const response = await fetch(`${baseUrl}${step.path}`, {
      method: step.method,
      headers: {
        'content-type': 'application/json',
        'x-load-probe': 'phase-23',
      },
      body: step.body ? JSON.stringify(step.body) : undefined,
      redirect: 'manual',
    });
    status = response.status;
    ok = step.expected.includes(status);
    await response.arrayBuffer();
  } catch (caught) {
    error = caught instanceof Error ? caught.message : String(caught);
  }

  const durationMs = Date.now() - requestStartedAt;
  const key = `${step.name}:${status || 'error'}`;
  const record = results.get(key) ?? {
    step: step.name,
    status: status || 'error',
    count: 0,
    failed: 0,
    totalDurationMs: 0,
    maxDurationMs: 0,
    lastError: null,
  };

  record.count += 1;
  record.failed += ok ? 0 : 1;
  record.totalDurationMs += durationMs;
  record.maxDurationMs = Math.max(record.maxDurationMs, durationMs);
  record.lastError = error;
  results.set(key, record);
}

function summarize() {
  const records = [...results.values()].map(record => ({
    ...record,
    avgDurationMs: Math.round(record.totalDurationMs / record.count),
  }));
  const totalRequests = records.reduce((sum, record) => sum + record.count, 0);
  const failedRequests = records.reduce((sum, record) => sum + record.failed, 0);
  const maxDurationMs = records.reduce((max, record) => Math.max(max, record.maxDurationMs), 0);

  return {
    baseUrl,
    concurrency,
    durationSeconds: Number(options.durationSeconds),
    totalRequests,
    failedRequests,
    successRate: totalRequests > 0 ? Number(((totalRequests - failedRequests) / totalRequests).toFixed(4)) : 1,
    maxDurationMs,
    records,
  };
}

function parseArgs(args) {
  const parsed = {
    baseUrl: process.env.API_BASE_URL ?? 'http://localhost:3000',
    concurrency: process.env.LOAD_CONCURRENCY ?? '4',
    durationSeconds: process.env.LOAD_DURATION_SECONDS ?? '30',
    delayMs: Number(process.env.LOAD_DELAY_MS ?? '50'),
    dryRun: false,
  };

  for (const arg of args) {
    if (arg === '--dry-run') {
      parsed.dryRun = true;
      continue;
    }
    const [key, value] = arg.split('=');
    if (key === '--base-url' && value) parsed.baseUrl = value;
    if (key === '--concurrency' && value) parsed.concurrency = value;
    if (key === '--duration-seconds' && value) parsed.durationSeconds = value;
    if (key === '--delay-ms' && value) parsed.delayMs = Number(value);
  }

  if (!Number.isInteger(Number(parsed.concurrency)) || Number(parsed.concurrency) < 1) {
    throw new Error('--concurrency must be a positive integer');
  }
  if (!Number.isInteger(Number(parsed.durationSeconds)) || Number(parsed.durationSeconds) < 1) {
    throw new Error('--duration-seconds must be a positive integer');
  }
  if (!Number.isInteger(parsed.delayMs) || parsed.delayMs < 0) {
    throw new Error('--delay-ms must be a non-negative integer');
  }

  return parsed;
}
