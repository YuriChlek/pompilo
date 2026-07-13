#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import process from 'node:process';
import readline from 'node:readline';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, '../..');

process.chdir(REPO_ROOT);

const OPTIONS = [
  {
    id: 'postgres',
    label: 'PostgreSQL database',
    services: ['postgres'],
    profiles: ['infra'],
    selected: true,
  },
  {
    id: 'redis',
    label: 'Redis cache/queues',
    services: ['redis'],
    profiles: ['infra'],
    selected: true,
  },
  {
    id: 'mailpit',
    label: 'Mailpit email sandbox',
    services: ['mailpit'],
    profiles: ['mail'],
    selected: false,
  },
  {
    id: 'api',
    label: 'NestJS API',
    services: ['api'],
    profiles: ['infra', 'identity'],
    selected: false,
  },
  {
    id: 'client',
    label: 'Next.js client',
    services: ['client'],
    profiles: ['infra', 'identity'],
    selected: false,
  },
  {
    id: 'gateway',
    label: 'Nginx gateway',
    services: ['gateway'],
    profiles: ['infra', 'identity'],
    selected: false,
  },
  {
    id: 'loki',
    label: 'Loki logs database',
    services: ['loki'],
    profiles: ['observability'],
    selected: false,
  },
  {
    id: 'alloy',
    label: 'Grafana Alloy log collector',
    services: ['alloy'],
    profiles: ['observability'],
    selected: false,
  },
  {
    id: 'grafana',
    label: 'Grafana UI',
    services: ['grafana'],
    profiles: ['observability'],
    selected: false,
  },
  {
    id: 'trading-bot',
    label: 'Spot Grid Trading Bot',
    services: ['spot_grid_bot'],
    profiles: ['trading'],
    selected: false,
  },
];

const args = process.argv.slice(2);
const isDryRun = args.includes('--dry-run');
const requestedIds = args.filter(arg => !arg.startsWith('--'));
const COMPOSE_PROJECT = 'pampilo-platform';
const COMPOSE_FILE_ARGS = [
  '-p',
  COMPOSE_PROJECT,
  '--env-file',
  'register-service/.env',
  '-f',
  'infra/compose/docker-compose.yaml',
  '-f',
  'infra/compose/docker-compose.dev.yaml',
];

if (requestedIds.length > 0) {
  const selected = selectByIds(requestedIds);
  await runCompose(selected, isDryRun);
  process.exit(0);
}

if (!process.stdin.isTTY || !process.stdout.isTTY) {
  console.error('Interactive dev container selection requires a TTY.');
  console.error('Use npm run docker:dev for selectable dev services or npm run docker:prod for production stack.');
  process.exit(1);
}

const options = OPTIONS.map(option => ({ ...option }));
let cursor = 0;

readline.emitKeypressEvents(process.stdin);
process.stdin.setRawMode(true);
process.stdin.resume();
render();

process.stdin.on('keypress', async (_str, key) => {
  if (key.name === 'c' && key.ctrl) {
    cleanup();
    process.exit(130);
  }

  if (key.name === 'q' || key.name === 'escape') {
    cleanup();
    process.exit(0);
  }

  if (key.name === 'up') {
    cursor = cursor === 0 ? options.length - 1 : cursor - 1;
    render();
    return;
  }

  if (key.name === 'down') {
    cursor = cursor === options.length - 1 ? 0 : cursor + 1;
    render();
    return;
  }

  if (key.name === 'space') {
    options[cursor].selected = !options[cursor].selected;
    render();
    return;
  }

  if (key.name === 'a') {
    const shouldSelectAll = options.some(option => !option.selected);
    for (const option of options) {
      option.selected = shouldSelectAll;
    }
    render();
    return;
  }

  if (key.name === 'return' || key.name === 'enter') {
    const selected = options.filter(option => option.selected);
    cleanup();
    await runCompose(selected, isDryRun);
    process.exit(0);
  }
});

function selectByIds(ids) {
  const selected = [];

  for (const id of ids) {
    const option = OPTIONS.find(item => item.id === id);
    if (!option) {
      console.error(`Unknown service: ${id}`);
      console.error(`Available services: ${OPTIONS.map(item => item.id).join(', ')}`);
      process.exit(1);
    }
    selected.push(option);
  }

  return selected;
}

function render() {
  readline.cursorTo(process.stdout, 0, 0);
  readline.clearScreenDown(process.stdout);
  process.stdout.write('Select dev containers to start\n\n');
  process.stdout.write('Use Up/Down, Space to toggle, A to toggle all, Enter to start, Q to quit.\n\n');

  options.forEach((option, index) => {
    const pointer = index === cursor ? '>' : ' ';
    const checkbox = option.selected ? '[x]' : '[ ]';
    const services = option.services.length > 1 ? ` -> ${option.services.join(', ')}` : '';
    process.stdout.write(`${pointer} ${checkbox} ${option.label} [${option.id}]${services}\n`);
  });

  const profiles = flattenProfiles(options.filter(option => option.selected));
  const selectedServices = flattenServices(options.filter(option => option.selected));
  process.stdout.write('\n');
  process.stdout.write(`Command: docker compose ${COMPOSE_FILE_ARGS.join(' ')} ${profiles.map(profile => `--profile ${profile}`).join(' ')} up -d --build ${selectedServices.join(' ')}\n`);
}

function cleanup() {
  process.stdin.setRawMode(false);
  process.stdin.pause();
  readline.cursorTo(process.stdout, 0);
  process.stdout.write('\n');
}

function flattenServices(selectedOptions) {
  return [...new Set(selectedOptions.flatMap(option => option.services))];
}

function flattenProfiles(selectedOptions) {
  return [...new Set(selectedOptions.flatMap(option => option.profiles))];
}

async function runCompose(selectedOptions, dryRun) {
  const services = flattenServices(selectedOptions);
  const profiles = flattenProfiles(selectedOptions);

  if (services.length === 0) {
    console.log('No services selected. Nothing to start.');
    return;
  }

  const profileArgs = profiles.flatMap(profile => ['--profile', profile]);
  const commandArgs = ['compose', ...COMPOSE_FILE_ARGS, ...profileArgs, 'up', '-d', '--build', ...services];
  console.log(`Running: docker ${commandArgs.join(' ')}`);

  if (dryRun) {
    return;
  }

  await new Promise((resolve, reject) => {
    const child = spawn('docker', commandArgs, {
      stdio: 'inherit',
      shell: process.platform === 'win32',
    });

    child.on('error', reject);
    child.on('exit', code => {
      if (code === 0) {
        resolve();
        return;
      }

      reject(new Error(`docker exited with code ${code}`));
    });
  });
}
