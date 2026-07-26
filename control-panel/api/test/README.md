## Test Suite Structure

```
test/
├── unit/               # isolated class/controller/service specs
├── fixtures/           # reusable DTO/entity builders
├── utils/              # lightweight helpers shared across suites
└── jest-unit.json      # Jest config for the current template
```

This repository was cleaned down to a unit-test-only baseline so the identity service can evolve from a minimal, coherent starting point.

### Commands

| Suite | Command |
| --- | --- |
| Unit | `npm run test:unit` |

`npm run test` currently executes the same unit suite. Run `npm run lint` before pushing to ensure TypeScript + ESLint constraints are satisfied.

### Adding New Tests

1. Place specs under `test/unit/<module-name>/`.
2. Reuse fixtures from `test/fixtures` or add new ones if a scenario repeats across specs.
3. Share helpers through `test/utils` only when at least two specs need them.
4. Reintroduce integration or e2e suites only when there is enough MVP behavior to justify them.

This keeps the template fast and easy to evolve while the first product features are still being defined.
