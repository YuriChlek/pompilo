# Gym Marketplace Client

Next.js фронтенд для MVP маркетплейсу спортивного коучингу.

## Поточний Обсяг

- Структура App Router з окремими зонами для `public`, `athlete`, `coach` і `admin`
- Cookie-based автентифікація для точок входу `athlete`, `coach` і `admin`
- Проксіювання frontend API через `src/app/api/[...slug]/route.ts`
- Керування доступом до маршрутів і сесіями через `src/proxy.ts` та auth session services
- Робочі простори чату для coach і athlete з realtime-підключенням через Socket.IO
- Флоу онбордингу профілю та керування акаунтом/профілем
- Створення, редагування, поширення програм і перегляд прогресу для coach та athlete
- Пошук тренерів і перегляд профілів тренерів
- Відгуки, feedback, sport data, керування темою та спільні UI-примітиви
- Шар даних на React Query і покриття критичних frontend-модулів тестами Vitest

## Нотатки По Архітектурі

- `src/app` містить route segments для `(public)`, `(auth)`, `athlete`, `coach` і `admin`
- `src/features` містить доменні модулі, зокрема `module-auth`, `module-chat`, `module-profile`, `module-program` і `module-trainer`
- `src/lib/providers/providers.tsx` ініціалізує спільні client providers, включно з React Query
- `next.config.ts` задає security headers і rewrites для `/chat-realtime`

## Команди

```bash
npm install
npm run dev
npm run build
npm run start
npm run lint
npm run test
npm run test:watch
```

## Примітки

- Client очікує `API_BASE_URL`, `NEXT_PUBLIC_API_URL` і `NEXT_PUBLIC_API_PORT` у `client/.env`.
- Вимоги до продукту та задум функціональності описані в `../MVP_SCOPE.md`.
