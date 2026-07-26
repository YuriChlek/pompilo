FROM node:24-alpine AS deps
WORKDIR /app
COPY package*.json .npmrc ./
RUN npm ci

FROM node:24-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build
RUN npm run build:cli

FROM node:24-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production
COPY package*.json .npmrc ./
RUN npm ci --omit=dev && npm cache clean --force
COPY --from=build /app/dist ./dist
COPY --from=build /app/drizzle-migrations ./drizzle-migrations
COPY --from=build /app/tsconfig*.json ./
EXPOSE 3000
CMD ["node", "dist/src/main.js"]
