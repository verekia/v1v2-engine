FROM node:22.11.0-alpine3.20

WORKDIR /app

COPY package.json package-lock.json .
COPY examples/react/package.json examples/react/package.json
COPY examples/vue/package.json examples/vue/package.json
COPY examples/svelte/package.json examples/svelte/package.json
COPY examples/vanilla/package.json examples/vanilla/package.json
COPY website/package.json website/package.json
COPY packages/core/package.json packages/core/package.json
COPY packages/react/package.json packages/react/package.json
COPY packages/svelte/package.json packages/svelte/package.json
COPY packages/vue/package.json packages/vue/package.json
COPY packages/vanilla/package.json packages/vanilla/package.json

RUN npm i

COPY . .

RUN npm run build && npm run build-examples && npm run build-website

CMD ["npm", "run", "website"]
