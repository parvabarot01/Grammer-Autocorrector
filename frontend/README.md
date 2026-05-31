# Grammar Autocorrector Frontend

Public-facing Next.js product UI for the Grammar Autocorrector project.

## Local Development

```bash
npm install
npm run dev
```

Open `http://localhost:3000`. The frontend calls the FastAPI public endpoint at
`http://localhost:8000` by default. Override it with:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Production Build

```bash
npm run build
npm start
```

The public UI intentionally consumes only `POST /public/correct`. It does not
display internal pipeline, model, dataset, or evaluation details.
