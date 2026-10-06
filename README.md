# AuraWear — Fashion + Credit Intelligence

A working local full-stack demo for a clothing brand that also operates an embedded merchant-credit intelligence layer.

## What it solves
1. Dead/slow inventory — SKU demand and return-rate signals.
2. Working-capital gap — AI-assisted merchant credit assessment.
3. High fashion returns — product-level return monitoring and fit/size interventions.

## Included
- Responsive dashboard
- Product intelligence table
- Live backend credit-risk prediction
- Backend AI training endpoint using logistic regression
- AI Lab / retraining screen
- Merchant chatbot
- Complete business-model page
- Demo CSV training dataset

## Run
Requires Node.js 18+.

```bash
npm install
npm start
```

Open http://localhost:3000

## API
- GET /api/summary
- GET /api/products
- POST /api/predict
- POST /api/train
- POST /api/chat

## Important
The CSV included here is DEMO data because the referenced attachment was not available to this build session. Replace `data/credit_data.csv` with your real dataset using the same column names, or adapt the loader in `server.js`.

For a production Indian lending product, use an appropriately regulated bank/NBFC lending partner and implement KYC, consent, privacy/data-governance, fair-lending, underwriting, audit logs and compliant collections. Do not make automated credit decisions solely from the demo model.
