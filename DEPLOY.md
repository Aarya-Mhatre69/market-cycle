# Shankh — Deployment Guide

Two independent deployments:
- **Backend** → Azure Container Apps (Python / FastAPI)
- **Frontend** → Vercel (Next.js)

---

## Prerequisites

```bash
# Azure CLI
az version                        # needs >= 2.57
az extension add --name containerapp --upgrade

# Docker Desktop running locally
docker version

# Logged in
az login
```

---

## Part 1 — Backend on Azure Container Apps

### 1. Set your variables

```bash
# Edit these once — everything below reuses them
RESOURCE_GROUP=shankh-rg
LOCATION=eastus
ACR_NAME=shankhregistry          # must be globally unique, lowercase, no hyphens
ENVIRONMENT=shankh-env
APP_NAME=shankh-backend
IMAGE_TAG=latest
```

### 2. Create resource group + Azure Container Registry

```bash
az group create --name $RESOURCE_GROUP --location $LOCATION

az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --admin-enabled true
```

### 3. Build and push the image

The Dockerfile is at the repo root. Model files are baked in — no volume needed.

```bash
# From the repo root (where Dockerfile lives)
az acr build \
  --registry $ACR_NAME \
  --image shankh-backend:$IMAGE_TAG \
  .
```

This streams the build to ACR directly — no local Docker daemon required.
If you prefer building locally first:

```bash
docker build -t $ACR_NAME.azurecr.io/shankh-backend:$IMAGE_TAG .
az acr login --name $ACR_NAME
docker push $ACR_NAME.azurecr.io/shankh-backend:$IMAGE_TAG
```

### 4. Create the Container Apps environment

```bash
az containerapp env create \
  --name $ENVIRONMENT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION
```

### 5. Get ACR credentials

```bash
ACR_PASSWORD=$(az acr credential show \
  --name $ACR_NAME \
  --query "passwords[0].value" \
  --output tsv)
```

### 6. Deploy the container app

Replace each `YOUR_*` value with your real API keys.

```bash
az containerapp create \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT \
  --image $ACR_NAME.azurecr.io/shankh-backend:$IMAGE_TAG \
  --registry-server $ACR_NAME.azurecr.io \
  --registry-username $ACR_NAME \
  --registry-password $ACR_PASSWORD \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3 \
  --cpu 1.0 \
  --memory 2.0Gi \
  --env-vars \
    GOOGLE_API_KEY=secretref:google-api-key \
    TAVILY_API_KEY=secretref:tavily-api-key \
    CEREBRAS_API_KEY=secretref:cerebras-api-key \
    MISTRAL_API_KEY=secretref:mistral-api-key \
  --secrets \
    google-api-key=YOUR_GOOGLE_API_KEY \
    tavily-api-key=YOUR_TAVILY_API_KEY \
    cerebras-api-key=YOUR_CEREBRAS_API_KEY \
    mistral-api-key=YOUR_MISTRAL_API_KEY
```

### 7. Get the backend URL

```bash
BACKEND_URL=$(az containerapp show \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "properties.configuration.ingress.fqdn" \
  --output tsv)

echo "Backend: https://$BACKEND_URL"
```

Test it:

```bash
curl https://$BACKEND_URL/health
# {"status":"healthy","service":"Shankh Financial Advisor"}
```

### 8. Update CORS in main.py

Once you have your Vercel URL (e.g. `https://shankh.vercel.app`), add it to the `origins` list in `main.py`:

```python
origins = [
    "https://shankh.vercel.app",   # <-- your Vercel URL
    "http://localhost:3000",
]
```

Then redeploy:

```bash
az acr build --registry $ACR_NAME --image shankh-backend:$IMAGE_TAG .

az containerapp update \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --image $ACR_NAME.azurecr.io/shankh-backend:$IMAGE_TAG
```

---

## Part 2 — Frontend on Vercel

The repo has a `vercel.json` at the root that points Vercel at the `frontend/` directory. Vercel will find `package.json` and detect Next.js automatically.

### Option A — Vercel dashboard (recommended)

1. Push the repo to GitHub
2. Go to [vercel.com/new](https://vercel.com/new) → Import your repo
3. Vercel will auto-detect the `vercel.json` root directory setting
4. Add one environment variable:
   - `NEXT_PUBLIC_SHANKH_API_URL` = `https://<your-backend-fqdn-from-step-7>`
5. Deploy

### Option B — Vercel CLI

```bash
npm i -g vercel
vercel login

# From the repo root
vercel --prod \
  --env NEXT_PUBLIC_SHANKH_API_URL=https://$BACKEND_URL
```

---

## Updating the backend after code changes

```bash
az acr build --registry $ACR_NAME --image shankh-backend:$IMAGE_TAG .

az containerapp update \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --image $ACR_NAME.azurecr.io/shankh-backend:$IMAGE_TAG
```

Azure Container Apps does a rolling restart with zero downtime.

---

## Updating secrets

```bash
az containerapp secret set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --secrets google-api-key=NEW_VALUE

# Restart to pick up the new secret value
az containerapp revision restart \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --revision $(az containerapp revision list \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query "[0].name" --output tsv)
```

---

## Tear down

```bash
az group delete --name $RESOURCE_GROUP --yes --no-wait
```

This removes the Container App, environment, ACR, and all associated resources.
