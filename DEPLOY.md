# Shankh — Deployment Guide

| Service  | Platform             | URL                                      |
|----------|----------------------|------------------------------------------|
| Frontend | Vercel               | https://shankh-finagent.vercel.app       |
| Backend  | Azure Container Apps | set after step 7 below                  |

---

## Prerequisites

```bash
# Azure CLI >= 2.57
az version
az extension add --name containerapp --upgrade

# Logged in
az login
```

`az acr build` streams the build to Azure — no local Docker daemon required.

---

## Backend — Azure Container Apps

### 1. Variables (set once, reuse throughout)

```bash
RESOURCE_GROUP=shankh-rg
LOCATION=eastus
ACR_NAME=shankhregistry        # globally unique, lowercase, no hyphens
ENVIRONMENT=shankh-env
APP_NAME=shankh-backend
IMAGE_TAG=latest
```

### 2. Resource group + Container Registry

```bash
az group create --name $RESOURCE_GROUP --location $LOCATION

az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --admin-enabled true
```

### 3. Build and push the image

Run from the repo root (where `Dockerfile` lives). Model files are baked in.

```bash
az acr build \
  --registry $ACR_NAME \
  --image shankh-backend:$IMAGE_TAG \
  .
```

### 4. Container Apps environment

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

### 6. Deploy

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

echo "https://$BACKEND_URL"
```

Smoke test:

```bash
curl https://$BACKEND_URL/health
# {"status":"healthy","service":"Shankh Financial Advisor"}
```

### 8. Point the frontend at the backend

In the Vercel dashboard → project settings → Environment Variables, set:

```
NEXT_PUBLIC_SHANKH_API_URL = https://<BACKEND_URL from step 7>
```

Then trigger a redeploy (Vercel dashboard → Deployments → Redeploy, or push any commit).

---

## Updating after code changes

```bash
az acr build --registry $ACR_NAME --image shankh-backend:$IMAGE_TAG .

az containerapp update \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --image $ACR_NAME.azurecr.io/shankh-backend:$IMAGE_TAG
```

Azure Container Apps does a rolling restart with zero downtime.

---

## Rotating secrets

```bash
az containerapp secret set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --secrets google-api-key=NEW_VALUE

# Restart to pick up the new value
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
