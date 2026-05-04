# Deploying the MVP demo lane on AWS App Runner

**Status:** Sprint 07 (S07) deliverable. AWS account `429100832336` has
existing grandfathered App Runner services — App Runner has been
closed to new customers as of 2026-04-30, so we **reuse** an existing
service rather than create a new one.

**Hard cost cap:** $5/mo. See `memory/serving_cost_constraints.md`.
First week of real demo traffic must be checked against actual billing.

---

## Architecture

```
Browser ──HTTPS──▶ App Runner service (FastAPI + SPA, single container)
                       │
                       ├─ Static SPA  ◀─ frontend/dist baked in via Dockerfile stage 1
                       │
                       └─ /v1/* ──▶ Ollama Cloud (proxied generations)
                                    HF Inference / OpenRouter (alternates)
```

Single container, single origin. The decision to fold the SPA into the
FastAPI process (rather than separate S3 + CloudFront) was driven by
the user's "less to manage" preference and the fact that the existing
Dockerfile already builds the frontend and copies `frontend/dist` into
the runtime image — uvicorn just needed to serve it (S07 closes that
gap with the StaticFiles mount in `app/main.py`).

**App Runner has no GPU and no room for an in-process model.**
Generation MUST proxy out to Ollama Cloud / HF Inference / OpenRouter.
Never set `VALONY_PREWARM_INFERENCE=1` on App Runner — it'd try to
download a multi-GB base model into a 0.5 GB container.

---

## One-time setup

Skip steps you've already done.

### 1. ECR repository

```bash
aws ecr create-repository \
    --repository-name valonylabs-studio \
    --region us-east-1
```

### 2. Secrets in AWS Secrets Manager

The container reads these from env at startup. Create whichever ones
you actually use:

```bash
aws secretsmanager create-secret \
    --name ollama-api-key \
    --secret-string "$OLLAMA_API_KEY" \
    --region us-east-1

# Optional: for gated HF models or Hub pushes
aws secretsmanager create-secret \
    --name hf-token \
    --secret-string "$HF_TOKEN" \
    --region us-east-1
```

### 3. App Runner service configuration

In the existing service (or when configuring a fresh one through
console / Terraform), set:

| Setting | Value | Why |
|---|---|---|
| Source | ECR — `<account>.dkr.ecr.us-east-1.amazonaws.com/valonylabs-studio:latest` | Container image path |
| Build target | `cpu` (built into image) | No torch-cuda / vLLM — they'd blow the 0.5 GB ceiling |
| Port | 8000 | Matches uvicorn default |
| CPU / Memory | 0.25 vCPU / 0.5 GB | Cheapest tier; bump only if pitch traffic warrants |
| Auto Pause | **ON** | Idle cost drops to ~$3.30/mo provisioned-memory-only |
| Health check | `/healthz` | Already implemented |
| Env: `VALONY_INFERENCE_BACKEND` | `ollama` | Force the proxy backend; never load in-process |
| Env: `VALONY_CORS_ORIGINS` | `https://<service>.awsapprunner.com` | Same-origin since SPA is bundled, but set explicitly |
| Secret: `OLLAMA_API_KEY` | `arn:aws:secretsmanager:us-east-1:429100832336:secret:ollama-api-key` | |

Note: the ECR-pull IAM role for App Runner must have
`AWSAppRunnerServicePolicyForECRAccess` attached, and the secrets
access role must allow `secretsmanager:GetSecretValue` on the ARNs
above.

---

## Subsequent deploys

Every code change goes out via:

```bash
export APPRUNNER_SERVICE_ARN=arn:aws:apprunner:us-east-1:429100832336:service/...
./infra/apprunner-deploy.sh
```

That script:

1. Builds the `cpu` Docker target locally
2. Pushes to ECR with both `:latest` and `:<git-short-sha>` tags
3. Triggers `aws apprunner start-deployment`
4. Polls `Service.Status` until `RUNNING` or fails

Smoke check after every deploy:

```bash
SERVICE_URL=$(aws apprunner describe-service \
    --service-arn "$APPRUNNER_SERVICE_ARN" \
    --query 'Service.ServiceUrl' --output text)
curl "https://$SERVICE_URL/healthz"
open "https://$SERVICE_URL"   # SPA loads
```

---

## Cost monitoring

Hard $5/mo ceiling. Set both an alarm AND a manual weekly check.

### CloudWatch billing alarm

```bash
aws budgets create-budget \
    --account-id 429100832336 \
    --budget '{
        "BudgetName": "valonylabs-studio-apprunner",
        "BudgetLimit": {"Amount": "5.00", "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST",
        "CostFilters": {"Service": ["AWS App Runner"]}
    }' \
    --notifications-with-subscribers '[
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 100
            },
            "Subscribers": [{
                "SubscriptionType": "EMAIL",
                "Address": "ataliba.miguel@valonylabs.com"
            }]
        }
    ]'
```

### Weekly cost check

After the first week of real demo traffic, run:

```bash
aws ce get-cost-and-usage \
    --time-period Start=$(date -v -7d +%Y-%m-%d),End=$(date +%Y-%m-%d) \
    --granularity DAILY \
    --metrics UnblendedCost \
    --filter '{"Dimensions": {"Key": "SERVICE", "Values": ["AWS App Runner"]}}'
```

If actual >= $1.50 in the first 7 days (extrapolates to ~$6/mo), throttle:
- Reduce instance to 0.25 vCPU / 0.5 GB if not already there
- Confirm Auto Pause is on
- Drop concurrent requests cap

If still creeping over, the **exit ramp is ECS Express Mode**, not
Fly.io (per `memory/serving_cost_constraints.md`).

---

## Rollback

App Runner keeps the previous image tag in ECR; rolling back is just
re-deploying from `:<previous-sha>`:

```bash
APPRUNNER_SERVICE_ARN=arn:... IMAGE_TAG=abc1234 ./infra/apprunner-deploy.sh
```

Where `abc1234` is the prior good tag (visible in `aws ecr list-images`).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/healthz` 502 | Container crashed at startup | Check App Runner logs; common: missing `OLLAMA_API_KEY` secret |
| SPA loads but `/v1/chat` returns CORS error | `VALONY_CORS_ORIGINS` doesn't include the App Runner URL | Update env var, redeploy |
| Image pull fails | ECR access role missing on App Runner service | Attach `AWSAppRunnerServicePolicyForECRAccess` |
| First deploy is slow (~5 min) | Cold start image pull + first health check | Normal; subsequent deploys are faster |
| `/v1/forge/upload` 422 with "path rejected" | S06 path validator rejecting an upload outside the allowlist | Set `VALONY_UPLOADS_DIR=/app/data/uploads` (default — confirm it survived in the App Runner env) |
