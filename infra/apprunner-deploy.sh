#!/usr/bin/env bash
# Push a new image to ECR and tell the existing App Runner service
# (account 429100832336, grandfathered) to redeploy from it.
#
# Required env (export or pass on the line):
#   ECR_REPO              ECR repo name (default: valonylabs-studio)
#   AWS_REGION            us-east-1 by default
#   APPRUNNER_SERVICE_ARN arn:aws:apprunner:... (found via `aws apprunner list-services`)
#
# Optional env:
#   IMAGE_TAG             default: $(git rev-parse --short HEAD)
#   DOCKER_TARGET         default: cpu (slim image, no torch-cuda/vllm)
#
# Usage:
#   APPRUNNER_SERVICE_ARN=arn:... ./infra/apprunner-deploy.sh
#
# What it does:
#   1. docker build --target $DOCKER_TARGET .
#   2. tag + push to $ECR_REPO:$IMAGE_TAG (and :latest)
#   3. apprunner start-deployment on $APPRUNNER_SERVICE_ARN
#   4. tail the deployment status until SUCCEEDED or FAILED
set -euo pipefail

ECR_REPO="${ECR_REPO:-valonylabs-studio}"
AWS_REGION="${AWS_REGION:-us-east-1}"
DOCKER_TARGET="${DOCKER_TARGET:-cpu}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"

if [[ -z "${APPRUNNER_SERVICE_ARN:-}" ]]; then
  echo "ERROR: APPRUNNER_SERVICE_ARN not set." >&2
  echo "  Find it with: aws apprunner list-services --region $AWS_REGION" >&2
  exit 2
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo "==> Building image ($DOCKER_TARGET target, tag=$IMAGE_TAG)"
docker build --target "$DOCKER_TARGET" -t "$ECR_REPO:$IMAGE_TAG" .

echo "==> Logging in to ECR ($ECR_URI)"
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_URI"

echo "==> Tagging and pushing"
docker tag "$ECR_REPO:$IMAGE_TAG" "$ECR_URI:$IMAGE_TAG"
docker tag "$ECR_REPO:$IMAGE_TAG" "$ECR_URI:latest"
docker push "$ECR_URI:$IMAGE_TAG"
docker push "$ECR_URI:latest"

echo "==> Triggering App Runner deployment"
DEPLOYMENT_ID=$(aws apprunner start-deployment \
  --service-arn "$APPRUNNER_SERVICE_ARN" \
  --region "$AWS_REGION" \
  --query 'OperationId' \
  --output text)
echo "    Deployment id: $DEPLOYMENT_ID"

echo "==> Waiting for deployment to complete"
while true; do
  STATUS=$(aws apprunner describe-service \
    --service-arn "$APPRUNNER_SERVICE_ARN" \
    --region "$AWS_REGION" \
    --query 'Service.Status' \
    --output text)
  echo "    Service status: $STATUS"
  case "$STATUS" in
    RUNNING)         echo "==> Deployment SUCCEEDED ($IMAGE_TAG)"; exit 0 ;;
    OPERATION_IN_PROGRESS) sleep 15 ;;
    *)               echo "==> Deployment status: $STATUS" >&2; exit 1 ;;
  esac
done
