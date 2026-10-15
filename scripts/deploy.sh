#!/usr/bin/env bash
# One-command deploy: build the image (model image + server/), push it to ECR, create/update the CloudFormation stack (infra/pmrf.yaml),
# wait for ECS to roll out, then check /healthz through API Gateway.
#   ./scripts/deploy.sh [version]          # default: $MODEL_VERSION or v7-20261010
# Env: AWS_PROFILE (nutrix), AWS_REGION (us-west-2), STACK (pmrf), SKIP_BUILD=1 to reuse the local model image
set -euo pipefail
# Windows installs that a shell opened before installation may not have on PATH
for d in "/c/Program Files/Amazon/AWSCLIV2" "/c/Program Files/Docker/Docker/resources/bin"; do
  [ -d "$d" ] && PATH="$d:$PATH"
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:-${MODEL_VERSION:-v7-20261010}}"
export AWS_PROFILE="${AWS_PROFILE:-nutrix}" AWS_REGION="${AWS_REGION:-us-west-2}"
STACK="${STACK:-pmrf}"
REPO=pmrf-api
: "${CLERK_JWT_ISSUER:?set CLERK_JWT_ISSUER before deploying}"
: "${CLERK_JWT_AUDIENCE:?set CLERK_JWT_AUDIENCE before deploying}"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com"
TAG="$VERSION-$(git -C "$ROOT" rev-parse --short HEAD)"
IMAGE="$REGISTRY/$REPO:$TAG"

echo "==> ECR repository $REPO"
aws ecr describe-repositories --repository-names "$REPO" >/dev/null 2>&1 ||
  aws ecr create-repository --repository-name "$REPO" --image-scanning-configuration scanOnPush=true >/dev/null

echo "==> image $IMAGE"
[ "${SKIP_BUILD:-0}" = 1 ] || "$ROOT/scripts/build.sh" "$VERSION"          # model team's image (model/deploy/Dockerfile)
# + server/: person data on DynamoDB + S3 instead of files
docker build --platform linux/amd64 -q -f "$ROOT/server/Dockerfile" --build-arg BASE_IMAGE="pmrf-api:$VERSION"   -t "pmrf-aws:$VERSION" "$ROOT" >/dev/null
docker tag "pmrf-aws:$VERSION" "$IMAGE"
aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
docker push "$IMAGE" | tail -1

echo "==> network (default VPC, two subnets in different AZs)"
VPC="$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)"
read -r SUBNET_A SUBNET_B < <(aws ec2 describe-subnets --filters Name=vpc-id,Values="$VPC" Name=default-for-az,Values=true \
  --query 'sort_by(Subnets,&AvailabilityZone)[0:2].SubnetId' --output text)

echo "==> stack $STACK (ECS waits for the new task to pass health checks; first run ~10 min)"
aws cloudformation deploy --stack-name "$STACK" --template-file "$ROOT/infra/pmrf.yaml" \
  --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset \
  --parameter-overrides ImageUri="$IMAGE" VpcId="$VPC" SubnetA="$SUBNET_A" SubnetB="$SUBNET_B" \
    ClerkJwtIssuer="$CLERK_JWT_ISSUER" ClerkJwtAudience="$CLERK_JWT_AUDIENCE" \
    MobileCorsOrigins="${MOBILE_CORS_ORIGINS:-http://localhost:8081,http://localhost:8082}" \
    MobileDemoPersonIds="${MOBILE_DEMO_PERSON_IDS:-cgmacros-003,cgmacros-002}"

URL="$(aws cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text)"
echo "==> $URL/healthz"
for _ in $(seq 1 12); do
  if body="$(curl -sf "$URL/healthz")"; then echo "$body"; echo "API: $URL"; exit 0; fi
  sleep 5
done
echo "health check through API Gateway failed: see CloudWatch log group /ecs/$STACK" >&2
exit 1
