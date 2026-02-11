docker buildx build --platform linux/arm64 --load -t verekia/v1v2-engine .
docker save -o /tmp/v1v2-engine.tar verekia/v1v2-engine
scp /tmp/v1v2-engine.tar midgar:/tmp/
ssh midgar docker load --input /tmp/v1v2-engine.tar
ssh midgar docker compose up -d v1v2-engine