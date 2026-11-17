# Kubernetes Deployment

Deploy `demand-ml` on Kubernetes with the provided manifests.

## Prerequisites

- Kubernetes cluster (1.19+)
- `kubectl` CLI configured to access the cluster
- Docker image built and available: `demand-ml:latest`

## Deployment

### 1. Build and push the Docker image

```bash
docker build -t demand-ml:latest .
# Push to your registry if using a private cluster
docker tag demand-ml:latest <your-registry>/demand-ml:latest
docker push <your-registry>/demand-ml:latest
```

Update `deployment.yaml` image reference if using a private registry.

### 2. Apply the manifests

```bash
kubectl apply -f k8s/
```

This deploys:
- **Deployment**: 2 replicas, rolling updates, health checks (liveness + readiness)
- **Service**: ClusterIP exposing port 80 → 8000

### 3. Verify the deployment

```bash
kubectl get pods -l app=demand-ml
kubectl get svc demand-ml
kubectl logs -l app=demand-ml -f
```

### 4. Test the service

Port-forward to access locally:

```bash
kubectl port-forward svc/demand-ml 8000:80
curl http://localhost:8000/health
```

## Configuration

Edit `deployment.yaml` to customize:

- **Replicas**: Change `spec.replicas` for scaling (default: 2)
- **Resources**: Update `resources.requests` and `resources.limits` for your workload
- **Volumes**: Replace `emptyDir` with `PersistentVolumeClaim` for persistent storage
- **Environment**: Add more env vars or ConfigMaps as needed

## Health Checks

The deployment includes:

- **Liveness probe**: Restarts unhealthy pods (30s initial delay, 10s interval)
- **Readiness probe**: Marks pods ready when `/health` responds (10s initial delay, 5s interval)

## Monitoring

Pod annotations enable Prometheus scraping:

```yaml
prometheus.io/scrape: "true"
prometheus.io/port: "8000"
prometheus.io/path: "/metrics"
```

Configure your Prometheus ServiceMonitor or scrape config to target these pods.

## Cleanup

Remove all resources:

```bash
kubectl delete -f k8s/
```

## Notes

- **Model volume**: Currently uses `emptyDir` (ephemeral). For production, mount a PVC or init-container that loads a pre-trained model.
- **Data volume**: Similarly ephemeral. Mount persistent storage for the UCI data cache.
- **MLflow tracking**: Configured as file-based (`./mlruns`). For production, use PostgreSQL or cloud storage backend.

See [../ROADMAP.md](../ROADMAP.md) for PostgreSQL and persistent storage enhancements.
