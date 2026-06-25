# Operational Scenarios

| Scenario | Solution |
| ------- | --------- |
| Model serves worse recommendations based on our real time metrics | Revert to the previous version and investigate |
| Influx of traffic creates a bottleneck for the model | Horizontal scaling, caching, storage sharding/partitioning |
| testing model performance | Implement feature release and A/B testing to determine which one performs better |
| user account deletion | identify management should send us a request, and we should scan our system to make sure account deletion has been fulfilled. keep a request id for regulatory purposes |
| CI/CD rejects a pull request | if this is related to an active incident, and it is a minor issue (e.g. noisy integration test or formatting), we should allow an override with director approval. If it's approved for an incident, we should have a separate ticket to investigate the failture to establish good hygiene. If it's not urgent, it should not be approved. |