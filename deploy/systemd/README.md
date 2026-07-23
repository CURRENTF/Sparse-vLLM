# systemd recovery for OpenAI serving

Run every GPU worker in its own service and keep the smart router in a separate
service. A fatal engine-step error makes the worker unready, stops Uvicorn, and
exits the worker process with status 1. `Restart=on-failure` then creates a new
process and CUDA context. The router removes an unready worker from routing and
automatically admits it again after `/readyz` succeeds.

The router does not replay a request whose worker failed. Client retries must
be bounded and idempotency-aware. A restarted worker also starts with an empty
prefix cache.

## Configure workers

Install the units for the current user:

```bash
mkdir -p ~/.config/systemd/user ~/.config/sparsevllm
cp deploy/systemd/sparsevllm-worker@.service ~/.config/systemd/user/
cp deploy/systemd/sparsevllm-router.service ~/.config/systemd/user/
```

Create one environment file per worker. For example,
`~/.config/sparsevllm/worker-gpu4.env`:

```bash
SPARSEVLLM_REPO=/home/USER/projects/Sparse-vLLM
SPARSEVLLM_PYTHON=/path/to/python
SPARSEVLLM_MODEL=/path/to/model
SPARSEVLLM_SERVED_MODEL_NAME=qwen36-27b-fp8
SPARSEVLLM_PORT=18004
SPARSEVLLM_ENGINE_KWARGS=/path/to/gpu4-engine-kwargs.json
SPARSEVLLM_REQUEST_LOG_DIR=/path/to/logs/gpu4/requests
CUDA_VISIBLE_DEVICES=4
```

Use a different port, log directory, and `CUDA_VISIBLE_DEVICES` value for each
worker. Keep model and engine configuration in versioned or archived JSON so a
restart uses exactly the same runtime settings.

## Configure the router

Create `~/.config/sparsevllm/router.env`:

```bash
SPARSEVLLM_REPO=/home/USER/projects/Sparse-vLLM
SPARSEVLLM_PYTHON=/path/to/python
SPARSEVLLM_WORKER_URLS=http://127.0.0.1:18004,http://127.0.0.1:18005
SPARSEVLLM_ROUTER_HOST=0.0.0.0
SPARSEVLLM_ROUTER_PORT=18000
SPARSEVLLM_ROUTER_REQUEST_TIMEOUT_S=30
SPARSEVLLM_ROUTER_CONTROL_TIMEOUT_S=5
SPARSEVLLM_ROUTE_LOG_DIR=/path/to/logs/router
```

Set `SPARSEVLLM_ROUTER_REQUEST_TIMEOUT_S` at least as high as the client
workload needs, while keeping the client timeout higher so routing and response
overhead cannot expire first. The simulated Deep Research benchmark uses a
900-second router timeout and a 930-second client timeout.
Keep `SPARSEVLLM_ROUTER_CONTROL_TIMEOUT_S` short so an unresponsive worker
cannot stall readiness and route selection for the full inference timeout.

Then load and start the services:

```bash
systemctl --user daemon-reload
systemctl --user enable --now sparsevllm-worker@gpu4 sparsevllm-worker@gpu5
systemctl --user enable --now sparsevllm-router
```

`StartLimitBurst=3` within five minutes prevents an indefinitely hot restart
loop. Inspect and fix the failure before clearing that limit with
`systemctl --user reset-failed SERVICE`. Use `/livez` for process liveness and
`/readyz` for traffic readiness. The router is ready only while at least one
worker is ready.
