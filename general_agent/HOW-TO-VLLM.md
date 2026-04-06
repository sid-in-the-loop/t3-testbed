# Running OSS Models

## Step 1 — Serve the model on a GPU node

```bash
cd /home/ssmurali/t3-testbed/general_agent

./scripts/launch_vllm_server.sh Qwen/Qwen3-1.7B 8001
./scripts/launch_vllm_server.sh Qwen/Qwen3-4B   8002
./scripts/launch_vllm_server.sh Qwen/Qwen3-8B   8003
./scripts/launch_vllm_server.sh openai/gpt-oss-20b 8004
```

Wait ~5 min, then verify the server is up:

```bash
curl -s http://$(cat results/vllm_servers/qwen3-8b.endpoint | sed 's|/v1||')/v1/models \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])"
```

## Step 2 — Submit experiment jobs

```bash
./scripts/submit_main_table_open.sh qwen3-8b
./scripts/submit_main_table_open.sh qwen3-4b
./scripts/submit_main_table_open.sh qwen3-1.7b
./scripts/submit_main_table_open.sh gpt-oss-20b
```

These are CPU-only jobs — they route through the vLLM server API. Each submission creates 24 experiment jobs + 24 judge jobs + 1 aggregate, all chained automatically.

**No OOM from concurrent jobs.** vLLM allocates GPU memory once at startup (weights + KV cache). Concurrent requests share the same weights — they don't stack on the GPU. Requests beyond `--max-num-seqs 512` are queued. Whether the experiment job has a GPU or not makes no difference to the server.

Monitor: `squeue -u $USER`
