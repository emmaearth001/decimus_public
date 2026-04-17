FROM runpod/worker-vllm:stable-cuda12.1.0

# Copy the merged Decimus LLM model
COPY decimus-llm-v1/ /model/

# Configure vLLM
ENV MODEL_NAME=/model/
ENV MAX_MODEL_LEN=2048
ENV DTYPE=float16
ENV GPU_MEMORY_UTILIZATION=0.90
