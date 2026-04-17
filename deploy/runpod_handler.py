"""RunPod serverless handler for Decimus LLM.

This wraps the fine-tuned model in RunPod's serverless worker format,
providing an OpenAI-compatible chat completions endpoint.

Deployment:
    1. Build Docker image with merged model
    2. Push to Docker Hub
    3. Create RunPod serverless endpoint pointing to the image
"""

import runpod
from vllm import LLM, SamplingParams

# Load model at worker startup
llm = LLM(
    model="/model/",
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.90,
)

SYSTEM_PROMPT = (
    "You are Decimus LLM, an expert orchestration advisor trained on "
    "Rimsky-Korsakov's Principles of Orchestration. You help composers "
    "transform piano sketches into full orchestral scores by recommending "
    "instrument assignments, doublings, voicings, and textures. "
    "When asked for an orchestration plan, respond with structured JSON."
)


def build_prompt(messages: list[dict]) -> str:
    """Build a Llama 3.1 chat prompt from messages."""
    prompt = "<|begin_of_text|>"

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

    # Add generation prompt for assistant
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt


def handler(event):
    """Handle inference requests."""
    input_data = event["input"]

    # Support both raw prompt and OpenAI-style messages
    if "messages" in input_data:
        messages = input_data["messages"]
        # Prepend system prompt if not present
        if not messages or messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        prompt = build_prompt(messages)
    elif "prompt" in input_data:
        prompt = input_data["prompt"]
    else:
        return {"error": "Provide 'messages' or 'prompt' in input"}

    params = SamplingParams(
        temperature=input_data.get("temperature", 0.7),
        max_tokens=input_data.get("max_tokens", 1024),
        top_p=input_data.get("top_p", 0.9),
        stop=["<|eot_id|>"],
    )

    outputs = llm.generate([prompt], params)
    generated_text = outputs[0].outputs[0].text

    return {
        "text": generated_text,
        "usage": {
            "prompt_tokens": len(outputs[0].prompt_token_ids),
            "completion_tokens": len(outputs[0].outputs[0].token_ids),
        },
    }


runpod.serverless.start({"handler": handler})
