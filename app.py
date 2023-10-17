import sys
import os
import time
import fire

import torch
from transformers import AutoTokenizer, GenerationConfig
from auto_gptq import AutoGPTQForCausalLM
import gradio as gr

from app_components import Prompter


model = None
tokenizer = None
prompter = None


def generate_prompt(instruction, input=None):
    return prompter.generate_prompt(instruction)


def stream_generator(model, tokenizer, **kwargs):
    from transformers_stream_generator.main import NewGenerationMixin, StreamGenerationConfig
    model.__class__.generate_stream = NewGenerationMixin.generate
    model.__class__.sample_stream = NewGenerationMixin.sample_stream
    
    generation_config = kwargs.pop('generation_config', GenerationConfig())
    stream_config = StreamGenerationConfig(**generation_config.to_dict(), do_stream=True)
    
    start = time.time()
    output = []
    with torch.amp.autocast(device_type=model.device.type):
        for next_tokens in model.generate_stream(
            generation_config = stream_config,
            seed=int(torch.empty((), dtype=torch.int64).random_().item()) % (2**32),
            **kwargs
        ):
            # next_tokens: (bs), where bs = 1
            output.append(next_tokens.item())
            end = time.time()
            print(f"{len(output)}: {round(len(output) / (end - start), 2)} token/s")
            yield tokenizer.decode(output, skip_special_tokens=True).strip()


@torch.no_grad()
def evaluate(
    instruction,
    input=None,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
    num_beams=1,
    max_new_tokens=64,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )
    
    for response in stream_generator(
        model, tokenizer,
        input_ids=input_ids,
        generation_config=generation_config
    ):
        yield response


def main(
    quantized_model_dir: str,
    model_type: str = "baichuan"
):
    global model, tokenizer, prompter
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    
    kernel_kwargs = {'disable_exllama': False, 'disable_exllamav2': True, 'use_triton': False}
    model = AutoGPTQForCausalLM.from_quantized(
        quantized_model_dir, torch_dtype=torch.float16, device_map={'': 0}, trust_remote_code=True, **kernel_kwargs
    )
    model.eval()
    if model.generation_config.pad_token_id is None:
        if tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = model.config.pad_token_id = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            model.generation_config.pad_token_id = model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = True
    
    prompter = Prompter(model_type)
    
    g = gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2, label="Instruction", placeholder="Tell me about alpacas."
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(minimum=0, maximum=1, value=1.0, label="Temperature"),
            gr.components.Slider(minimum=0, maximum=1, value=0.8, label="Top p"),
            gr.components.Slider(minimum=0, maximum=100, step=1, value=50, label="Top k"),
            gr.components.Slider(minimum=1, maximum=4, step=1, value=1, label="Beams"),
            gr.components.Slider(
                minimum=1, maximum=1024, step=1, value=512, label="Max tokens"
            ),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="Baichuan2 - 4bit - 26 token/s",
        description="Baichuan2 is a 13B-parameter model.",
    )
    g.queue(concurrency_count=1)
    g.launch(share=True)


if __name__ == '__main__':
    fire.Fire(main)
    