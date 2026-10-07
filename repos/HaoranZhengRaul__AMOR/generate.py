"""Cached greedy completion for the custom AMOR release."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from load_model import load_model
from amor_decode import AMORDecodeWrapper

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--router", action="store_true", help="Use the fitted inference routers.")
    p.add_argument("--prompt",required=True)
    p.add_argument("--max-new-tokens",type=int,default=32)
    p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    p.add_argument("--model-dir",type=Path,default=Path(__file__).resolve().parent)
    a=p.parse_args()
    if a.max_new_tokens<1:p.error("--max-new-tokens must be positive")
    config=json.loads((a.model_dir/"config.json").read_text())
    if a.device=="cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable; install a CUDA-enabled PyTorch build, or use --device cpu for Mamba2.")
    if a.device=="cpu" and config["model_kwargs"]["backbone"]=="gdn":
        p.error("This Gated DeltaNet implementation requires compatible GPU kernels.")
    tokenizer=AutoTokenizer.from_pretrained(config["tokenizer"],revision=config.get("tokenizer_revision"))
    model=load_model(a.model_dir,device=a.device,dtype=torch.float32,use_router=a.router)
    ids=tokenizer(a.prompt,return_tensors="pt")["input_ids"].to(a.device)
    if not ids.shape[1]:p.error("Prompt must tokenize to at least one token")
    wrapper=AMORDecodeWrapper(model)
    cast=torch.autocast("cuda",dtype=torch.bfloat16) if a.device=="cuda" else nullcontext()
    generated=[]
    with torch.inference_mode(),cast:
        logits,cache=wrapper.prefill(ids)
        for i in range(a.max_new_tokens):
            scores=logits[:,-1] if logits.ndim==3 else logits
            token=scores.argmax(-1,keepdim=True)
            value=int(token.item())
            if value==tokenizer.eos_token_id:break
            generated.append(value)
            if i+1<a.max_new_tokens:
                logits,cache,_=wrapper.decode_step(token,cache,step_idx=ids.shape[1]+i)
    print(a.prompt+tokenizer.decode(generated,skip_special_tokens=True))
if __name__=="__main__":main()
