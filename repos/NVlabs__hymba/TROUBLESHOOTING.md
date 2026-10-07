# Common Issues & Solutions

## 1. CUDA Version Mismatch

**Error:**
```
ImportError: /usr/local/lib/python3.10/dist-packages/torch/lib/../../../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: nvJitLinkComplete_12_4, version libnvJitLink.so.12
```

**Solution:**
- The PyTorch installation was compiled with CUDA 12.4 while the system CUDA is 12.3
- Re-install PyTorch with CUDA 12.1 to resolve this compatibility issue

## 2. Batch Size Error

**Error:**
```
IndexError: index -1 is out of bounds for dimension 1 with size 0
```

**Solution:**
- This typically occurs when trying to use batch size > 1
- Set batch size to 1

## 3. Torch Dynamo Error

**Error:**
```
AssertionError: <module 'torch.nn.init'> ... is in the tuple returned by torch.overrides.get_ignored_functions but still has an explicit override
```

**Solution:**
You can suppress this error by adding:
```python
import torch.dynamo
torch.dynamo.config.suppress_errors = True
```

## 4. Model Architecture Error

**Error:**
```
AttributeError: 'HymbaFlexAttention' object has no attribute 'q_proj'
```

**Solution:**
When loading the model, ensure you set the correct data type:
```python
# Incorrect
model = AutoModelForCausalLM.from_pretrained(repo_name, trust_remote_code=True).cuda()

# Correct
model = AutoModelForCausalLM.from_pretrained(repo_name, trust_remote_code=True).cuda().to(torch.bfloat16)
```