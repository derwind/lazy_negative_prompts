# Lazy Negative Prompts

Based on arXiv:2406.02965 [Understanding the Impact of Negative Prompts: When and How Do They Take Effect?](https://arxiv.org/abs/2406.02965), mechanisms to delay the application of negative prompts are introduced.

## Installation

```sh
pip install git+https://github.com/derwind/lazy_negative_prompts.git
```

## Quickstart

```python
from diffusers import StableDiffusionPipeline
from lazy_negative_prompts import enable_lazy_negative

pipe = StableDiffusionPipeline(...)
succeeded = enable_lazy_negative(pipe)
if succeeded:
    print("lazy_negative is now enabled.")

result = pipe(
    prompt="cat",
    negative_prompt="dog",
    width=512,
    height=512,
    num_inference_steps=50,
    critical_step=5,
)
```
