# LLaMA Box (V2)

[![](https://img.shields.io/github/actions/workflow/status/gpustack/llama-box/ci.yml?label=ci)](https://github.com/gpustack/llama-box/actions)
[![](https://img.shields.io/github/license/gpustack/llama-box?label=license)](https://github.com/gpustack/llama-box#license)
[![](https://img.shields.io/github/downloads/gpustack/llama-box/total)](https://github.com/gpustack/llama-box/releases)

LLaMA Box is an LM inference server(pure API, w/o frontend assets) based on
the [llama.cpp](https://github.com/ggml-org/llama.cpp)
and [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

## Agenda

- [Features](#features)
- [Supports](#supports)
- [Examples](#examples)
- [Usage](#usage)
- [Server API](#server-api)
- [Tools](#tools)

## Features

- Compatible with [OpenAI Chat API](https://platform.openai.com/docs/api-reference/chat).
    + Support [OpenAI Chat Vision API](https://platform.openai.com/docs/guides/vision).
        - [x] LLaVA Series (w/ `--chat-tempalte llava` or `--chat-template llava-mistral`)
        - [x] MiniCPM VL Series
        - [x] Qwen2 VL Series
        - [x] GLM-Edge-V Series (w/ `--chat-template llava`)
        - [x] Granite VL Series
        - [x] Gemma3 VL Series
        - [x] SmolVLM Series
        - [x] Pixtral Series
        - [x] MobileVLM Series (w/ `--chat-template deepseek`)
        - [x] Mistral Small 3.1 (w/ `--chat-template mistral-v7`)
        - [x] Qwen2.5 VL Series, please
          use the model files converted by [ggml-org/llama.cpp#12402](https://github.com/ggml-org/llama.cpp/pull/12402).
        - [x] InternVL2/InternVL3 Series
        - [x] LLaMA4 Series
        - [x] ...
    + ([Experimental](https://github.com/ggml-org/llama.cpp/discussions/13759))
      Support [OpenAI Chat Audio-In API](https://platform.openai.com/docs/guides/audio?example=audio-in)
        - [x] UltraVox Series
        - [x] Qwen2 Audio Series
        - [x] ...
    ```shell
      $ # Avoid memory raising when processing high-resolution images, like Qwen2 VL model, launch box with --visual-max-image-size 1344.
      $ llama-box -c 8192 -np 4 --host 0.0.0.0 -m ... --mmproj ... --visual-max-image-size 1344
      $ # The box will resize the image automatically when the image size exceeds 1344x1344.
    ```
    + Support [OpenAI Function calling API](https://platform.openai.com/docs/guides/function-calling).
        - [x] LLaMA3 Series
        - [x] Granite Series
        - [x] Mistral Series
        - [x] Hermes2 / Qwen2 Series
        - [x] ChatGLM4 Series
        - [x] CommandR Series (w/ `--jinja`)
        - [x] FunctionaryV3 Series (w/ `--jinja`)
        - [x] DeepSeekR1 Series (w/ `--jinja`, experimental)
        - [x] ...
- Compatible with [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings).
- Compatible with [OpenAI Images API](https://platform.openai.com/docs/api-reference/images),
  see our [Image Collection](https://huggingface.co/collections/gpustack/image-672dafeb2fa0d02dbe2539a9).
- Compatible with [(Legacy) OpenAI Completions API](https://beta.openai.com/docs/api-reference/completions).
- Compatible with [Jina Rerank API](https://api.jina.ai/redoc#tag/rerank),
  see our [Reranker Collection](https://huggingface.co/collections/gpustack/reranker-6721a234527f6fcd90deedc4).
- Support speculative decoding: draft model or n-gram lookup.
- Support RPC server mode, which can serve as a remote inference backend.
- For non-image models, split offloading layers across multiple devices, including remote RPC server.
  ```shell
    $ # Assume that there are 1 remote RPC server and 3 available GPUs, launch box as below.
    $ llama-box -c 8192 -np 4 --host 0.0.0.0 -m <non-image model> --rpc remote-ip:remote-port --tensor-split 1,2,3
    $ # Same as --tensor-split 1,2,3,0. 
    $ # The remote RPC server will handle 1/6 of the model, the 1st GPU will handle 1/3 of the model, and the 2nd GPU will handle 1/2 of the model. 
    $ # Nothing to do with the 3rd GPU.
    
    $ # Assume that there are 1 remote RPC servers and 3 available GPUs, launch box as below.
    $ llama-box -c 8192 -np 4 --host 0.0.0.0 -m <non-image model> --rpc remote-ip:remote-port --tensor-split 0,0,1,1
    $ # The 2nd GPU will handle 1/2 of the model, and the 3rd GPU will handle 1/2 of the model.
    $ # Nothing to do with the remote RPC server and the 1st GPUs.
  ```
- For image models, split offloading different components across multiple devices, include remote RPC server.
  ```shell
    $ # Assume that there are 1 remote RPC server and 3 available GPUs, launch box as below.
    $ llama-box -np 4 --host 0.0.0.0 -m <image model> --rpc remote-ip:remote-port --tensor-split 1,1,1
    $ # Same as --tensor-split 1,1,1,0.
    $ # The remote RPC server will handle text encoder part, the 1st GPU will handle VAE part, and the 2nd GPU will handle diffusion part.
    $ # Nothing to do with the 3rd GPU.
    
    $ # Assume that there are 1 remote RPC server and 3 available GPUs, launch box as below.
    $ llama-box -np 4 --host 0.0.0.0 -m <image model> --rpc remote-ip:remote-port --tensor-split 0,0,1,1
    $ # Then 2nd GPU will handle text encoder and VAE parts, and the 3rd GPU will handle diffusion part.
    $ # Nothing to do with the remote RPC server and the 1st GPUs.
  ```
- Support injecting `X-Request-ID` http header for tracking requests.
  ```shell
    $ # Launch box.
    $ llama-box -c 8192 -np 4 --host 0.0.0.0 -m <model>
    
    $ # Inject X-Request-ID: trace-id to track the request.
    $ curl --silent --no-buffer http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -H "X-Request-ID: trace-id" -d '{"model": "demo", "messages": [{"role":"user", "content":"Introduce Beijing in 50 words."}]}'
    $ # View logs
  ```
- Support `X-Request-Tokens-Per-Second` http header for limiting the number of tokens per second.
  ```shell
    $ # Launch box with -tps -1.
    $ llama-box -c 8192 -np 4 --host 0.0.0.0 -m <model> --tokens-per-second -1
  
    $ # For level 1 users, inject X-Request-Tokens-Per-Second: 10 to limit the number of tokens per second to 10.
    $ curl --silent --no-buffer http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -H "X-Request-Tokens-Per-Second: 10" -d '{"stream": true, "model": "demo", "messages": [{"role":"user", "content":"Introduce Beijing in 50 words."}]}'

    $ # For level 2 users, inject X-Request-Tokens-Per-Second: 20 to limit the number of tokens per second to 20.
    $ curl --silent --no-buffer http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -H "X-Request-Tokens-Per-Second: 20" -d '{"stream": true, "model": "demo", "messages": [{"role":"user", "content":"Introduce Beijing in 50 words."}]}'

    $ # For super users, let the box handle the request without limiting the number of tokens per second.
    $ curl --silent --no-buffer http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"stream": true, "model": "demo", "messages": [{"role":"user", "content":"Introduce Beijing in 50 words."}]}'
  ```

## Supports

Download LLaMA Box from [the latest release](https://github.com/gpustack/llama-box/releases/latest) page please, now
LLaMA Box supports the following platforms.

| Backend                                             | OS/Arch<sup>Built-on-Release</sup>                                                                                                    | Device Requirement                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|-----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **NVIDIA CUDA <br/>12.8 (12.8.0)**                  | `linux/amd64`<sup>RockyLinux 8.9</sup><br/> `linux/arm64`<sup>RockyLinux 8.9</sup><br/> `windows/amd64`<sup>Windows Server 2019</sup> | [Compute capability](https://developer.nvidia.com/cuda-gpus) matches: <br/>`6.0`, `6.1`, <br/>`7.0`, `7.5`, <br/>`8.0`, `8.6`, `8.9` <br/>`9.0`, <br/>`10.0`, `10.1`, <br/>`12.0`. <br/>[Driver version](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id4) requires: <br/>`>=525.60.13(linux)` <br/>`>=528.33(windows)`.                                                                                                                         |
| **NVIDIA CUDA <br/>12.4 (12.4.0)**                  | `linux/amd64`<sup>CentOS 7</sup><br/> `linux/arm64`<sup>RockyLinux 8.9</sup><br/> `windows/amd64`<sup>Windows Server 2019</sup>       | [Compute capability](https://developer.nvidia.com/cuda-gpus) matches: <br/>`6.0`, `6.1`, <br/>`7.0`, `7.5`, <br/>`8.0`, `8.6`, `8.9` <br/>`9.0`. <br/>[Driver version](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id4) requires: <br/>`>=525.60.13(linux)` <br/>`>=528.33(windows)`.                                                                                                                                                           |
| **AMD ROCm/HIP <br/>6.2 (6.2.4)**                   | `linux/amd64`<sup>CentOS 7</sup><br/> `windows/amd64`<sup>Windows Server 2019</sup>                                                   | LLVM target matches: <br/>`gfx906 (linux)`, <br/>`gfx908 (linux)`, <br/>`gfx90a (linux)`, <br/>`gfx942 (linux)`, <br/>`gfx1030`, `gfx1031`, `gfx1032`, <br/>`gfx1100`, `gfx1101`, `gfx1102`, <br/>see [Linux Installation](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.4/reference/system-requirements.html), <br/>[Windows Installation](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-6.2.4/reference/system-requirements.html). |
| **Intel oneAPI <br/>2025.0 (2025.0.0)**             | `linux/amd64`<sup>Ubuntu 22.04</sup><br/> `windows/amd64`<sup>Windows Server 2019</sup>                                               | Support [Intel oneAPI](https://en.wikipedia.org/wiki/OneAPI_(compute_acceleration)), <br/>see [Toolkit Requirements](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-base-toolkit-system-requirements.html).                                                                                                                                                                                                               |
| **LunarG Vulkan <br/>1.4 (1.4.304)**                | `linux/amd64`<sup>Ubuntu 22.04</sup><br/> `windows/amd64`<sup>Windows Server 2019</sup>                                               | Support [LunarG Vulkan](https://vulkan.lunarg.com/), <br/>check [Support Devices](https://vulkan.gpuinfo.org/listdevices.php).                                                                                                                                                                                                                                                                                                                                         |
| **Huawei Ascend CANN <br/>8.1.rc1 (8.1.rc1.beta1)** | `linux/amd64`<sup>Ubuntu 20.04</sup><br/> `linux/arm64`<sup>Ubuntu 20.04</sup>                                                        | `Ascend 910b`, `Ascend 310p`, <br/>see [Ascend Document](https://www.hiascend.com/en/document).                                                                                                                                                                                                                                                                                                                                                                        |
| **Huawei Ascend CANN <br/>8.0 (8.0.0.beta1)**       | `linux/amd64`<sup>Ubuntu 20.04</sup><br/> `linux/arm64`<sup>Ubuntu 20.04</sup>                                                        | `Ascend 910b`, `Ascend 310p`, <br/>see [Ascend Document](https://www.hiascend.com/en/document).                                                                                                                                                                                                                                                                                                                                                                        |
| **HYGON DTK(DCU Toolkit) <br/>24.04 (24.04.3)**     | `linux/amd64`<sup>Ubuntu 20.04</sup><br/>                                                                                             | `K100-AI`, <br/>see [DTK Community](https://developer.sourcefind.cn/?s=Note).                                                                                                                                                                                                                                                                                                                                                                                          |
| **Moore Threads MUSA <br/>rc4.0 (rc4.0.1)**         | `linux/amd64`<sup>Ubuntu 22.04</sup><br/>                                                                                             | `MTT S4000`, `MTT S80`, <br/>see [Moor Threads Website](https://en.mthreads.com).                                                                                                                                                                                                                                                                                                                                                                                      |
| **Apple Metal 3**                                   | `darwin/amd64`<sup>macOS 13.7</sup><br/> `darwin/arm64`<sup>macOS 14.7</sup>                                                          | Support [Apple Metal](https://developer.apple.com/metal/), <br/>see [Metal Framework](https://developer.apple.com/documentation/metal?language=objc).                                                                                                                                                                                                                                                                                                                  |
| _AVX2_                                              | `darwin/amd64`<sup>macOS 13.7</sup><br/> `linux/amd64`<sup>CentOS 7</sup><br/> `windows/amd64`<sup>Windows Server 2019</sup>          | CPUs support AVX2, <br/>see [Wikipedia](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions_2).                                                                                                                                                                                                                                                                                                                                        |
| _Advanced SIMD (NEON)_                              | `linux/arm64`<sup>Ubuntu 18.04</sup><br/> `windows/arm64`<sup>Windows Server 2019</sup>                                               | CPUs support <br/>Advanced SIMD (NEON), <br/>see [Wikipedia](https://en.wikipedia.org/wiki/ARM_architecture_family#Advanced_SIMD_(Neon)).                                                                                                                                                                                                                                                                                                                              |
| _AVX512_                                            | `linux/amd64`<sup>RockyLinux 8.9</sup><br/> `windows/amd64`<sup>Windows Server 2019</sup>                                             | CPUs support AVX512, <br/>see [Wikipedia](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-512).                                                                                                                                                                                                                                                                                                                                                           |

> [!NOTE]
> - CentOS 7 (glibc 2.17)
> - Ubuntu 18.04 (glibc 2.27)
> - RockyLinux 8.9 (glibc 2.28)
> - Ubuntu 20.04 (glibc 2.31)
> - Ubuntu 22.04 (glibc 2.34)
> - macOS
    13.7.5 ([Xcode 15.2](https://github.com/actions/runner-images/blob/main/images/macos/macos-13-Readme.md#xcode))
> - macOS
    14.7.5 ([Xcode 16.2](https://github.com/actions/runner-images/blob/main/images/macos/macos-14-Readme.md#xcode))
> - Windows Server
    2019 ([MSVC 14.42](https://github.com/actions/runner-images/blob/main/images/windows/Windows2019-Readme.md#microsoft-visual-c))

## Examples

> **Note**:
> [LM Studio](https://lmstudio.ai/) provides a fantastic UI for downloading the GGUF model from Hugging Face.
> The GGUF model files used in the following examples are downloaded via LM Studio.

- Chat completion via [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
  model. Use GGUF files
  from [unsloth/Qwen3-8B-GGUF](https://huggingface.co/unsloth/Qwen3-8B-GGUF/tree/main?show_file_info=Qwen3-8B-Q8_0.gguf).

    ```shell
    $ # Provide 4 sessions(allowing 4 parallel chat users), with a max of 8192 tokens per session.
    $ llama-box -c 8192 -np 4 --host 0.0.0.0 -m ~/.cache/lm-studio/models/unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf
    
    $ # Call with curl,
    $ curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "qwen3", "messages": [{"role":"user", "content":"Introduce Beijing in 50 words."}]}'

    $ # or use the chat.sh tool.
    $ ./llama-box/tools/chat.sh "Introduce Beijing in 50 words."
    ```

- Chat completion with vision explanation
  via [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
  model. Use GGUF files
  from [ggml-org/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF/tree/main?show_file_info=Qwen2.5-VL-7B-Instruct-Q8_0.gguf).

    ```shell
    $ # Provide 4 session(allowing 4 parallel chat users), with a max of 8192 tokens per session.
    $ llama-box -c 8192 -np 4 --host 0.0.0.0 -m ~/.cache/lm-studio/models/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q8_0.gguf --mmproj ~/.cache/lm-studio/models/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf
    
    $ # Chat with image base64.
    $ IMAGE_URL="$(echo "data:image/jpeg;base64,$(curl https://raw.githubusercontent.com/haotian-liu/LLaVA/main/llava/serve/examples/extreme_ironing.jpg --output - | base64)")"; \
      echo "{\"model\": \"qwen2.5-vl\", \"temperature\": 0.1, \"messages\": [{\"role\":\"system\", \"content\": [{\"type\": \"text\", \"text\": \"You are a helpful assistant.\"}]}, {\"role\":\"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"$IMAGE_URL\"}}, {\"type\": \"text\", \"text\": \"What is unusual about this image?\"}]}]}" > /tmp/data.json
    
    $ # Call with curl,
    $ curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/data.json

    $ # or use the chat.sh tool.
    $ ./llama-box/tools/chat.sh @/tmp/data.json
  
    $ # Chat with image url.
    $ IMAGE_URL="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"; \
      echo "{\"model\": \"qwen2.5-vl\", \"temperature\": 0.1, \"messages\": [{\"role\":\"user\", \"content\": [{\"type\":\"text\",\"text\":\"What is in this image?\"}, {\"type\": \"image_url\", \"image_url\": {\"url\": \"$IMAGE_URL\"}}]}]}" > /tmp/data.json
    
    $ # Call with curl,
    $ curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/data.json
  
    $ # or use the chat.sh tool.
    $ ./llama-box/tools/chat.sh @/tmp/data.json
    ```

- Chat completion with audio explanation
  via [fixie-ai/ultravox-v0_5-llama-3_2-1b](https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_2-1b) model. Use GGUF
  files
  from [ggml-org/ultravox-v0_5-llama-3_2-1b-GGU](https://huggingface.co/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF/tree/main?show_file_info=Llama-3.2-1B-Instruct-Q8_0.gguf)

    ```shell
    $ # Provide 4 session(allowing 4 parallel chat users), with a max of 8192 tokens per session.
    $ llama-box -c 8192 -np 4 --host 0.0.0.0 -m ~/.cache/lm-studio/models/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf --mmproj ~/.cache/lm-studio/models/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF/mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf
  
    $ # Chat with audio base64.
    $ AUDIO_DATA="$(curl https://upload.wikimedia.org/wikipedia/commons/transcoded/6/6f/Apollo13-wehaveaproblem.ogg/Apollo13-wehaveaproblem.ogg.mp3 --output - | base64)"; \
      echo "{\"model\": \"ultravox\", \"temperature\": 0.1, \"messages\": [{\"role\":\"system\", \"content\": [{\"type\": \"text\", \"text\": \"You are a helpful assistant.\"}]}, {\"role\":\"user\", \"content\": [{\"type\": \"input_audio\", \"input_audio\": {  \"format\": \"mp3\", \"data\": \"$AUDIO_DATA\"}}, {\"type\": \"text\", \"text\": \"How many times has roger appeared?\"}]}]}" > /tmp/data.json
    
    $ # Call with curl,
    $ curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/data.json

    $ # or use the chat.sh tool.
    $ ./llama-box/tools/chat.sh @/tmp/data.json
    ```

- Chat with function calling via [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) model.
  Use GGUF files
  from [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/main?show_file_info=qwen2.5-0.5b-instruct-fp16.gguf).

    ```shell
    $ # Provide 4 session(allowing 4 parallel chat users), with a max of 8192 tokens per session.
    $ llama-box -c 8192 -np 4 --host 0.0.0.0 -m ~/.cache/lm-studio/models/Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-fp16.gguf
    
    $ # Call with curl,
    $ curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "qwen2.5", "messages": [{"role":"user","content":"What is the weather like in Paris today?"}], "tools": [{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]}'
    
    $ # or use the chat.sh tool.
    $ TOOLS_WITH=true ./llama-box/tools/chat.sh "What is the weather like in Paris today?"
    ```

- Image generation
  via [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
  model.
  Use GGUF files
  from [gpustack/stable-diffusion-v3-5-medium-GGUF](https://huggingface.co/gpustack/stable-diffusion-v3-5-medium-GGUF/tree/main?show_file_info=stable-diffusion-v3-5-medium-FP16.gguf).

    ```shell
    $ # Provide 1 session(allowing 1 parallel chat user).
    $ llama-box -np 1 --host 0.0.0.0 -m ~/.cache/lm-studio/models/gpustack/stable-diffusion-v3.5-medium-GGUF/stable-diffusion-v3-5-medium-FP16.gguf --images
    
    $ # Call with curl,
    $ curl http://localhost:8080/v1/images/generations -H "Content-Type: application/json" -d '{"model": "sd3-medium", "prompt": "A lovely cat"}'
    
    $ # or use the image_generate.sh tool.
    $ ./llama-box/tools/image_generate.sh "A lovely cat"
    ```

- Image editing(inpainting)
  via [black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) model. Use
  GGUF files
  from [gpustack/FLUX.1-Fill-dev-GGUF](https://huggingface.co/gpustack/FLUX.1-Fill-dev-GGUF/tree/main?show_file_info=FLUX.1-Fill-dev-Q8_0.gguf).

    ```shell
    $ # Provide 1 session(allowing 1 parallel chat user).
    $ llama-box -np 1 --host 0.0.0.0 -m ~/.cache/lm-studio/models/gpustack/FLUX.1-Fill-dev-GGUF/FLUX.1-Fill-dev-Q8_0.gguf --images
    
    $ # Call with curl,
    $ curl https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png --output /tmp/input.png
    $ curl https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png --output /tmp/mask.png
  
    $ # or use the image_edit.sh tool.
    $ IMAGE=/tmp/input.png MASK=/tmp/mask.png ./llama-box/tools/image_edit.sh "a tiger sitting on a park bench"
    ```

- Draft model speculative decoding via [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
  and [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) models. Use GGUF files
  from [Qwen/Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/tree/main?show_file_info=qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf)
  and [Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/main?show_file_info=qwen2.5-1.5b-instruct-q8_0.gguf).

    ```shell
    $ # Provide 4 session(allowing 4 parallel chat users), with a max of 8192 tokens per session.
    $ llama-box -c 8192 -np 4 --host 0.0.0.0 -m ~/.cache/lm-studio/models/Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf -md ~/.cache/lm-studio/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q8_0.gguf --draft 8

    $ # Call with curl,
    $ curl http://localhost:8080/v1/completions -H "Content-Type: application/json" -d '{"model": "qwen2.5", "stream": true, "prompt": "Write a short story about a cat and a dog, more than 100 words."}'

    $ # or use the chat.sh tool.
    $ ./llama-box/tools/chat.sh "Write a short story about a cat and a dog, more than 100 words."
    ```

- Lookup speculative decoding
  via [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) model. Use GGUF files
  from [QuantFactory/Mistral-Nemo-Instruct-2407-GGUF](https://huggingface.co/QuantFactory/Mistral-Nemo-Instruct-2407-GGUF/tree/main?show_file_info=Mistral-Nemo-Instruct-2407.Q5_K_M.gguf).

    ```shell
    $ # Provide 2 session(allowing 2 parallel chat users), with a max of 16384 tokens per session.
    $ llama-box -c 16384 -np 2 --host 0.0.0.0 -m ~/.cache/lm-studio/models/QuantFactory/Mistral-Nemo-Instruct-2407-GGUF/Mistral-Nemo-Instruct-2407.Q5_K_M.gguf --lookup-ngram-min 1 --draft 8

    $ CONTENT="$(curl https://en.wikipedia.org/w/api.php\?action\=query\&format\=json\&titles\=Medusa\&prop\=extracts\&exintro\&explaintext | jq '.query.pages | to_entries | .[0].value.extract | gsub("\n"; "\\n") | gsub("\t"; "\\t")')"; \
      echo "{\"model\": \"mistral-nemo\", \"stream\": true, \"messages\": [{\"role\":\"user\", \"content\": [{\"type\": \"text\", \"text\": \"Please read the following content and summarize the article in 5 sentences.\"}, {\"type\": \"text\", \"text\": "$CONTENT"}]}]}" > /tmp/data.json

    $ # Call with curl,
    $ curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/data.json

    $ # or use the chat.sh tool.
    $ ./llama-box/tools/chat.sh @/tmp/data.json
    ```

- Maximize search relevancy and RAG accuracy
  via [jinaai/jina-reranker-v1-tiny-en](https://huggingface.co/jinaai/jina-reranker-v1-tiny-en) model. Use GGUF files
  from [gpustack/jina-reranker-v1-tiny-en-GGUF](https://huggingface.co/gpustack/jina-reranker-v1-tiny-en-GGUF/tree/main?show_file_info=jina-reranker-v1-tiny-en-FP16.gguf).

    ```shell
    $ # Provide 4 session(allowing 4 parallel chat users), with a max of 8192 tokens per session.
    $ llama-box -c 8192 -np 4 --host 0.0.0.0 -m ~/.cache/lm-studio/models/gpustack/jina-reranker-v1-tiny-en-GGUF/jina-reranker-v1-tiny-en-FP16.gguf --rerank
    
    $ # Call with curl.
    $ curl http://localhost:8080/v1/rerank -H "Content-Type: application/json" -d '{"model":"jina-reranker-v1-tiny-en","query":"Organic skincare products for sensitive skin","top_n":3,"documents":["Eco-friendly kitchenware for modern homes","Biodegradable cleaning supplies for eco-conscious consumers","Organic cotton baby clothes for sensitive skin","Natural organic skincare range for sensitive skin","Tech gadgets for smart homes: 2024 edition","Sustainable gardening tools and compost solutions","Sensitive skin-friendly facial cleansers and toners","Organic food wraps and storage solutions","All-natural pet food for dogs with allergies","oga mats made from recycled materials"]}'
    ```

- RPC server mode.

  In RPC server mode, LLaMA Box functions as the `ggml` backend on a remote host. This setup allows non-RPC server
  instances (clients) to communicate with the RPC servers, offloading computational tasks to them.

  While the RPC server facilitates the use of larger models, it requires the RPC client to transfer the necessary
  computational materials. This transfer can lead to increased startup times for the RPC client. Additionally, network
  latency and bandwidth limitations may impact the overall performance of the RPC client.

  By understanding these dynamics, users can better manage expectations and optimize their use of LLaMA Box in an RPC
  server environment.

    ```mermaid
    flowchart TD
    clix-->|TCP|srva
    clix-->|TCP|srvb
    cliy-->|TCP|srvb
    cliy-.->|TCP|srvn
    subgraph hostn[Any]
    srvn["llama-box-*-cuda/metal/... (rpc server)"]
    end
    subgraph hostb[Apple Mac Studio]
    srvb["llama-box-*-metal (rpc server)"]
    end
    subgraph hosta[NVIDIA 4090]
    srva["llama-box-*-cuda (rpc server)"]
    end
    subgraph hosty[Apple Mac Max]
    cliy["llama-box-*-metal"]
    end
    subgraph hostx[NVIDIA 4080]
    clix["llama-box-*-cuda"]
    end
    style hostn stroke:#66,stroke-width:2px,stroke-dasharray: 5 5
    ```

    ```shell
    $ # Start the RPC server on the main GPU 0, reserve 1 GiB memory.
    $ llama-box --rpc-server-host 0.0.0.0 --rpc-server-port 8081 --rpc-server-main-gpu 0 --rpc-server-reserve-memory 1024
    ```

## Usage

```shell
usage: llama-box [options]

general:

  -h,    --help, --usage          Print usage and exit
         --version                Print version and exit
         --system-info            Print system info and exit
         --list-devices           Print list of available devices and exit
  -v,    --verbose, --log-verbose 
                                  Set verbosity level to infinity (i.e. log all messages, useful for debugging)
  -lv,   --verbosity, --log-verbosity V
                                  Set the verbosity threshold, messages with a higher verbosity will be ignored
         --log-colors             Enable colored logging

server:

         --host HOST              IP address to listen, or bind to an UNIX socket if the address ends with .sock (default: 127.0.0.1)
         --port PORT              Port to listen (default: 8080)
  -to    --timeout N              Server read/write timeout in seconds (default: 600)
         --threads-http N         Number of threads used to process HTTP requests (default: -1)
         --conn-idle N            Server connection idle in seconds (default: 60)
         --conn-keepalive N       Server connection keep-alive in seconds (default: 15)
  -m,    --model FILE             Model path (default: models/7B/ggml-model-f16.gguf)
  -a,    --alias NAME             Model name alias
         --lora FILE              Apply LoRA adapter (implies --no-mmap)
         --lora-scaled FILE SCALE 
                                  Apply LoRA adapter with user defined scaling S (implies --no-mmap)
         --lora-init-without-apply
                                  Load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: disabled)
  -s,    --seed N                 RNG seed (default: -1, use random seed for -1)
         --no-flash-attn          Disable Flash Attention, which can increase (V)RAM but reduce computation
  -fa,   --flash-attn             Enable Flash Attention, which can reduce (V)RAM but increase computation
         --swa-full               Use full-size SWA cache (default disabled)
         --metrics                Enable prometheus compatible metrics endpoint (default: disabled)
         --embeddings             Enable embedding endpoint (default: disabled)
         --images                 Enable image endpoint (default: disabled)
         --rerank                 Enable reranking endpoint (default: disabled)
         --rpc SERVERS            A comma-separated list of RPC server
  -ts,   --tensor-split SPLIT     Fraction of the model to offload to each device, comma-separated list of proportions, e.g. 3,1
                                  For image models, indicate which device should be able to offload
  -ngl,  --gpu-layers,  --n-gpu-layers N
                                  Number of layers to store in VRAM
                                  -ngl 0 means no offloading
  -ot,   --override-tensor PATTERN_1=BUFFER_TYPE_1,PATTERN_2=BUFFER_TYPE_2,...
                                  Override tensor buffer type, for example, use --override-tensor "[2-9][0-9]\.ffn_.*_exps\.=CPU" to keep experts of layers 20-99 in the CPU
         --no-warmup              Disable warm up the model with an empty run
         --warmup                 Enable warm up the model with an empty run, which is used to occupy the (V)RAM before serving

server/completion:

  -dev,  --device <dev1,dev2,...> 
                                  A comma-separated list of devices to use for offloading (none = don't offload)
                                  Use --list-devices to see a list of available devices
  -sm,   --split-mode SPLIT_MODE  How to split the model across multiple GPUs, one of:
                                    - none: use one GPU only
                                    - layer (default): split layers and KV across GPUs
                                    - row: split rows across GPUs, store intermediate results and KV in --main-gpu
  -mg,   --main-gpu N             The device to use for the model
                                  Work with --split-mode none|row, or indicate the device to offload projector model specified by --mmproj (default: 0)
         --override-kv KEY=TYPE:VALUE
                                  Advanced option to override model metadata by key, may be specified multiple times
                                  Types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false
         --chat-template BUILTIN  Set built-in chat template (default: analyze from model's metadata)
                                  Only built-in templates are accepted, implicit reset --jinja setting
                                  List of built-in templates: bailing, chatglm3, chatglm4, chatml, command-r, deepseek, deepseek2, deepseek3, exaone3, falcon, falcon3, gemma, gigachat, glmedge, granite, llama2, llama2-sys, llama2-sys-bos, llama2-sys-strip, llama3, llama4, llava, llava-mistral, megrez, minicpm, mistral-v1, mistral-v3, mistral-v3-tekken, mistral-v7, mistral-v7-tekken, monarch, openchat, orion, phi3, phi4, rwkv-world, smolvlm, vicuna, vicuna-orca, yandex, zephyr
         --jinja                  Enable jinja template for chat, implicit reset --chat-template and --chat-template-file setting (default: disabled)
         --chat-template-file FILE
                                  Set jinja chat template (default: take from model's metadata)
                                  Required --jinja set before
                                  
         --slot-save-path PATH    Path to save slot kv cache (default: disabled)
  -tps   --tokens-per-second N    Maximum number of tokens per second (default: 0, 0 = disabled, -1 = try to detect)
                                  When enabled, limit the request within its X-Request-Tokens-Per-Second HTTP header
  -t,    --threads N              Number of threads to use during generation (default: -1)
  -C,    --cpu-mask M             Set CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: "")
  -Cr,   --cpu-range lo-hi        Range of CPUs for affinity. Complements --cpu-mask
         --cpu-strict <0|1>       Use strict CPU placement (default: 0)
                                  
         --prio N                 Set process/thread priority (default: 0), one of:
                                    - 0-normal
                                    - 1-medium
                                    - 2-high
                                    - 3-realtime
         --poll <0...100>         Use polling level to wait for work (0 - no polling, default: 50)
                                  
  -tb,   --threads-batch N        Number of threads to use during batch and prompt processing (default: same as --threads)
  -Cb,   --cpu-mask-batch M       Set CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask)
  -Crb,  --cpu-range-batch lo-hi  Ranges of CPUs for affinity. Complements --cpu-mask-batch
         --cpu-strict-batch <0|1> 
                                  Use strict CPU placement (default: same as --cpu-strict)
         --prio-batch N           Set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: --priority)
         --poll-batch <0...100>   Use polling to wait for work (default: same as --poll
  -c,    --ctx-size N             Size of the prompt context (default: 4096, 0 = loaded from model)
         --no-context-shift       Disable context shift on infinite text generation and long prompt embedding
         --context-shift          Enable context shift on infinite text generation and long prompt embedding
  -b,    --batch-size N           Logical batch size.
                                  Increasing this value above the value of the physical batch size may improve prompt processing performance when using multiple GPUs with pipeline parallelism. (default: 2048)
  -ub,   --ubatch-size N          Physical batch size, which is the maximum number of tokens that may be processed at a time.
                                  Increasing this value may improve performance during prompt processing, at the expense of higher memory usage. (default: 512)
         --keep N                 Number of tokens to keep from the initial prompt (default: 0)
         --no-escape              Disable process escape sequences
  -e,    --escape                 Process escapes sequences (\n, \r, \t, \', \", \\) (default: true)
         --samplers SAMPLERS      Samplers that will be used for generation in the order, separated by ';' (default: penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature)
         --sampling-seq SEQUENCE  Simplified sequence for samplers that will be used (default: edskypmxt)
         --temp T                 Temperature (default: 0.8)
         --top-k N                Top-K sampling (default: 40, 0 = disabled)
         --top-p N                Top-P sampling (default: 0.9, 1.0 = disabled)
         --min-p N                Min-P sampling (default: 0.1, 0.0 = disabled)
         --top-nsigma N           Top-N-Sigma sampling (default: -1.0, -1.0 = disabled)
         --xtc-probability N      XTC probability (default: 0.0, 0.0 = disabled)
         --xtc-threshold N        XTC threshold (default: 0.1, 1.0 = disabled)
         --typical N              Locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)
         --repeat-last-n N        Last n tokens to consider for penalize (default: 64, 0 = disabled, -1 = ctx_size)
         --repeat-penalty N       Penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled)
         --presence-penalty N     Repeat alpha presence penalty (default: 0.0, 0.0 = disabled)
         --frequency-penalty N    Repeat alpha frequency penalty (default: 0.0, 0.0 = disabled)
         --dry-multiplier N       Set DRY sampling multiplier (default: 0.0, 0.0 = disabled)
         --dry-base N             Set DRY sampling base value (default: 1.75)
         --dry-allowed-length N   Set allowed length for DRY sampling (default: 2)
         --dry-penalty-last-n N   Set DRY penalty for the last n tokens (default: -1, 0 = disable, -1 = context size)
         --dry-sequence-breaker N 
                                  Add sequence breaker for DRY sampling, clearing out default breakers (
                                  ;:;";*) in the process; use "none" to not use any sequence breakers
         --dynatemp-range N       Dynamic temperature range (default: 0.0, 0.0 = disabled)
         --dynatemp-exp N         Dynamic temperature exponent (default: 1.0)
         --mirostat N             Use Mirostat sampling, Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
         --mirostat-lr N          Mirostat learning rate, parameter eta (default: 0.1)
         --mirostat-ent N         Mirostat target entropy, parameter tau (default: 5.0)
  -l     --logit-bias TOKEN_ID(+/-)BIAS
                                  Modifies the likelihood of token appearing in the completion, i.e. "--logit-bias 15043+1" to increase likelihood of token ' Hello', or "--logit-bias 15043-1" to decrease likelihood of token ' Hello'
         --grammar GRAMMAR        BNF-like grammar to constrain generations (see samples in grammars/ dir) (default: '')
         --grammar-file FILE      File to read grammar from
  -j,    --json-schema SCHEMA     JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object. For schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead
         --rope-scaling {none,linear,yarn}
                                  RoPE frequency scaling method, defaults to linear unless specified by the model
         --rope-scale N           RoPE context scaling factor, expands context by a factor of N
         --rope-freq-base N       RoPE base frequency, used by NTK-aware scaling (default: loaded from model)
         --rope-freq-scale N      RoPE frequency scaling factor, expands context by a factor of 1/N
         --yarn-orig-ctx N        YaRN original context size of model (default: 0 = model training context size)
         --yarn-ext-factor N      YaRN extrapolation mix factor (default: -1.0, 0.0 = full interpolation)
         --yarn-attn-factor N     YaRN scale sqrt(t) or attention magnitude (default: 1.0)
         --yarn-beta-fast N       YaRN low correction dim or beta (default: 32.0)
         --yarn-beta-slow N       YaRN high correction dim or alpha (default: 1.0)
  -nkvo, --no-kv-offload          Disable KV offload
         --no-cache-prompt        Disable caching prompt
         --cache-reuse N          Min chunk size to attempt reusing from the cache via KV shifting (default: 0)
  -ctk,  --cache-type-k TYPE      KV cache data type for K, allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1 (default: f16)
  -ctv,  --cache-type-v TYPE      KV cache data type for V, allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1 (default: f16)
  -dt,   --defrag-thold N         KV cache defragmentation threshold (default: 0.1, < 0 - disabled)
  -np,   --parallel N             (Deprecated, use --threads-http instead) Number of parallel sequences to decode (default: 1)
         --mmproj FILE            Path to a multimodal projector file for LLaVA
         --mlock                  Force system to keep model in RAM rather than swapping or compressing
         --no-mmap                Disable memory-map model, slower load but may reduce pageouts if not using mlock
         --mmap                   Enable memory-map model, faster load but may increase pageouts if not using mlock
         --numa TYPE              Attempt optimizations that help on some NUMA systems
                                    - distribute: spread execution evenly over all nodes
                                    - isolate: only spawn threads on CPUs on the node that execution started on
                                    - numactl: use the CPU map provided by numactl
                                  If run without this previously, it is recommended to drop the system page cache before using this, see https://github.com/ggerganov/llama.cpp/issues/1437
         --control-vector FILE    Add a control vector
         --control-vector-scaled FILE SCALE
                                  Add a control vector with user defined scaling SCALE
         --control-vector-layer-range START END
                                  Layer range to apply the control vector(s) to, start and end inclusive
  -sp,   --special                Special tokens output enabled (default: false)
         --enable-reasoning       Enable reasoning (default: true)
         --no-enable-reasoning    Disable reasoning

server/completion/speculative:

         --draft-max, --draft, --draft-n N
                                  Number of tokens to draft for speculative decoding (default: 16)
         --draft-min, --draft-n-min N
                                  Minimum number of draft tokens to use for speculative decoding (default: 0)
         --draft-p-min N          Minimum speculative decoding probability (greedy) (default: 0.8)
  -md,   --model-draft FNAME      Draft model for speculative decoding (default: unused)
  -devd, --device-draft <dev1,dev2,...>
                                  A comma-separated list of devices to use for offloading the draft model (none = don't offload)
                                  Use --list-devices to see a list of available devices
  -ngld, --gpu-layers-draft, --n-gpu-layers-draft N
                                  Number of layers to store in VRAM for the draft model
         --lookup-ngram-min N     Minimum n-gram size for lookup cache (default: 0, 0 = disabled)

server/completion/multimodal:

         --visual-max-image-size N
                                  Maximum image size when completion with vision, resize the image size automatically if exceed, must be larger than 224 and be multiples of 14 (default: 0, 0 = disabled)
         --visual-max-image-cache N
                                  (Deprecated, use --max-projected-cache instead) Specify how many images to cache after encoding, which is used to speed up chat completion (default: 0, 0 = disabled)
         --max-projected-cache N  Specify how many projected embedding cache (default: 0, 0 = disabled)

server/embedding:

         --pooling {none,mean,cls,last,rank}
                                  Pooling type for embeddings, use model default if unspecified
         --attention {causal,non-causal}
                                  Attention type for embeddings, use model default if unspecified

server/images:

         --image-max-batch N      Maximum batch count (default: 4)
         --image-max-height N     Image maximum height, in pixel space, must be larger than 256 and be multiples of 64 (default: 1024)
         --image-max-width N      Image maximum width, in pixel space, must be larger than 256 and be multiples of 64 (default: 1024)
         --image-guidance N       The value of guidance during the computing phase (default: 3.500000)
         --image-strength N       Strength for noising, range of [0.0, 1.0], automatically retrieve the default value according to --model
         --image-sample-method, --image-sampler TYPE
                                  Sample method that will be used for generation, automatically retrieve the default value according to --model, allowed values: euler_a, euler, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd
         --image-sampling-steps, --image-sample-steps N
                                  Number of sampling steps, automatically retrieve the default value according to --model, and +2 when requesting high definition generation
         --image-cfg-scale N      The scale of classifier-free guidance(CFG), automatically retrieve the default value according to --model (1.0 = disabled)
         --image-slg-scale N      The scale of skip-layer guidance(SLG), only for DiT model, automatically retrieve the default value according to --model (0.0 = disabled)
         --image-slg-skip-layer   The layers to skip when processing SLG, may be specified multiple times. (default: 7;8;9)
         --image-slg-start N      The phase to enable SLG (default: 0.01)
         --image-slg-end N        The phase to disable SLG (default: 0.20)
                                  SLG will be enabled at step int([STEP]*[--image-slg-start]) and disabled at int([STEP]*[--image-slg-end])
         --image-schedule-method, --image-schedule TYPE
                                  Denoiser sigma schedule method, allowed values: default, discrete, karras, exponential, ays, gits (default: discrete)
         --image-no-text-encoder-model-offload
                                  Disable text-encoder(clip-l/clip-g/t5xxl) model offload
         --image-clip-l-model PATH
                                  Path to the CLIP Large (clip-l) text encoder, or use --model included
         --image-clip-g-model PATH
                                  Path to the CLIP Generic (clip-g) text encoder, or use --model included
         --image-t5xxl-model PATH 
                                  Path to the Text-to-Text Transfer Transformer (t5xxl) text encoder, or use --model included
         --image-no-vae-model-offload
                                  Disable vae(taesd) model offload
         --image-vae-model PATH   Path to Variational AutoEncoder (vae), or use --model included
         --image-vae-tiling       Indicate to process vae decoder in tiles to reduce memory usage (default: disabled)
         --image-no-vae-tiling    Disable vae decoder in tiles
         --image-taesd-model PATH 
                                  Path to Tiny AutoEncoder For StableDiffusion (taesd), or use --model included
         --image-upscale-model PATH
                                  Path to the upscale model, or use --model included
         --image-upscale-repeats N
                                  How many times to run upscaler (default: 1)
         --image-no-control-net-model-offload
                                  Disable control-net model offload
         --image-control-net-model PATH
                                  Path to the control net model, or use --model included
         --image-control-strength N
                                  How strength to apply the control net (default: 0.900000)
         --image-control-canny    Indicate to apply canny preprocessor (default: disabled)
         --image-free-compute-memory-immediately
                                  Indicate to free compute memory immediately, which allow generating high resolution image (default: disabled)

rpc-server:

         --rpc-server-host HOST   IP address to RPC server listen (default: 0.0.0.0)
         --rpc-server-port PORT   Port to RPC server listen (default: 0, 0 = disabled)
         --rpc-server-main-gpu N  The GPU VRAM to use for the RPC server (default: 0, -1 = disabled, use RAM)
         --rpc-server-reserve-memory MEM
                                  Reserve memory in MiB (default: 0)
         --rpc-server-threads N   Number of threads for the CPU backend (default: according to OS)
         --rpc-server-cache       Enable caching large tensors locally (default: disabled)
         --rpc-server-cache-dir PATH
                                  Path to store large tensors (default: according to OS)

```

## Server API

The available endpoints for the LLaMA Box server mode are:

- **GET** `/health`: Returns the heath check result of the LLaMA Box.

    ```
    RESPONSE : (application/json)
    CASE 1: model is still being loaded
      404
    CASE 2: model is successfully loaded and the server is ready
      {"status": "ok" }
    ```

- **GET** `/metrics`: Returns the Prometheus compatible metrics of the LLaMA Box.
    + This endpoint is only available if the `--metrics` flag is enabled.
    + `llamabox:image_forward_total`: (Counter) Number of image forwarded (steps) in diffusion processing.
    + `llamabox:image_forward_seconds_total`: (Counter) Image forward process time.
    + `llamabox:image_reverse_total`: (Counter) Number of image reversed (steps) in diffusion processing.
    + `llamabox:image_reverse_seconds_total`: (Counter) Image reverse process time.
    + `llamabox:tokens_prefill_total`: (Counter) Number of prompt tokens processed.
    + `llamabox:tokens_prefill_seconds_total`: (Counter) Prompt process time.
    + `llamabox:tokens_decode_total`: (Counter) Number of generation tokens processed.
    + `llamabox:tokens_decode_seconds_total`: (Counter) Predict process time.
    + `llamabox:tokens_drafted_total`: (Counter) Number of speculative decoding tokens processed.
    + `llamabox:tokens_drafted_accepted_total`: (Counter) Number of speculative decoding tokens to be accepted
    + `llamabox:image_forward_steps_per_second`: (Gauge) Average image forwarded diffusion throughput in steps/s.
    + `llamabox:image_reverse_steps_per_second`: (Gauge) Average image reversed diffusion throughput in steps/s.
    + `llamabox:tokens_prefill_per_second`: (Gauge) Average prompt throughput in tokens/s.
    + `llamabox:tokens_decode_per_second`: (Gauge) Average generation throughput in tokens/s.
    + `llamabox:kv_cache_usage_ratio`: (Gauge) KV-cache usage. 1 means 100 percent usage.
    + `llamabox:kv_cache_tokens`: (Gauge) KV-cache tokens.

    ```
    RESPONSE : (text/plain)
    # HELP llamabox:prompt_tokens_total Number of prompt tokens processed.
    ....
    ```

- **POST** `/tokenize`: Convert text to tokens.
    + This is only work to `Text-To-Text` or `Embedding` models.

    ```
    REQUEST : (application/json)
    {
      "content": "",
      "add_special": false,
      "with_pieces": false
    }
    
    RESPONSE : (application/json)
    CASE 1: without pieces
      {
        "tokens": [123, ...]
      }
    CASE 2: with pieces
      {
        "tokens": [
          {"id": 123, "piece": "Hello"},
          ...
        ]
      }
    ```

- **POST** `/detokenize`: Convert tokens to text.
    + This is only work to `Text-To-Text` or `Embedding` models.

    ```
    REQUEST : (application/json)
    {
      "tokens": [123, ...]
    }
    
    RESPONSE : (application/json)
    {
      "content": "..."
    }
    ```

- **GET** `/lora-adapters`: Returns the available LoRA adapters.
    + This is only work to `Text-To-Text`/`Text-To-Image`/`Image-To-Image` models.
    + This endpoint is only available if any LoRA adapter is applied with `--lora` or `--lora-scaled`.

    ```
    RESPONSE : (application/json)
    [
      {
        "id": 0, 
        "path": "...", 
        "init_scale": 1.0 // initial scale, may not be the same as the one used currently
      },
      ...
    ]
    ```

- **POST** `/completion`: Returns the completion of the given prompt.
    + This is only work to `Text-To-Text` models.

- **GET** `/v1/models`: (OpenAI-compatible) Returns the list of available models,
  see https://platform.openai.com/docs/api-reference/models/list.
    + Return the metadata of the model in `meta` field.

- **POST** `/v1/chat/completions` (OpenAI-compatible) Returns the completion of the given prompt,
  see https://platform.openai.com/docs/api-reference/chat/create.
    + This is only work to `Text-To-Text` or `Image-To-Text` models.
    + This endpoint is compatible with [OpenAI Chat Vision API](https://platform.openai.com/docs/guides/vision) when
      enabled `--mmproj` flag,
      see https://huggingface.co/xtuner/llava-phi-3-mini-gguf/tree/main.
    + Allow adjusting the scale of LoRA adapters with `lora` field.

- **POST** `/v1/embeddings`: (OpenAI-compatible) Returns the embeddings of the given prompt,
  see https://platform.openai.com/docs/api-reference/embeddings/create.
    + This is only work to `Text-To-Text` or `Embedding` models.
    + This endpoint is available if the `--embeddings` or `--rerank` flag is enabled.

- **POST** `/v1/completions`: (*LEGACY* OpenAI-compatible) Returns the completion of the given prompt,
  see https://platform.openai.com/docs/api-reference/completions/create.
    + This is only work to `Text-To-Text` models.
    + Allow adjusting the scale of LoRA adapters with `lora` field.

- **POST** `/v1/images/generations`: (OpenAI-compatible) Returns a generated image from the given prompt,
  see https://platform.openai.com/docs/api-reference/images/generations/create.
    + This is only work to `Text-To-Image` models.
    + This endpoint is available if the `--images` flag is enabled.
    + This endpoint supports `stream: true` to return the progressing of the generation.
      ```
      REQUEST : (application/json)
      {
        "n": 1,
        "response_format": "b64_json",
        "size": "512x512",
        "prompt": "A lovely cat",
        "quality": "standard",
        "stream": true,
        "stream_options": {
          "include_usage": true,  // return usage information
          "chunk_result": true,   // split the final image b64_json into chunks to avoid browser caching
          "chunk_size": 4096,     // split the final image b64_json into chunks with the given size, default 4k
          "preview": true,        // enable preview mode
          "preview_faster": true  // enable preview mode (deprecated)
        }
      }
      
      RESPONSE : (text/event-stream)
      data: {"created":1731916353,"data":[{"index":0,"object":"image.chunk","progress":10.0}], ...}
      ...
      data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":50.0}], ...}
      ...
      data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":100.0,"b64_json":"..."}], "usage":{"generation_per_second":...,"time_per_generation_ms":...,"time_to_process_ms":...}, ...}
      data: [DONE]
      ```
    + This endpoint also supports some options
      like [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
      ![https://github.com/AUTOMATIC1111/stable-diffusion-webui](https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/82a973c04367123ae98bd9abdf80d9eda9b910e2/screenshot.png)
      ```
      REQUEST : (application/json)
      {
        "n": 1,
        "response_format": "b64_json",
        "size": "512x512",
        "prompt": "A lovely cat",
        "sample_method": "euler",        // required, alias "sampler", select from euler_a;euler;heun;dpm2;dpm++2s_a;dpm++2m;dpm++2mv2;ipndm;ipndm_v;lcm;ddim_trailing;tcd
        "sampling_steps": 20,            // optional, alias "sample_steps", number of sampling steps
        "schedule_method": "default",    // optional, alias "schedule", select from default;discrete;karras;exponential;ays;gits
        "seed": null,                    // optional, random seed
        "guidance": 3.5,                 // optional, unconditional guidance value
        "cfg_scale": 4.5,                // optional, the scale of classifier-free guidance in the output phase
        "negative_prompt": "",           // optional, negative prompt
        "stream": true,
        "stream_options": {
          "include_usage": true,         // return usage information
          "chunk_result": true,          // split the final image b64_json into chunks to avoid browser caching
          "chunk_size": 4096,            // split the final image b64_json into chunks with the given size, default 4k
          "preview": true,               // enable preview mode
          "preview_faster": true         // enable preview mode (deprecated)
        },
        "lora": [                        // specify LoRA adapters' scale
          {"id": 0, "scale": 1.0},
          ...
         ]
      }
      
      RESPONSE : (text/event-stream)
      data: {"created":1731916353,"data":[{"index":0,"object":"image.chunk","progress":10.0}], ...}
      ...
      data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":50.0}], ...}
      ...
      data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":100.0,"b64_json":"..."}], "usage":{"generation_per_second":...,"time_per_generation_ms":...,"time_to_process_ms":...}, ...}
      data: [DONE]
      ```

- **POST** `/v1/images/edits`: (OpenAI-compatible) Returns an edited image from the given prompt and initial image,
  see https://platform.openai.com/docs/api-reference/images/edits/create.
    + This is only work to `Image-To-Image` models.
    + This endpoint is available if the `--images` flag is enabled.
    + This endpoint supports `stream: true` to return the progressing of the generation.
      ```
      REQUEST: (multipart/form-data)
      n=1
      response_format=b64_json
      size=512x512
      prompt="A lovely cat"
      quality=standard
      image=...                          // required
      mask=...                           // optional
      stream=true
      stream_options_include_usage=true  // return usage information
      stream_options_chunk_result=true   // split the final image b64_json into chunks to avoid browser caching
      stream_options_chunk_size=4096     // split the final image b64_json into chunks with the given size, default 4k
      stream_options_preview=true        // enable preview mode
      stream_options_preview_faster=true // enable preview mode (deprecated)
      
      RESPONSE : (text/event-stream)
      CASE 1: correct input image
        data: {"created":1731916353,"data":[{"index":0,"object":"image.chunk","progress":10.0}], ...}
        ...
        data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":50.0}], ...}
        ...
        data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":100.0,"b64_json":"..."}], "usage":{"generation_per_second":...,"time_per_generation_ms":...,"time_to_process_ms":...}, ...}
        data: [DONE]
      CASE 2: illegal input image
        error: {"code": 400, "message": "Invalid image", "type": "invalid_request_error"}
      ```
    + This endpoint also supports some options
      like [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
      ```
      REQUEST: (multipart/form-data)
      n=1
      response_format=b64_json
      size=512x512
      prompt="A lovely cat"
      image=...                          // required
      mask=...                           // optional
      sample_method=euler                // required, alias "sampler", select from euler_a;euler;heun;dpm2;dpm++2s_a;dpm++2m;dpm++2mv2;ipndm;ipndm_v;lcm;ddim_trailing;tcd
      sampling_steps=20                  // optional, alias "sample_steps", number of sampling steps
      schedule_method=default            // optional, alias "schedule", select from default;discrete;karras;exponential;ays;gits
      seed=null                          // optional, random seed
      guidance=3.5                       // optional, unconditional guidance value
      strength=0.75                      // optional, the strength of noising/unnoising
      cfg_scale=4.5                      // optional, the scale of classifier-free guidance in the output phase
      negative_prompt=""                 // optional, negative prompt
      stream=true
      stream_options_include_usage=true  // return usage information
      stream_options_chunk_result=true   // split the final image b64_json into chunks to avoid browser caching
      stream_options_chunk_size=4096     // split the final image b64_json into chunks with the given size, default 4k
      stream_options_preview=true        // enable preview mode
      stream_options_preview_faster=true // enable preview mode (deprecated)
      lora="[{\"id\":0,\"scale\":1.0}]"  // specify LoRA adapters' scale

      RESPONSE : (text/event-stream)
      CASE 1: correct input image
        data: {"created":1731916353,"data":[{"index":0,"object":"image.chunk","progress":10.0}], ...}
        ...
        data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":50.0}], ...}
        ...
        data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":100.0,"b64_json":"..."}], "usage":{"generation_per_second":...,"time_per_generation_ms":...,"time_to_process_ms":...}, ...}
        data: [DONE]
      CASE 2: illegal input image
        error: {"code": 400, "message": "Invalid image", "type": "invalid_request_error"}
      ```

- **POST** `/v1/rerank`: Returns the completion of the given prompt via lookup cache.
    + This is only work to `Reranker` models, like [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3).
    + This endpoint is only available if the `--rerank` flag is provided.
    + This is unavailable for the GGUF files created
      before [llama.cpp#pr9510](https://github.com/ggml-org/llama.cpp/pull/9510).
    + This endpoint supports `normalize: false` to return the original `relevance_score` score.
      ```
      REQUEST: (application/json)
      {
        "model": "...",
        "query": "...",
        "documents": [
          "..."
        ],
        "normalize": false
      }
      ```

## Tools

It was so hard to find a Chat UI that was directly compatible with OpenAI,
that mean, no installation required (we can live with `docker run`),
no tokens (or optional), no [Ollama](https://github.com/ollama/ollama) required, just a simple RESTful API.

So we are inspired by
the [llama.cpp/chat.sh](https://github.com/ggml-org/llama.cpp/blob/e6f291d15844398f8326940fe5ad7f2e02b5aa56/examples/server/chat.sh)
and adjust it to interact with LLaMA Box.

All you need is a Bash shell, curl and jq.

- **chat.sh**: A simple script to interact with the `/v1/chat/completions` endpoint.
- **image_generate.sh**: Script to interact with the `/v1/images/generations` endpoint.
- **image_edit.sh**: Script to interact with the `/v1/images/edits` endpoint.

```shell
$ # one-shot chat
$ MAX_TOKENS=4096 ./llama-box/tools/chat.sh "Tell me a joke"

$ # interactive chat
$ MAX_TOKENS=4096 ./llama-box/tools/chat.sh

$ # one-shot image generation
$ ./llama-box/tools/image_generate.sh "A lovely cat"

$ # interactive image generation
$ ./llama-box/tools/image_generate.sh

$ # one-shot image editing
$ IMAGE=/path/to/image.png ./llama-box/tools/image_edit.sh "A lovely cat"

$ # interactive image editing
$ IMAGE=/path/to/image.png ./llama-box/tools/image_generate.sh
```

And we also provide a tool for batch testing with the LLaMA Box.

```shell
$ # warm up once then test within (1 2 4 8 4 2 1) parallel requests used preset prompts
$ ./llama-box/tools/batch_chat.sh

$ # testing with 10 parallel requests used preset prompts
$ ./llama-box/tools/batch_chat.sh 10

$ # testing with 10 parallel requests used a random prompt owned 1000 characters
$ ./llama-box/tools/batch_chat.sh 10 RANDOM_1000
```

## License

MIT
