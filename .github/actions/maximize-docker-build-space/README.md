# Maximize Docker Build Space

This action maximizes the space available to Docker building. It inspired by [easimon/maximize-build-space](https://github.com/easimon/maximize-build-space).

## Usage

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Maximize Docker Build Space
        uses: gpustack/llama-box/.github/actions/maximize-docker-build-space@main
```

## Inputs

```yaml
inputs:
  root-reserve-mb:
    description: 'Space to be left free on the root filesystem, in Megabytes.'
    required: false
    default: '2048'
  temp-reserve-mb:
    description: 'Space to be left free on the temp filesystem (/mnt), in Megabytes.'
    required: false
    default: '2048'
  swap-size-mb:
    description: 'Swap space to create, in Megabytes.'
    required: false
    default: '2048'
  deep-clean:
    description: 'Perform a deep clean, removing unnecessary packages.'
    required: false
    default: 'false'
```
