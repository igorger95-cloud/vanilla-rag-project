# vLLM installation in Google Colab

Рабочая конфигурация:

- GPU: Tesla T4
- Python 3.13
- PyTorch 2.13.0+cu129
- CUDA 12.9
- vLLM 0.28.0+cu129

## Установка vLLM

Использовался wheel под CUDA 12.9:

pip install -U "https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" --extra-index-url https://download.pytorch.org/whl/cu129

После установки необходимо перезапустить runtime.

## TorchAudio

Если возникает конфликт версий CUDA, использовалась совместимая версия:

pip install --force-reinstall torchaudio==2.11.0+cu129 --index-url https://download.pytorch.org/whl/cu129

После изменения PyTorch, TorchAudio или vLLM рекомендуется перезапустить runtime.