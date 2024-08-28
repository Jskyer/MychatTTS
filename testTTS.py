import torch
from IPython.lib.display import Audio

import ChatTTS
import torchaudio

# torch._dynamo.config.cache_size_limit = 64
# torch._dynamo.config.suppress_errors = True
# torch.set_float32_matmul_precision('high')

# 查看可用后端
torchaudio.list_audio_backends()


# MODEL_PATH = 'C:/devFiles/projects/chatTTS/ChatTTS'
#
# chat = ChatTTS.Chat()
# chat.load_models(source="local", local_path=MODEL_PATH, compile=False)

# chat.load_models(
#     vocos_config_path=f'{MODEL_PATH}/config/vocos.yaml',
#     vocos_ckpt_path=f'{MODEL_PATH}/asset/Vocos.pt',
#     gpt_config_path=f'{MODEL_PATH}/config/gpt.yaml',
#     gpt_ckpt_path=f'{MODEL_PATH}/asset/GPT.pt',
#     decoder_config_path=f'{MODEL_PATH}/config/decoder.yaml',
#     decoder_ckpt_path=f'{MODEL_PATH}/asset/Decoder.pt',
#     tokenizer_path=f'{MODEL_PATH}/asset/tokenizer.pt',
# )

# texts = ["你好，欢迎使用ChatTTS"]
# wavs = chat.infer(texts, use_decoder=True)
#
# # torchaudio.save("./output/output-01.wav", torch.from_numpy(wavs[0]), 24000)
# Audio(wavs[0], rate=24_000, autoplay=True)