import torch
from IPython.lib.display import Audio
import numpy as np
import ChatTTS
from ChatTTS.infer.api import refine_text, infer_code
import torchaudio

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

class TTSManager:
    def __init__(self):
        self.model_path = 'C:/devFiles/projects/chatTTS/ChatTTS'
        self.chat = ChatTTS.Chat()
        self.chat.load_models(source="local", local_path=self.model_path, compile=False)


    def generate_audio(self, text, temperature = 0.8, top_P = 0.7, top_K = 20,
                       audio_seed_input = 42, text_seed_input = 42, refine_text_flag = True):
        torch.manual_seed(audio_seed_input)
        rand_spk = torch.randn(768)
        params_infer_code = {
            'spk_emb': rand_spk,
            'temperature': temperature,
            'top_P': top_P,
            'top_K': top_K,
        }
        params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}

        torch.manual_seed(text_seed_input)

        text_tokens = refine_text(self.chat.pretrain_models, text, **params_refine_text)['ids']
        # print("text_tokens:")
        # print(type(text_tokens))
        # print(text_tokens)
        # print("===========")

        text_tokens = [i[i < self.chat.pretrain_models['tokenizer'].convert_tokens_to_ids('[break_0]')] for i in text_tokens]
        # print("text_tokens:")
        # print(type(text_tokens))
        # print(text_tokens)
        # print("===========")

        text = self.chat.pretrain_models['tokenizer'].batch_decode(text_tokens)
        # result = infer_code(chat.pretrain_models, text, **params_infer_code, return_hidden=True)

        print(f'ChatTTS微调文本：{text}')

        wav = self.chat.infer(text,
                         params_refine_text=params_refine_text,
                         params_infer_code=params_infer_code,
                         use_decoder=True,
                         skip_refine_text=True,
                         )

        # 保存文件，返回文件名
        # torchaudio.save("output/output-01.wav", torch.from_numpy(wav[0]), 24000)
        torchaudio.save("C:/devFiles/projects/chatTTS/output/output-01.wav", torch.from_numpy(wav[0]), 24000)
        return "output-01.wav"

        # audio_data = np.array(wav[0]).flatten()
        # sample_rate = 24000
        # text_data = text[0] if isinstance(text, list) else text
        #
        # return [(sample_rate, audio_data), text_data]