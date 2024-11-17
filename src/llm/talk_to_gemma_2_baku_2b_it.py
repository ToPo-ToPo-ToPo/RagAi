
import platform
import torch
from gemma_2_baku_2b_it import Gemma2Baku2bIt
#======================================================================
# メインプログラム
#======================================================================
if __name__ == '__main__':

    # 条件設定
    system_name = platform.system()
    if system_name == 'Windows':
        device_map = "auto"
        #dtype = torch.float32
        dtype = torch.bfloat16
        
    elif system_name == 'Darwin':
        device_map = "mps"
        #dtype = torch.float32
        dtype = torch.bfloat16

    # モデルを生成
    llm = Gemma2Baku2bIt(
        device_map=device_map,
        dtype=dtype
    )

    # プロンプトを作成
    prompt = llm.generate_prompt()

    # Chainを作成
    chain = llm.make_chain(prompt=prompt)

    # 回答を生成
    answer = llm.response(chain=chain, query="あなたは、人気配信者のニャン太です。構造最適化手法の1つであるトポロジー最適化の代表的な手法とは？")
    print(answer)