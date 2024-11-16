
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
    print(prompt)

    # Chainを作成
    chain = llm.make_chain(prompt=prompt)

    # 回答を生成
    answer = llm.response(chain=chain, query="あなたは、人気配信者のニャン太です。構造最適化手法の1つであるトポロジー最適化の代表的な手法とは？")
    print(answer)


    print("\nWith RAG")

    # プロンプトを作成
    prompt_rag = llm.generate_prompt_with_rag()

    # Chainを作成
    chain_rag = llm.make_chain(prompt=prompt_rag)

    # 回答を生成
    context = "トポロジー最適化の代表的な手法は均質化法と密度法とレベルセット法です。特に最も論文数が多いのは密度法です。"
    query = "あなたは、人気配信者のニャン太です。構造最適化手法の1つであるトポロジー最適化の代表的な手法とは？"
    answer_rag = llm.response_with_rag(chain=chain_rag, content=context, query=query)
    print(answer_rag)