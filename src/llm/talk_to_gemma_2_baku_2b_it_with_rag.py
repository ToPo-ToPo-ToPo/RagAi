
import platform
import torch
from gemma_2_baku_2b_it import Gemma2Baku2bIt
#
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
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

    #--------------------------------------------------------------------------
    # 通常の生成
    #--------------------------------------------------------------------------
    # プロンプトを作成
    prompt = llm.generate_prompt()

    # Chainを作成
    chain = llm.make_chain(prompt=prompt)

    # 回答を生成
    answer = llm.response(chain=chain, query="応力制約の方法について説明してください。")
    print(answer)


    #-------------------------------------------------------------------------
    # RAGを使った生成
    #-------------------------------------------------------------------------
    print("\nWith RAG")

    # プロンプトを作成
    prompt_rag = llm.generate_prompt_with_rag()

    # Chainを作成
    chain_rag = llm.make_chain(prompt=prompt_rag)

    # インデックスのパス
    index_path = "../../database/topology_optimization_database"

    # 埋め込みモデルの読み込み
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )

    # インデックスの読み込み
    index = FAISS.load_local(
        folder_path=index_path, 
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    # 回答を生成
    query = "応力制約の方法について説明してください。"
    answer_rag = llm.response_with_rag(chain=chain_rag, vector_db=index, query=query)
    print(answer_rag)