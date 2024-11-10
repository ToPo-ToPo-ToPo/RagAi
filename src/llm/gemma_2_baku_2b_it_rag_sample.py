import platform
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig # 量子化によるコスト低減
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

model_name = "rinna/gemma-2-baku-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)

system_name = platform.system()
if system_name == 'Windows':
    # 量子化によるコスト低減の設定
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    # モデルの設定
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config 
    )
    
elif system_name == 'Darwin':
	model = AutoModelForCausalLM.from_pretrained(
        model_name,
    )

pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
)
local_llm = HuggingFacePipeline(pipeline=pipe)


# Prompt
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser

# プロンプトを準備
template = """
<bos><start_of_turn>user
{query}
<end_of_turn><start_of_turn>model
"""
prompt = PromptTemplate.from_template(template)

# チェーンを準備
chain = (
    prompt
    | local_llm
)

# 推論を実行
query = "なごや個人開発者の集いとは何ですか。"
answer = chain.invoke({'query':query})
print(answer)

#-----------------------------------------
# VectorDB構築
#-----------------------------------------
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

urls = [
    'https://758indies.connpass.com/',
    ...
]

#
#docs = langchain_community.document_loaders.UnstructuredURLLoader(urls=urls).load()

# Seleniumを使う場合
loader = SeleniumURLLoader(
    urls=urls
)
docs = loader.load()

# チャンクの分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # チャンクの最大文字数
    chunk_overlap=10,  # オーバーラップの最大文字数
)
docs = text_splitter.split_documents(docs)

import langchain.vectorstores
import langchain.embeddings

# ベクトル化する準備
embedding = langchain.embeddings.HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

# 読込した内容を保存
vectorstore = langchain.vectorstores.Chroma.from_documents(
    documents=docs,
    embedding=embedding
)

# 検索する文章
query = "なごや個人開発者の集いとは何ですか。"

# 検索する
docs = vectorstore.similarity_search(query=query, k=5)
# 検索結果の一覧を表示
#for index, doc in enumerate(docs):
#    print("%d:" % (index + 1))
#    print(doc.page_content)

#------------------------------------------------------------
# RAGを使って生成をする
#------------------------------------------------------------
# プロンプトを準備
template = """
<bos><start_of_turn>system
次の文脈を使用して、最後の質問に答えてください。
{context}
<end_of_turn><start_of_turn>user
{query}
<end_of_turn><start_of_turn>model
"""
prompt = PromptTemplate.from_template(template)

# チェーンを準備
chain = (
    prompt
    | local_llm
)

# 質問を入力
query = "なごや個人開発者の集いとは何ですか。"

# 検索して関連する文脈を作成
docs = vectorstore.similarity_search(query=query, k=5)
content = "\n".join([f"Content:\n{doc.page_content}" for doc in docs])

# 推論を実行
answer = chain.invoke({'query':query, 'context':content})
print(answer)