
import os
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

#--------------------------------------------------------
# データの加工
#--------------------------------------------------------
# 資料のファイル名
file_name = "topology_sample.txt"

# 元データの保存場所とファイルパスの作成
input_dir = "../../database/text_data/"
input_file_path = os.path.join(input_dir, file_name)

# 資料の格納場所（ディレクトリ）とファイルパスの作成
data_dir = "../../database/data/"
file_path = os.path.join(data_dir, file_name)

# データの読み込み
with open(input_file_path, "r",encoding="utf-8") as f:
    data = f.read()

# 改行を削除
update_data = data.replace('\n', '')
print(update_data)

#
with open(file_path, 'w',encoding="utf-8") as f:
  f.write(update_data)


#--------------------------------------------------------
# データベースの作成
#--------------------------------------------------------
# ベクトル化したインデックスの保存場所（ディレクトリ）
index_path = "../../database/topology_optimization_database"

# ディレクトリの読み込み
loader = DirectoryLoader(data_dir)

# 埋め込みモデルの読み込み
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

# テキストをチャンクに分割
split_texts = loader.load_and_split(
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=200,# 分割したチャンクごとの文字数
        chunk_overlap=50 # チャンク間で被らせる文字数
    )
)

# インデックスの作成
index = FAISS.from_documents(
    documents=split_texts,
    embedding=embedding_model,
)

# インデックスの保存
index.save_local(
    folder_path=index_path
)