
# モデルの定義関係
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig # 量子化によるコスト低減
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
# Promptの定義関係
from langchain.prompts import PromptTemplate
#
from langchain.schema import StrOutputParser
#------------------------------------------------------------------------------------------------
# gemma-2-baku-2b-itのモデルを管理するクラス
#------------------------------------------------------------------------------------------------
class Gemma2Baku2bIt:
    
    #----------------------------------------------------------------------
    # コンストラクタ
    #----------------------------------------------------------------------
    def __init__(self, device_map, dtype, load_in_8bit=False, load_in_4bit=False) -> None:
        
        # モデル名の設定
        model_name = "rinna/gemma-2-baku-2b-it"
        
        # トークナイザーの設定
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 量子化によるコスト低減の設定
        #quantization_config = BitsAndBytesConfig(
        #    load_in_8bit=load_in_8bit,
        #    load_in_4bit=load_in_4bit
        #)

        # モデルの設定
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
            attn_implementation="eager",
            #quantization_config=quantization_config
        )

        # パイプラインの作成
        pipe = pipeline(
            task='text-generation',
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=2048,
        )
        
        # モデルを生成
        self.llm = HuggingFacePipeline(pipeline=pipe)
    
    #----------------------------------------------------------------------
    # プロンプトの定義
    #----------------------------------------------------------------------
    def generate_prompt(self):
        
        #
        question_prompt_template_format = self.tokenizer.apply_chat_template(
            conversation = [
                {"role": "user", "content": "{query}"}
            ], 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # プロンプトを定義
        prompt = PromptTemplate(
            template=question_prompt_template_format,
            input_variables=["query"]
        )

        #
        return prompt
    
    #----------------------------------------------------------------------
    # RAG用のプロンプトの定義
    #----------------------------------------------------------------------
    def generate_prompt_with_rag(self):

        #
        question_prompt_template_format = self.tokenizer.apply_chat_template(
            conversation = [
                {"role": "user", "content": "次の文脈を使用して、最後の質問に答えてください。\n{context}"},
                {"role": "model", "content": "わかりました。"},
                {"role": "user", "content": "{query}"}
            ], 
            tokenize=False, 
            add_generation_prompt=True
        )

        # プロンプトを定義
        prompt = PromptTemplate(
            template=question_prompt_template_format,
            input_variables=["context", "query"]
        )
        
        #
        return prompt
    
    #----------------------------------------------------------------------
    # 会話用のChainを作成
    #----------------------------------------------------------------------
    def make_chain(self, prompt):
        
        #
        chain = prompt | self.llm | StrOutputParser()

        #
        return chain
    
    #----------------------------------------------------------------------
    # 入力された質問に対する回答を生成
    #----------------------------------------------------------------------
    def response(self, chain, query):

        # 推論を実行
        answer = chain.invoke({'query':query})
        return answer
    
    #----------------------------------------------------------------------
    # 入力された質問に対する回答を生成
    #----------------------------------------------------------------------
    def response_with_rag(self, chain, vector_db, query):

        # 検索して関連する文脈を作成
        docs = vector_db.similarity_search(query=query, k=5)
        content = "\n".join([f"Content:\n{doc.page_content}" for doc in docs])

        # 推論を実行
        answer = chain.invoke({'query':query, 'context':content})
        return answer


