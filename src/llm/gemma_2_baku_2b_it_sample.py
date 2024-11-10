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

base_template = '''
<bos><start_of_turn>user
{question}
<end_of_turn><start_of_turn>model
'''

qa_chain = (
    {'question': RunnablePassthrough()}
    | PromptTemplate.from_template(base_template)
    | local_llm
    | StrOutputParser()
)
res = qa_chain.invoke('日本で一番高い山の名前と標高は何ですか？')
print(res)

