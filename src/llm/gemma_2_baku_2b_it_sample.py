
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

model_name = "rinna/gemma-2-baku-2b-it"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
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

