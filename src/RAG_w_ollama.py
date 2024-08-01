from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

#llm = ChatOllama(model='gemma2')
# 모델 파라미터 설정
params = {
    'temperature' : 0.7,
    'max_tokens' : 100,
}

kwargs = {
    'frequency_penalty' : 0.5,
    'presence_penalty' : 0.5,
    'stop': ['\n']
}

llm = ChatOllama(model='gemma2')

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, professional assistant named saemiYangBot. Introduce yourself first, and answer the questions. answer me in Korean no matter what. "),
    ("user", "{input}")
])


#chain.invoke({"input": "What is stock?"})

#print(chain.invoke({"input": "What is stock?"}))
output_parser = StrOutputParser()

chain = prompt | llm | output_parser
for token in chain.stream(
    {"input": "What is stock?"}
):
    print(token, end="")