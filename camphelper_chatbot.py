# from openai import OpenAI
#     client = OpenAI(api_key="")

#     completion = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": ""},
#         {"role": "user", "content": "클라우드 설명해줘."}
#         ]
#     )

#     print(completion.choices[0].message)

#


# import os
# import sys

# from langchain_openai import ChatOpenAI
# from langchain_community.document_loaders import TextLoader
# from langchain.indexes import VectorstoreIndexCreator

# os.environ["OPENAI_API_KEY"] = ""
# query = sys.argv[1]
# loader = TextLoader('data.txt' )
# index = VectorstoreIndexCreator().from_loaders([loader])
# print(index.query(query, llm=ChatOpenAI()))


from openai import OpenAI
import os
import json

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


# Example OpenAI Python library request
MODEL = "gpt-4o"
# response = client.chat.completions.create(
#     model=MODEL,
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Knock knock."},
#         {"role": "assistant", "content": "Who's there?"},
#         {"role": "user", "content": "Orange."},
#     ],
#     temperature=0,
# )

# print(json.dumps(json.loads(response.model_dump_json()), indent=4))

# response.choices[0].message.content

# response = client.chat.completions.create(
#     model=MODEL,
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Explain asynchronous programming in the style of the pirate Blackbeard."},
#     ],
#     temperature=0,
# )

# print(response.choices[0].message.content)


response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": ""},
        {"role": "user", "content": "보디빌딩 중 비키니 규정 포즈에 대해서 설명해줘"},
    ],
    temperature=0,
)

print(response.choices[0].message.content)