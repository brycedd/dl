from openai import OpenAI

client = OpenAI(
    api_key='',
)

response = client.chat.completions.create(
    messages=[
        {'role': 'user', 'content': '1+1'},
    ],
    model='gpt-3.5-turbo',
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="", flush=True)