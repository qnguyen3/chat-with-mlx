from openai import OpenAI
openai_api_base = "http://127.0.0.1:8080/v1"
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string", 
                        "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },   
    }
] 

client = OpenAI(api_key='EMPTY',base_url=openai_api_base)

def get_completion(messages, model="mlx-community/stablelm-2-zephyr-1_6b", temperature=0, max_tokens=300, tools=None):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools
    )
    return response.choices[0].message

messages = [
    {
        "role": "user",
        "content": "What is the weather like in London?"
    }
]
response = get_completion(messages, tools=tools)
print(response)
# student_1_description = "David Nguyen is a sophomore majoring in computer science at Stanford University. He is Asian American and has a 3.8 GPA. David is known for his programming skills and is an active member of the university's Robotics Club. He hopes to pursue a career in artificial intelligence after graduating."
# student_2_description="Ravi Patel is a sophomore majoring in computer science at the University of Michigan. He is South Asian Indian American and has a 3.7 GPA. Ravi is an active member of the university's Chess Club and the South Asian Student Association. He hopes to pursue a career in software engineering after graduating."

# # response = client.chat.completions.create(model='mlx-community/stablelm-2-zephyr-1_6b',
# #   messages=[
# #     {"role": "user", "content": "Who won the world series in 2020?"},
# #   ],
# # )
