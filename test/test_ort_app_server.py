import requests
import sseclient
import json
import argparse
import sys

def get_args() -> argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_stream", action="store_true")
    parser.add_argument("--max_length", type=int, default=1024)
    return parser.parse_args()

def main() -> int:
    args = get_args()
    url = "http://localhost:8080/v1/chat/completions"
    chat_template = '< |user|>\\n{input} <|end|>\\n<|assistant|>'
    session = requests.Session()
    while True:
        text = input("Prompt: ")
        if not text:
            print("Error, input cannot be empty")
            continue
        prompt = f'{chat_template.format(input=text)}'
        #print("prompt: ", prompt)
        headers = {'Accept': 'text/event-stream'}
        req_body = '{"max_length": %d, "stream": %s, "messages":[{"content":"%s","role":"user"}]}' % \
            (args.max_length, "true" if args.do_stream else "false", prompt)
        #print("Sending...", req_body)
        j = json.loads(req_body)
        print("Sending...", j)
        response = session.post(url, headers=headers, json=j)
        if not response.ok:
            print("Error: ", response.content)
            continue
        if args.do_stream:
            client = sseclient.SSEClient(response)
            for event in client.events():
                print(json.loads(event.data)["choices"][0]["delta"]["content"], end='')
            print("\n")
        else:
            #print(response.content)
            print(json.loads(response.content)["choices"][0]["message"]["content"])

    return 0

if __name__ == "__main__":
    sys.exit(main())
