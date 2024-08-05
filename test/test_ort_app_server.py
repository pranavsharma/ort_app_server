# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import requests
import sseclient
import json
import argparse
import sys
base_url = "http://localhost:8080/v1"
current_model_id = ""

def get_args() -> argparse:
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def print_help_msg():
    print("Available Commands:\n"
        "/?,/help         Print this message\n"
        "/bye,/quit       Quit\n"
        "/pull <model id> Pull a model from the registry\n"
        "/load <model id> Load a model in the server\n"
        "/list            List models in the registry\n"
        "/ps              List currently loaded models\n")

def pull_model(model_id):
    print("Pulling model (this might take a while): ", model_id)
    req_body = '{"model": "%s"}' % (model_id)
    j = json.loads(req_body)
    res = requests.post(base_url + "/pull", json=j)
    if (res.ok):
        print("Pulled model successfully.")
    else:
        print("Error pulling model ", res.content)

def load_model(model_id):
    print("Loading model (this might take a while): ", model_id)
    req_body = '{"model": "%s"}' % (model_id)
    j = json.loads(req_body)
    res = requests.post(base_url + "/load", json=j)
    if (res.ok):
        print("Loaded model successfully.")
        global current_model_id
        current_model_id = model_id
        print("Will use ", model_id, " for all subsequent chats")
    else:
        print("Error loading model ", res.content)

def get_loaded_models(do_print=True):
    res = requests.get(base_url + "/ps")
    model_list = []
    if (res.ok):
        j = json.loads(res.content)
        if (len(j["models"]) == 0):
            print("No models were loaded.")
            return model_list
        for model in j["models"]:
            model_list.append(model.strip())
        if do_print:
            print("List of models loaded")
            for model in model_list:
                print(model, " ")
        return model_list
    else:
        if do_print:
            print("Error getting list of loaded models: ", res.content)
        return model_list
    
def list_models():
    res = requests.get(base_url + "/models")
    if (res.ok):
        j = json.loads(res.content)
        if (len(j["models"]) == 0):
            print("No models in registry.")
            return
        for model in j["models"]:
            print(model, " ")
    else:
        print("Error listing models: ", res.content)

def check_server_health():
    print("Checking server health...")
    try:
        res = requests.get(base_url + "/health")
        if (not res.ok):
            print("Server is not healthy, got status code: ", res.status_code)
            exit(1)
    except requests.exceptions.RequestException as e:
        print(e)
        exit(1)
    print("All good.")

def exit_program():
    sys.exit(0)

def main() -> int:
    check_server_health()
    args = get_args()
    chat_template = '< |user|>\\n{input} <|end|>\\n<|assistant|>'
    cmd_handlers = {"/?": print_help_msg,
                    "/bye": exit_program,
                    "/quit": exit_program,
                    "/help": print_help_msg,
                    "/pull": pull_model,
                    "/load": load_model,
                    "/ps": get_loaded_models,
                    "/list": list_models}
    headers = {'Accept': 'text/event-stream'}
    while True:
        user_input = input("\n>>> Send a message (/? for help)\n")
        if not user_input:
            print("Error, input cannot be empty")
            continue
        if user_input.startswith("/"):
            tokens = user_input.split(' ', 1)
            cmd_key = tokens[0]
            if cmd_key not in cmd_handlers:
                print("Invalid cmd. Please run /? or /help\n")
                continue
            cmd_to_run = cmd_handlers[cmd_key]
            if len(tokens) > 1:
                cmd_args = tokens[1]
                cmd_to_run(cmd_args)
            else: cmd_to_run()
            continue
        prompt = f'{chat_template.format(input=user_input)}'

        # Determine which model id to use. If current_model_id is populate, use that
        # else get the list of models loaded in the server. If more than one model is loaded
        # ask the user to load a model, else use the one model that has been loaded
        # Get the current list of models loaded in the server
        global current_model_id
        if (not current_model_id):
            model_list = get_loaded_models(False)
            if (len(model_list) == 1):
                current_model_id = model_list[0]
            else:
                print("Load a model first using the /load command. You can get a list of models by running /list")
                continue
        
        # now construct the request
        print("Using model_id: [", current_model_id, "]")
        req_body = '{"model": "%s", "stream": true, "messages":[{"content":"%s", "role":"user"}]}' % (current_model_id, prompt)
        j = json.loads(req_body)
        try:
            response = requests.post(base_url + "/chat/completions", headers=headers, json=j, stream=True)
        except requests.exceptions.RequestException as e:
            print(e)
            continue
        if not response.ok:
            print("Error: ", response.content)
            continue
        client = sseclient.SSEClient(response)
        for event in client.events():
            #print(event.data)
            print(json.loads(event.data)["choices"][0]["delta"]["content"], end='')
            sys.stdout.flush()
        print(end = "\n")
    return 

if __name__ == "__main__":
    sys.exit(main())
