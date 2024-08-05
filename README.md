# ORT App Server (experimental)
* This is a _working_ prototype and under active development.

### What does this do?
   * Load a model from the disk by supplying the path on the cmd line (--model option)
   * Pull (download) a model from hugging face or local disk
      * ```curl http://localhost:8080/v1/pull -d '{"model": "model_3"}'```
   * Load a model in memory
      * ```curl http://localhost:8080/v1/load -d '{"model": "model_3"}'```
   * Chat with a model in streaming mode
      * ```python test/test_ort_app_server.py```
   * List currently loaded models
      * ```curl http://localhost:8080/v1/ps```
   * List models in registry
      * ```curl http://localhost:8080/v1/models```      
   * OpenAI compatible API

### Other notable things
   * A single server process that both manages and serves models.
   * Can add new model sources and downloaders easily by writing just one function. See [this](src/model_downloader.h).
   * A separate [Model Manager](src/model_manager.h) module that can be integrated in the GenAI lib.
   * Recognizes that models were downloaded before and doesn't download them again.
   * Supports multiple models.

## Build and Install
### Important Issues
   * genai lib requires a fix in ort_genai.h to make a few functions 'inline' for the server to compile. For now this file
     has been checked into this repo.
      * SetLogBool, SetLogString, SetCurrentGpuDeviceId, GetCurrentGpuDeviceId
   * Openssl 3.x or later is required

First install the latest onnxruntime-genai pkg here https://github.com/microsoft/onnxruntime-genai/releases/tag/v0.3.0
*
```
mkdir build
cd build
For release build: cmake -DORT_GENAI_DIR=<path of genai pkg installation> ..
For debug build: cmake -DORT_GENAI_DIR=<path of genai pkg installation> -DCMAKE_BUILD_TYPE=Debug ..
```

## Usage
You can use the server in the following two ways.

```
pranav@pranav-HP-Z440-Workstation:~/work_projects/ort_app_server/build$ LD_LIBRARY_PATH=~/work_projects/ort_app_server.local/onnxruntime-genai-0.3.0-linux-x64/lib ./ort_app_server -v -f ../test/model_manifest.json
ORT App Server
Usage: ./ort_app_server [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -v,--verbose
  -n,--hostname TEXT          Hostname to listen on (default: localhost)
  -p,--port INT               Port number to listen on (default: 8080)
  -t,--nthreads INT           Numbter of threads to use
  -i,--model_id TEXT          Model id (required if --model is used. Model id is used to identify the model in the server.)
  -m,--model TEXT             Model folder containing the model to load
  -f,--model_manifest_file TEXT
                              Model manifest file
  -d,--downloaded_models_path TEXT
                              Folder where models are downloaded (default: /tmp/ort_app_server/models/).
```

### Use a model from the cmd line
1. In one terminal run the server as below
```
pranav@pranav-HP-Z440-Workstation:~/work_projects/ort_app_server/build$ LD_LIBRARY_PATH=~/work_projects/ort_app_server.local/onnxruntime-genai-0.3.0-linux-x64/lib ./ort_app_server -v -m ~/work_projects/ort_app_server.local/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/ --model_id foo
[2024-08-05 15:40:13.505] [info] Loading info for models that were downloaded before
[2024-08-05 15:40:13.505] [info] Loaded info for [3] models
[2024-08-05 15:40:13.505] [info] Loading model from the cmd line [/home/pranav/work_projects/ort_app_server.local/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/]
[2024-08-05 15:40:16.102] [info] Model [/home/pranav/work_projects/ort_app_server.local/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/] loaded successfully
```
2. In another terminal run ```python test/test_ort_app_server.py``` and start sending messages.

### Pull models from remote sources (like HF)
1. Hydrade the manifest file with the models. See [test manifest file](test/model_manifest.json) for an example.
A manifest file is the same as ONNX model hub/registry.
1. In one terminal, run the server as below
```
pranav@pranav-HP-Z440-Workstation:~/work_projects/ort_app_server/build$ LD_LIBRARY_PATH=~/work_projects/ort_app_server.local/onnxruntime-genai-0.3.0-linux-x64/lib ./ort_app_server -v -f ../test/model_manifest.json
[2024-08-05 15:37:35.640] [info] Loading info for models that were downloaded before
[2024-08-05 15:37:35.640] [info] Loaded info for [3] models
[2024-08-05 15:37:35.640] [info] Reading manifest file [../test/model_manifest.json]
[2024-08-05 15:37:35.640] [info] Read manifest for [3] models
```
1. In another terminal run ```python test/test_ort_app_server.py```, 
   load the model of your choice by using ```/load``` command and start sending messages.
Here's a sample run:
```
pranav@pranav-HP-Z440-Workstation:~/work_projects/ort_app_server$ python test/test_ort_app_server.py
Checking server health...
All good.

>>> Send a message (/? for help)
/?
Available Commands:
/?,/help         Print this message
/bye,/quit       Quit
/pull <model id> Pull a model from the registry
/load <model id> Load a model in the server
/list            List models in the registry
/ps              List currently loaded models


>>> Send a message (/? for help)

```

### TODO
   * Clean up code
   * Support EPs other than CPU
   * Cross-platform
   * Multi-modal APIs
   * More OpenAI APIs (like embedding, tokenization)
   * Additional parameters in chat completion API
   * Write a better client application
   * Handle concurrency concerns
   * Unload model API
   * Streaming updates when model files are downloaded
   * Security (authentication, etc.)
   * Intelligent eviction of models based on resource constraints
   * Packaging

## Contribute
   * See [ORT coding guidelines](https://github.com/microsoft/onnxruntime/blob/main/docs/Coding_Conventions_and_Standards.md)

