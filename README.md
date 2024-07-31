# ORT App Server (experimental)
* This is a _working_ prototype and under active development.

### What does this do?
   * Pull (download) a model from hugging face or local disk
      * ```curl http://localhost:8080/v1/pull -d '{"model": "model_3"}'```
   * Load a model in memory
      * ```curl http://localhost:8080/v1/load -d '{"model": "model_3"}'```
   * Chat with a model in both streaming and non-streaming mode
      * ```python test/test_ort_app_server.py --model model_3 --do_stream```
   * List currently loaded models
      * ```curl http://localhost:8080/v1/models```
   * OpenAI compatible API

### Other notable things
   * A single server process that both manages and serves models.
   * Can add new model sources and downloaders easily by writing just one function. See [this](src/model_downloader.h).
   * A separate [Model Manager](src/model_manager.h) module that can be integrated in the GenAI lib.
   * Recognizes that models were downloaded before and doesn't download them again (obvious? yeah, but requires code).
   * Supports multiple models.

### Input to the server
   * A [manifest file](test/model_manifest.json) that defines the model and its source. Think of this as the registry of all ONNX model manifests. For now it's local but can be stored on a server.
   * Path to the downloaded models that is managed by the server. This is where all the models will be downloaded.

### Pending
   * Clean up code
   * Support EPs other than CPU
   * Cross-platform
   * Multi-modal APIs
   * Write a better client application
   * Handle concurrency concerns
   * Streaming updates when model files are downloaded

## Build and Install
### Important Issues
   * genai lib requires a fix in the header file to make a few functions 'inline' or else server won't compile
   * Openssl 3.x or later is required

First install the latest onnxruntime-genai pkg here https://github.com/microsoft/onnxruntime-genai/releases/tag/v0.3.0
*
```
mkdir build
cd build
For release build: cmake -DORT_GENAI_DIR=<path of genai pkg installation> ..
For debug build: cmake -DORT_GENAI_DIR=<path of genai pkg installation> -DCMAKE_BUILD_TYPE=Debug ..
```

## Run
```
cd build
LD_LIBRARY_PATH=<path of genai pkg installation> ./ort_app_server -h (to see help)
```

## Test
* Hydrade the manifest file with the models
* In one terminal, run the server as shown above (preferably in verbose mode)
* Now, in a second terminal - pull a model
* In a second terminal run ```python test/test_ort_app_server.py --model model_3 --do_stream```

## Contribute
   * See [ORT coding guidelines](https://github.com/microsoft/onnxruntime/blob/main/docs/Coding_Conventions_and_Standards.md)

