# ORT App Server (experimental)
* This is under active development.
* Current limitations
   * Works on Linux only
   * Only CPU
   * Only language models
   * The HTTP API is OpenAI compliant but supports only a limited set of (optional) HTTP request params
        like max_length, top_p, temperature, top_k, repetition_penalty, num_beams and do_sample.
        See https://onnxruntime.ai/docs/genai/reference/config.html#search-combinations

## Build and Install
* First install the latest onnxruntime-genai pkg here https://github.com/microsoft/onnxruntime-genai/releases/tag/v0.3.0
*
```
mkdir build
cd build
cmake -DORT_GENAI_DIR=<path of genai pkg installation> ..
```

## Run
```
cd build
LD_LIBRARY_PATH=<path of genai pkg installation> ./ort_app_server -h (to see help)
LD_LIBRARY_PATH=<path of genai pkg installation> ./ort_app_server -v -m <path of model folder>
```

## Test
* First run the server as shown above.
* In a second terminal run ```python test/test_ort_app_server.py```
