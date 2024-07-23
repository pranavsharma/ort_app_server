# ORT App Server (experimental)
* This is under development and is rough.
* Only CPU is supported at this time.
* Only language models are supported at this time.
* The HTTP API is OpenAI _mostly_ compliant as in it supports a limited set of HTTP request params like top_p, temperature, top_k, repetition_penalty.

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