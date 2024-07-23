#include "json.hpp"

namespace oga {
static nlohmann::json FormatStreamingChatResponse(const char* content, bool stop) {
  /*
  OpenAI format
      /*
      {
          "id": "chatcmpl-123",
          "object": "chat.completion.chunk",
          "created": 1694268190,
          "model": "gpt-4o-mini",
          "system_fingerprint": "fp_44709d6fcb",
          "choices": [
              {
                  "index": 0,
                  "delta": {
                      "role": "assistant",
                      "content": ""
                  },
                  "logprobs": null,
                  "finish_reason": null
              }
          ]
      }
      */
  // std::cout << "forming response, content: " << content << "\n";
  nlohmann::json response;
  response["id"] = "ort-app-server-123";
  response["object"] = "chat.completion.chunk";
  nlohmann::json choice_object;
  choice_object["index"] = 0;
  nlohmann::json delta_object;
  delta_object["content"] = content;
  choice_object["delta"] = delta_object;
  response["choices"] = {choice_object};
  return response;
}

static nlohmann::json FormatNonStreamingChatResponse(const char* content) {
  /*
  OpenAI format
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-4o-mini",
  "system_fingerprint": "fp_44709d6fcb",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "\n\nHello there, how may I assist you today?",
    },
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}      */
  // std::cout << "forming response, content: " << content << "\n";
  nlohmann::json response;
  response["id"] = "ort-app-server-123";
  response["object"] = "chat.completion";
  nlohmann::json choice_object;
  choice_object["index"] = 0;
  nlohmann::json message_object;
  message_object["content"] = content;
  choice_object["message"] = message_object;
  response["choices"] = {choice_object};
  return response;
}

static bool HasJsonKey(const nlohmann::json& body, const std::string& key) {
  // Fallback null to default value
  return body.contains(key) && !body.at(key).is_null();
}

template <typename T>
static T JsonValue(const nlohmann::json& body, const std::string& key,
                   const T& default_value) {
  // Fallback null to default value
  return body.contains(key) && !body.at(key).is_null()
             ? body.value(key, default_value)
             : default_value;
}
}  // namespace oga
