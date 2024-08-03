// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <sstream>
#include <thread>
#include <experimental/filesystem>

#include "CLI11.hpp"
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"
#include "json.hpp"
#include "spdlog/spdlog.h"
#include "utils.h"
#include "model_manager.h"

using json = nlohmann::json;
namespace fs = std::experimental::filesystem;

struct ServerConfig {
  std::string host = "localhost";
  int port = 8080;
  bool verbose_mode = false;
  int nthreads = 0;
  std::string model_manifest_file;
  std::string downloaded_models_path = "/tmp/ort_app_server/models";
  std::string cmd_line_model_path;
  std::string cmd_line_model_id;
};

static void SetSearchOptions(const json& req_data, std::unique_ptr<OgaGeneratorParams>& params) {
  std::vector<std::string> float_params{"min_length", "max_length", "top_p", "temperature",
                                        "top_k", "repetition_penalty", "num_beams", "num_return_sequences",
                                        "length_penalty"};
  std::vector<std::string> bool_params{"do_sample", "early_stopping"};

  for (auto& param : float_params) {
    if (oas::ContainsJsonKey(req_data, param.c_str())) {
      spdlog::debug("setting param [{}]", param);
      params->SetSearchOption(param.c_str(), req_data.value(param, 100 /* this default should not get used */));
    }
  }

  for (auto& param : bool_params) {
    if (oas::ContainsJsonKey(req_data, param.c_str()))
      params->SetSearchOptionBool(param.c_str(), req_data.value(param, false));
  }
}

static void HandleNonStreamingChatCompletion(
    const json& req_data,
    const std::string& prompt_str,
    const std::string& model_id,
    oas::ModelManager& model_mgr,
    const httplib::Request& req,
    httplib::Response& res) {
  spdlog::debug("Serving non-streaming request");
  auto sequences = OgaSequences::Create();

  std::string to_search = "<|user|>";
  auto pos = prompt_str.rfind(to_search);

  auto* model_runner = model_mgr.GetModelRunner(model_id);
  auto& oga_model = model_runner->oga_model;
  auto& oga_tokenizer = model_runner->oga_tokenizer;

  if (pos != std::string::npos) {
    auto prompt_str_new = prompt_str.substr(pos);
    oga_tokenizer->Encode(prompt_str_new.c_str(), *sequences);
  } else {
    oga_tokenizer->Encode(prompt_str.c_str(), *sequences);
  }
  auto params = OgaGeneratorParams::Create(*oga_model);
  SetSearchOptions(req_data, params);
  params->SetInputSequences(*sequences);
  auto output_sequences = oga_model->Generate(*params);
  auto out_string = oga_tokenizer->Decode(output_sequences->SequenceData(0), output_sequences->SequenceCount(0));
  json json_res = oas::FormatNonStreamingChatResponse(static_cast<const char*>(out_string));
  const std::string response = json_res.dump(-1, ' ', false, json::error_handler_t::replace);
  res.set_content(response, "application/json; charset=utf-8");
}

static void HandleStreamingChatCompletion(
    const json& req_data,
    const std::string& prompt_str,
    const std::string& model_id,
    oas::ModelManager& model_mgr,
    const httplib::Request& req,
    httplib::Response& res) {
  spdlog::debug("Serving streaming request for model [{}] for prompt [{}]", model_id, prompt_str);
  auto chunked_content_provider = [&, prompt_str, model_id](size_t, httplib::DataSink& sink) {
    // spdlog::debug("inside chunked, model_id: [{}], prompt: [{}]", model_id, prompt_str);
    auto sequences = OgaSequences::Create();

    std::string to_search = "<|user|>";
    if (prompt_str.empty()) {
      spdlog::debug("prompt is empty inside chunked_content_provider");
      return false;
    }
    auto pos = prompt_str.rfind(to_search);

    auto* model_runner = model_mgr.GetModelRunner(model_id);
    auto& oga_model = model_runner->oga_model;
    auto& oga_tokenizer = model_runner->oga_tokenizer;
    auto& oga_tokenizer_stream = model_runner->oga_tokenizer_stream;

    if (pos != std::string::npos) {
      auto prompt_str_new = prompt_str.substr(pos);
      oga_tokenizer->Encode(prompt_str_new.c_str(), *sequences);
    } else {
      oga_tokenizer->Encode(prompt_str.c_str(), *sequences);
    }
    auto params = OgaGeneratorParams::Create(*oga_model);
    if (!params) {
      spdlog::debug("nullptr params");
      return false;
    }
    SetSearchOptions(req_data, params);
    params->SetInputSequences(*sequences);
    auto generator = OgaGenerator::Create(*oga_model, *params);
    if (!generator) {
      spdlog::debug("nullptr generator");
      return false;
    }
    while (!generator->IsDone()) {
      generator->ComputeLogits();
      generator->GenerateNextToken();

      const auto num_tokens = generator->GetSequenceCount(0);
      const auto new_token = generator->GetSequenceData(0)[num_tokens - 1];
      const auto decode_c_str = oga_tokenizer_stream->Decode(new_token);
      // spdlog::debug("after decode, before format...");
      json json_res = oas::FormatStreamingChatResponse(decode_c_str, false);
      const std::string str = "data: " + json_res.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
      // spdlog::debug("Writing to stream [{}]", str);
      if (!sink.write(str.c_str(), str.size())) {
        spdlog::info("Failed to write to the sink (probably because the client severed the connection)");
        return false;
      }
    }

    json json_res = oas::FormatStreamingChatResponse("", true);
    const std::string str = "data: " + json_res.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
    // spdlog::debug("Writing to stream [{}]", str);
    if (!sink.write(str.c_str(), str.size())) {
      spdlog::info("Failed to write to the sink (probably because the client severed the connection)");
      return false;
    }
    sink.done();
    return true;
  };

  auto on_complete = [](bool) {
    // cancel
    spdlog::debug("On_complete finished");
  };

  res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
}

static void SetBadRequest(httplib::Response& res, const std::string& msg) {
  res.status = 400;
  res.set_content(msg, "application/text");
}

static void HandleChatCompletions(oas::ModelManager& model_mgr, const httplib::Request& req, httplib::Response& res) {
  res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
  json req_data = json::parse(req.body);

  if (!oas::ContainsJsonKey(req_data, "messages")) {
    SetBadRequest(res, "The key 'messages' was missing in request");
    return;
  }

  if (!oas::ContainsJsonKey(req_data, "model")) {
    SetBadRequest(res, "The key 'model' was missing in request");
    return;
  }
  auto model_id = req_data["model"].get<std::string>();

  if (!model_mgr.GetModelRunner(model_id)) {
    spdlog::info("Model [{}] was not loaded before. Attempting to load the model first.", model_id);
    auto st = model_mgr.LoadModel(model_id);
    if (st != oas::Status::kOk) {
      switch (st) {
        case oas::Status::kFail: {
          res.status = 500;
          res.set_content("Failed to load model", "application/text");
          break;
        }
        case oas::Status::kModelNotDownloaded: {
          res.status = 400;
          res.set_content("Model was not downloaded before", "application/text");
          break;
        }
      }
      return;
    }
  }

  json messages_arr = req_data["messages"];
  std::string prompt_str{};
  for (auto& element : messages_arr) {
    auto role = oas::GetJsonValue<std::string>(element, "role", "");
    auto content = oas::GetJsonValue<std::string>(element, "content", "");
    if (role == "user" && !content.empty()) {
      prompt_str = content;
      break;
    }
  }
  if (prompt_str.empty()) {
    SetBadRequest(res, "User content missing in request");
    return;
  }

  spdlog::debug("Received prompt: [{}]", prompt_str);
  if (oas::GetJsonValue<bool>(req_data, "stream", false)) {
    HandleStreamingChatCompletion(req_data, prompt_str, model_id, model_mgr, req, res);
  } else {
    HandleNonStreamingChatCompletion(req_data, prompt_str, model_id, model_mgr, req, res);
  }
}

static void HandleListLoadedModels(oas::ModelManager& model_mgr, const httplib::Request& req, httplib::Response& res) {
  auto models = model_mgr.GetLoadedModelsList();
  json ret;
  ret["models"] = json::array();
  for (auto& s : models) {
    ret["models"].push_back(s);
  }
  res.status = 200;
  res.set_content(ret.dump(), "application/json");
}

static void HandleListModels(oas::ModelManager& model_mgr, const httplib::Request& req, httplib::Response& res) {
  auto models = model_mgr.GetModelsFromManifest();
  json ret;
  ret["models"] = json::array();
  for (auto& s : models) {
    ret["models"].push_back(s);
  }
  res.status = 200;
  res.set_content(ret.dump(), "application/json");
}

static void HandleUnloadModel(oas::ModelManager& model_mgr, const httplib::Request& req, httplib::Response& res) {
  res.set_content("Not implemented", "application/text");
}

static void HandlePullModel(oas::ModelManager& model_mgr, const ServerConfig& svr_config, const httplib::Request& req, httplib::Response& res) {
  res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
  json req_data = json::parse(req.body);
  const std::string model_id = req_data["model"].get<std::string>();  // TODO: validation
  spdlog::debug("Pulling model [{}]", model_id);
  auto ret = model_mgr.DownloadModel(model_id);
  switch (ret.first) {
    case oas::Status::kOk: {
      res.status = 200;
      res.set_content("Pulled model successfully.", "application/text");
      break;
    }
    case oas::Status::kModelAlreadyDownloaded: {
      res.status = 200;
      res.set_content("Model was already pulled.", "application/text");
      break;
    }
    case oas::Status::kModelNotRecognized: {
      res.status = 400;
      res.set_content("Model not recognized as it's not in the manifest.", "application/text");
      break;
    }
    case oas::Status::kFail: {
      res.status = 500;
      std::string err = "Failed to pull model. Error: " + ret.second;
      res.set_content(err, "application/text");
      break;
    }
  }
}

static void HandleLoadModel(oas::ModelManager& model_mgr, const httplib::Request& req, httplib::Response& res) {
  res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
  json req_data = json::parse(req.body);
  const std::string model_id = req_data["model"].get<std::string>();
  spdlog::debug("Loading model [{}]", model_id);
  auto st = model_mgr.LoadModel(model_id);
  switch (st) {
    case oas::Status::kFail: {
      res.status = 500;
      res.set_content("Failed to load model", "application/text");
      break;
    }
    case oas::Status::kModelAlreadyLoaded: {
      res.status = 200;
      res.set_content("Model already loaded", "application/text");
      break;
    }
    case oas::Status::kModelNotDownloaded: {
      res.status = 400;
      res.set_content("Model was not pulled before", "application/text");
      break;
    }
    case oas::Status::kOk: {
      res.status = 200;
      res.set_content("Loaded model successfully", "application/text");
      break;
    }
  }
}

static void SetupEndpoints(httplib::Server& svr, oas::ModelManager& model_mgr, const ServerConfig& svr_config) {
  svr.Get("/v1/health", [&](const httplib::Request& req, httplib::Response& res) {
    res.status = 200;
    std::string content = "I'm Good!";
    res.set_content(content, "application/text");
  });

  svr.Get("/v1/ps", [&model_mgr](const httplib::Request& req, httplib::Response& res) {
    HandleListLoadedModels(model_mgr, req, res);
  });

  svr.Get("/v1/models", [&model_mgr](const httplib::Request& req, httplib::Response& res) {
    HandleListModels(model_mgr, req, res);
  });

  svr.Post("/v1/pull", [&model_mgr, &svr_config](const httplib::Request& req, httplib::Response& res) {
    HandlePullModel(model_mgr, svr_config, req, res);
  });

  svr.Post("/v1/load", [&model_mgr](const httplib::Request& req, httplib::Response& res) {
    HandleLoadModel(model_mgr, req, res);
  });

  svr.Post("/v1/unload", [&model_mgr](const httplib::Request& req, httplib::Response& res) {
    HandleUnloadModel(model_mgr, req, res);
  });

  svr.Post("/v1/chat/completions", [&model_mgr](const httplib::Request& req, httplib::Response& res) {
    HandleChatCompletions(model_mgr, req, res);
  });
}

static oas::Status LoadModelFromCmdLine(const std::string& model_id, const std::string& model_path, oas::ModelManager& model_mgr) {
  spdlog::info("Loading model from the cmd line [{}]", model_path);
  model_mgr.AddModelMetadata(model_id, model_path);
  return model_mgr.LoadModel(model_id);
}

static void ServerLogger(const httplib::Request& req, const httplib::Response& res) {
  if (spdlog::get_level() == spdlog::level::debug) {
    std::ostringstream ostr;
    ostr << "Request: " << req.path << " " << req.body;
    spdlog::debug(ostr.str());
  }
}

static void SetupServer(const ServerConfig& svr_config, httplib::Server& svr) {
  svr.set_logger(ServerLogger);
  if (svr_config.nthreads != 0) {
    spdlog::debug("Using threadcount of [{}]", svr_config.nthreads);
    svr.new_task_queue = [&svr_config] { return new httplib::ThreadPool(svr_config.nthreads); };
  }
  svr.set_error_handler([](const httplib::Request&, httplib::Response& res) {
    if (res.status == 401) {
      res.set_content("Unauthorized", "text/plain; charset=utf-8");
    } else if (res.status == 404) {
      res.set_content("File Not Found", "text/plain; charset=utf-8");
      res.status = 404;
    }
  });

  svr.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
    const char fmt[] = "500 Internal Server Error\n%s";
    char buf[BUFSIZ];
    try {
      std::rethrow_exception(std::move(ep));
    } catch (std::exception& e) {
      snprintf(buf, sizeof(buf), fmt, e.what());
    } catch (...) {
      snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
    }
    res.set_content(buf, "text/plain; charset=utf-8");
    res.status = 500;
  });

  // TODO: setup read/write timeouts
}

static void RunServer(const ServerConfig& svr_config, httplib::Server& svr) {
  std::thread http_server_thread([&]() {
    if (!svr.listen(svr_config.host, svr_config.port)) {
      return 1;  // TODO?
    }
    return 0;  // TODO?
  });
  http_server_thread.join();
}

static void ReadCmdLineParams(int argc, char** argv, ServerConfig& svr_config) {
  CLI::App app("ORT App Server");
  argv = app.ensure_utf8(argv);
  app.add_flag("-v, --verbose", svr_config.verbose_mode);
  app.add_option("-n,--hostname", svr_config.host, "Hostname to listen on (default: localhost)");
  app.add_option("-p,--port", svr_config.port, "Port number to listen on (default: 8080)");
  app.add_option("-t,--nthreads", svr_config.nthreads, "Numbter of threads to use");
  app.add_option("-i,--model_id", svr_config.cmd_line_model_id,
                 "Model id (required if --model is used. "
                 "Model id is used to identify the model in the server.)");
  app.add_option("-m,--model", svr_config.cmd_line_model_path, "Model folder containing the model to load");
  app.add_option("-f, --model_manifest_file", svr_config.model_manifest_file, "Model manifest file");
  app.add_option("-d, --downloaded_models_path", svr_config.downloaded_models_path,
                 "Folder where models are downloaded (default: /tmp/ort_app_server/models/).");
  try {
    app.parse(argc, argv);
  } catch (CLI::Error& e) {
    exit(app.exit(e));
  }
}

int main(int argc, char** argv) {
  ServerConfig svr_config;

  ReadCmdLineParams(argc, argv, svr_config);
  if (svr_config.verbose_mode)
    spdlog::set_level(spdlog::level::level_enum::debug);

  oas::ModelManager model_mgr(svr_config.downloaded_models_path);

  // Read manifest file if supplied
  if (!svr_config.model_manifest_file.empty()) {
    auto rc = model_mgr.InitializeModelManifestRegistry(svr_config.model_manifest_file);
    if (rc != oas::Status::kOk) {
      spdlog::error("Failed to read manifest file [{}]", svr_config.model_manifest_file);
      exit(1);
    }
  }

  // Load model from the cmd line if supplied
  if (!svr_config.cmd_line_model_path.empty()) {
    if (svr_config.cmd_line_model_id.empty()) {
      spdlog::error("--model_id is required if --model is used. Model id is used to identify the model in the server.");
      exit(1);
    }
    auto st = LoadModelFromCmdLine(svr_config.cmd_line_model_id, svr_config.cmd_line_model_path, model_mgr);
    if (st != oas::Status::kOk) {
      spdlog::error("Failed to load model supplied on the cmd line.");
      exit(1);
    }
  }

  httplib::Server svr;
  SetupServer(svr_config, svr);
  SetupEndpoints(svr, model_mgr, svr_config);
  RunServer(svr_config, svr);
  return 0;
}