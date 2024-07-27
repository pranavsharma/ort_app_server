#include <iostream>
#include <sstream>
#include <thread>

#include "CLI11.hpp"
#include "httplib.h"
#include "json.hpp"
#include "ort_genai.h"
#include "spdlog/spdlog.h"
#include "utils.h"

using json = nlohmann::json;
using namespace ops;

static const std::string CMDLINE_MODEL_NAME = "cmdline_model";
struct ModelManifest {
  std::vector<std::string> urls{"", "", "", ""};
};
struct ModelManifestRegistry {
  std::unordered_map<std::string, ModelManifest> reg;
};
struct ServerConfig {
  ModelManifestRegistry model_manifest_registry;
  std::string host = "localhost";
  int port = 8080;
  std::string model_path = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4";
  bool verbose_mode = false;
  int nthreads = 0;
};

struct ModelRunner {
  std::unique_ptr<OgaModel> oga_model;
  std::unique_ptr<OgaTokenizer> oga_tokenizer;
  std::unique_ptr<OgaTokenizerStream> oga_tokenizer_stream;
};

struct ModelLoader {
  static Status LoadModel(const std::string& model_path, ModelRunner& model_runner) {
    model_runner.oga_model = OgaModel::Create(model_path.c_str());
    if (!model_runner.oga_model) {
      spdlog::error("could not create model for {}", model_path);
      return Status::FAIL;
    }
    model_runner.oga_tokenizer = OgaTokenizer::Create(*model_runner.oga_model);
    model_runner.oga_tokenizer_stream = OgaTokenizerStream::Create(*model_runner.oga_tokenizer);
    spdlog::info("Model [{}] loaded successfully", model_path);
    return Status::OK;
  }
};
struct ModelInfo {
  std::string model_name;
  std::string model_path;
};
struct ModelRegistry {
  bool IsModelDownloaded(const std::string& model_name) {
    return model_info_registry.count(model_name);
  }
  const std::string& GetModelPath(const std::string& model_name) {
    return model_info_registry[model_name].model_path;
  }
  void AddModelRunner(const std::string& model_name, ModelRunner&& model_runner) {
    model_runner_registry[model_name] = std::move(model_runner);
  }
  void AddModel(const std::string& model_name, const std::string& model_path, ModelRunner&& model_runner) {
    model_runner_registry[model_name] = std::move(model_runner);
    model_info_registry[model_name] = {model_name, model_path};
  }
  // TODO returns a ptr to an internal member, not good fix it later
  ModelRunner* GetModelRunner(const std::string& model_name) {
    if (!model_runner_registry.count(model_name)) return nullptr;
    return &model_runner_registry[model_name];
  }
  std::unordered_map<std::string, ModelRunner> model_runner_registry;
  std::unordered_map<std::string, ModelInfo> model_info_registry;
};
struct ServerState {
  ModelRegistry model_registry;
};

static void SetSearchOptions(const json& req_data, std::unique_ptr<OgaGeneratorParams>& params) {
  std::vector<std::string> float_params{"min_length", "max_length", "top_p", "temperature",
                                        "top_k", "repetition_penalty", "num_beams", "num_return_sequences",
                                        "length_penalty"};
  std::vector<std::string> bool_params{"do_sample", "early_stopping"};

  for (auto& param : float_params) {
    if (HasJsonKey(req_data, param.c_str())) {
      spdlog::debug("setting param {}", param);
      params->SetSearchOption(param.c_str(), req_data.value(param, 100 /* this default should not get used */));
    }
  }

  for (auto& param : bool_params) {
    if (HasJsonKey(req_data, param.c_str()))
      params->SetSearchOptionBool(param.c_str(), req_data.value(param, false));
  }
}

static void HandleNonStreamingChatCompletion(
    const json& req_data,
    const std::string& prompt_str,
    const std::string& model_name,
    ServerState& svr_state,
    const httplib::Request& req,
    httplib::Response& res) {
  spdlog::debug("Serving non-streaming request");
  auto sequences = OgaSequences::Create();

  std::string to_search = "<|user|>";
  auto pos = prompt_str.rfind(to_search);

  auto* model_runner = svr_state.model_registry.GetModelRunner(model_name);
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
  json json_res = FormatNonStreamingChatResponse(static_cast<const char*>(out_string));
  const std::string response = json_res.dump(-1, ' ', false, json::error_handler_t::replace);
  res.set_content(response, "application/json; charset=utf-8");
}

static void HandleStreamingChatCompletion(
    const json& req_data,
    const std::string& prompt_str,
    const std::string& model_name,
    ServerState& svr_state,
    const httplib::Request& req,
    httplib::Response& res) {
  spdlog::debug("Serving streaming request");
  const auto chunked_content_provider = [&req_data, &prompt_str, &model_name, &svr_state](size_t, httplib::DataSink& sink) {
    auto sequences = OgaSequences::Create();

    std::string toSearch = "<|user|>";
    auto pos = prompt_str.rfind(toSearch);

    auto* model_runner = svr_state.model_registry.GetModelRunner(model_name);
    if (!model_runner) {
      spdlog::error("nullptr");
      return false;
    }
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
    SetSearchOptions(req_data, params);
    params->SetInputSequences(*sequences);
    auto generator = OgaGenerator::Create(*oga_model, *params);

    while (!generator->IsDone()) {
      generator->ComputeLogits();
      generator->GenerateNextToken();

      const auto num_tokens = generator->GetSequenceCount(0);
      const auto new_token = generator->GetSequenceData(0)[num_tokens - 1];
      const auto decode_c_str = oga_tokenizer_stream->Decode(new_token);

      json json_res = FormatStreamingChatResponse(decode_c_str, false);
      const std::string str = "data: " + json_res.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
      // spdlog::debug("Writing to stream {}", str);
      if (!sink.write(str.c_str(), str.size())) {
        spdlog::info("Failed to write to the sink (probably because the client severed the connection)");
        return false;
      }
    }

    json json_res = FormatStreamingChatResponse("", true);
    const std::string str = "data: " + json_res.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
    // spdlog::debug("Writing to stream {}", str);
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

static void HandleChatCompletions(ServerState& svr_state, const httplib::Request& req, httplib::Response& res) {
  res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
  json req_data = json::parse(req.body);

  if (!HasJsonKey(req_data, "messages")) {
    SetBadRequest(res, "The key 'messages' was missing in request");
    return;
  }

  auto model_name = CMDLINE_MODEL_NAME;
  if (!HasJsonKey(req_data, "model")) {
    spdlog::warn("Using the model that was supplied on the cmd line when the server was started.");
  } else {
    model_name = req_data["model"].dump();
  }
  if (!svr_state.model_registry.GetModelRunner(model_name)) {
    spdlog::error("Model {} was not loaded before", model_name);
    SetBadRequest(res, "This model was not loaded before");
    return;
  }

  json messages_arr = req_data["messages"];
  std::string prompt_str{};
  for (auto& element : messages_arr) {
    auto role = JsonValue<std::string>(element, "role", "");
    auto content = JsonValue<std::string>(element, "content", "");
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
  if (JsonValue<bool>(req_data, "stream", false)) {
    HandleStreamingChatCompletion(req_data, prompt_str, model_name, svr_state, req, res);
  } else {
    HandleNonStreamingChatCompletion(req_data, prompt_str, model_name, svr_state, req, res);
  }
}

static void HandleListModels(ServerState& svr_state, const httplib::Request& req, httplib::Response& res) {
  res.set_content("Not implemented", "application/text");
}

static void HandlePullModel(ServerState& svr_state, const ServerConfig& svr_config, const httplib::Request& req, httplib::Response& res) {
  res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
  json req_data = json::parse(req.body);
  const std::string model_name = req_data["model"];  // TODO: validation
  const auto& urls_to_download = svr_config.model_manifest_registry.reg.at(model_name).urls;
  // TODO download these urls in a separate thread
}

static void HandleLoadModel(ServerState& svr_state, const httplib::Request& req, httplib::Response& res) {
  res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
  json req_data = json::parse(req.body);
  const std::string model_name = req_data["model"];  // TODO: validation
  ModelRunner model_runner;
  auto st = ModelLoader::LoadModel(model_name, model_runner);
  if (st != Status::FAIL) {
    res.status = 500;
    res.set_content("Failed to load model", "application/text");
    return;
  }
  svr_state.model_registry.AddModelRunner(model_name, std::move(model_runner));
}

static void SetupEndpoints(httplib::Server& svr, ServerState& svr_state, const ServerConfig& svr_config) {
  svr.Get("/health", [&](const httplib::Request& req, httplib::Response& res) {
    res.status = 200;
    std::string content = "I'm Good!";
    res.set_content(content, "application/text");
  });

  svr.Get("/v1/models", [&svr_state](const httplib::Request& req, httplib::Response& res) {
    HandleListModels(svr_state, req, res);
  });

  svr.Post("/v1/pull", [&svr_state, &svr_config](const httplib::Request& req, httplib::Response& res) {
    HandlePullModel(svr_state, svr_config, req, res);
  });

  svr.Post("/v1/load", [&svr_state](const httplib::Request& req, httplib::Response& res) {
    HandleLoadModel(svr_state, req, res);
  });

  svr.Post("/v1/chat/completions", [&svr_state](const httplib::Request& req, httplib::Response& res) {
    HandleChatCompletions(svr_state, req, res);
  });
}

static Status LoadModel(const std::string& model_name, ServerState& svr_state) {
  if (!svr_state.model_registry.IsModelDownloaded(model_name)) {
    spdlog::error("Model {} was not downloaded before.", model_name);
    return Status::FAIL;
  }
  ModelRunner model_runner;
  auto rc = ModelLoader::LoadModel(svr_state.model_registry.GetModelPath(model_name), model_runner);
  if (rc != Status::OK) {
    spdlog::error("Loading model {} failed", model_name);
    return rc;
  }
  svr_state.model_registry.AddModelRunner(model_name, std::move(model_runner));
  return Status::OK;
}

static Status LoadModelFromCmdLine(const std::string& model_path, ServerState& svr_state) {
  ModelRunner model_runner;
  auto rc = ModelLoader::LoadModel(model_path, model_runner);
  if (rc != Status::OK) {
    spdlog::error("Loading model {} failed", model_path);
    return rc;
  }
  svr_state.model_registry.AddModel(CMDLINE_MODEL_NAME, model_path, std::move(model_runner));
  return Status::OK;
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
    spdlog::debug("Using threadcount of {}", svr_config.nthreads);
    svr.new_task_queue = [&svr_config] { return new httplib::ThreadPool(svr_config.nthreads); };
  }
  svr.set_error_handler([](const httplib::Request&, httplib::Response& res) {
    if (res.status == 401) {
      res.set_content("Unauthorized", "text/plain; charset=utf-8");
    }
    if (res.status == 400) {
      res.set_content("Invalid request", "text/plain; charset=utf-8");
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
  app.add_option("-m,--model", svr_config.model_path, "Model path");
  app.add_option("-n,--hostname", svr_config.host, "Hostname to listen on");
  app.add_option("-p,--port", svr_config.port, "Port number to listen on");
  app.add_option("-t,--nthreads", svr_config.nthreads, "Numbter of threads to use");
  app.add_flag("-v, --verbose", svr_config.verbose_mode);
  try {
    app.parse(argc, argv);
  } catch (CLI::Error& e) {
    exit(app.exit(e));
  }
}

int main(int argc, char** argv) {
  ServerConfig svr_config;
  ServerState svr_state;

  ReadCmdLineParams(argc, argv, svr_config);
  if (svr_config.verbose_mode)
    spdlog::set_level(spdlog::level::level_enum::debug);

  LoadModelFromCmdLine(svr_config.model_path, svr_state);

  httplib::Server svr;
  SetupServer(svr_config, svr);
  SetupEndpoints(svr, svr_state, svr_config);
  RunServer(svr_config, svr);

  return 0;
}
