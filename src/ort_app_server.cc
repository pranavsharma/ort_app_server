#include <iostream>
#include <sstream>
#include <thread>

#include "httplib.h"
#include "json.hpp"
#include "ort_genai.h"
#include "spdlog/spdlog.h"
#include "CLI11.hpp"
#include "utils.h"

using json = nlohmann::json;

struct ServerConfig {
  std::string host = "localhost";
  int port = 8080;
  std::string model_dir = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4";
  bool verbose_mode = false;
};

struct ServerState {
  std::unique_ptr<OgaModel> oga_model;
  std::unique_ptr<OgaTokenizer> oga_tokenizer;
  std::unique_ptr<OgaTokenizerStream> oga_tokenizer_stream;
};

static void SetSearchOptions(const json& req_data, std::unique_ptr<OgaGeneratorParams>& params) {
  std::vector<std::string> float_params{"max_length", "top_p", "temperature", "top_k", "repetition_penalty"};
  std::vector<std::string> bool_params{"do_sample"};

  for (auto& param : float_params) {
    if (oga::HasJsonKey(req_data, param.c_str())) {
      spdlog::debug("setting param ", param);
      params->SetSearchOption(param.c_str(), req_data.value(param, 100));
    }
  }

  for (auto& param : bool_params) {
    if (oga::HasJsonKey(req_data, param.c_str())) params->SetSearchOptionBool(param.c_str(), req_data.value(param, false));
  }
}

static void HandleNonStreamingChatCompletion(const json& req_data, const std::string& prompt_str,
                                             const ServerState& svr_state, const httplib::Request& req,
                                             httplib::Response& res) {
  auto sequences = OgaSequences::Create();

  std::string to_search = "<|user|>";
  auto pos = prompt_str.rfind(to_search);

  if (pos != std::string::npos) {
    auto prompt_str_new = prompt_str.substr(pos);
    svr_state.oga_tokenizer->Encode(prompt_str_new.c_str(), *sequences);
  } else {
    svr_state.oga_tokenizer->Encode(prompt_str.c_str(), *sequences);
  }
  auto params = OgaGeneratorParams::Create(*svr_state.oga_model);
  SetSearchOptions(req_data, params);
  params->SetInputSequences(*sequences);
  auto output_sequences = svr_state.oga_model->Generate(*params);
  auto out_string = svr_state.oga_tokenizer->Decode(output_sequences->SequenceData(0), output_sequences->SequenceCount(0));
  json json_res = oga::FormatNonStreamingChatResponse(static_cast<const char*>(out_string));
  const std::string response = json_res.dump(-1, ' ', false, json::error_handler_t::replace);
  res.set_content(response, "application/json; charset=utf-8");
}

static void HandleStreamingChatCompletion(const json& req_data, const std::string& prompt_str,
                                          const ServerState& svr_state, const httplib::Request& req,
                                          httplib::Response& res) {
  const auto chunked_content_provider = [&req_data, prompt_str, &svr_state](size_t, httplib::DataSink& sink) {
    auto sequences = OgaSequences::Create();

    std::string toSearch = "<|user|>";
    auto pos = prompt_str.rfind(toSearch);

    if (pos != std::string::npos) {
      auto prompt_str_new = prompt_str.substr(pos);
      svr_state.oga_tokenizer->Encode(prompt_str_new.c_str(), *sequences);
    } else {
      svr_state.oga_tokenizer->Encode(prompt_str.c_str(), *sequences);
    }
    auto params = OgaGeneratorParams::Create(*svr_state.oga_model);
    SetSearchOptions(req_data, params);
    params->SetInputSequences(*sequences);
    auto generator = OgaGenerator::Create(*svr_state.oga_model, *params);

    while (!generator->IsDone()) {
      generator->ComputeLogits();
      generator->GenerateNextToken();

      const auto num_tokens = generator->GetSequenceCount(0);
      const auto new_token = generator->GetSequenceData(0)[num_tokens - 1];
      const auto decode_c_str = svr_state.oga_tokenizer_stream->Decode(new_token);

      json json_res = oga::FormatStreamingChatResponse(decode_c_str, false);
      const std::string str =
          "data: " +
          json_res.dump(-1, ' ', false, json::error_handler_t::replace) +
          "\n\n";
      if (!sink.write(str.c_str(), str.size())) {
        spdlog::info("Failed to write to the sink (probably because the client severed the connection)");
        return false;
      }
    }

    json json_res = oga::FormatStreamingChatResponse("", true);
    const std::string str =
        "data: " +
        json_res.dump(-1, ' ', false, json::error_handler_t::replace) +
        "\n\n";
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

  res.set_chunked_content_provider("text/event-stream",
                                   chunked_content_provider, on_complete);
}

static void HandleChatCompletions(const ServerState& svr_state, const httplib::Request& req, httplib::Response& res) {
  res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
  json req_data = json::parse(req.body);
  std::string prompt_str = req_data["prompt"].dump();

  if (!oga::JsonValue(req_data, "stream", false)) {
    HandleNonStreamingChatCompletion(req_data, prompt_str, svr_state, req, res);
  } else {
    HandleStreamingChatCompletion(req_data, prompt_str, svr_state, req, res);
  }
}

static void SetupEndpoints(httplib::Server& svr, const ServerState& svr_state) {
  svr.Get("/health", [&](const httplib::Request& req, httplib::Response& res) {
    res.status = 200;
    std::string content = "I'm Good!";
    res.set_content(content, "application/text");
  });

  svr.Post("/v1/chat/completions", [&svr_state](
                                       const httplib::Request& req,
                                       httplib::Response& res) {
    HandleChatCompletions(svr_state, req, res);
  });
}

static void LoadModel(const std::string& model_dir, ServerState& svr_state) {
  svr_state.oga_model = OgaModel::Create(model_dir.c_str());
  svr_state.oga_tokenizer = OgaTokenizer::Create(*svr_state.oga_model);
  svr_state.oga_tokenizer_stream = OgaTokenizerStream::Create(*svr_state.oga_tokenizer);
  spdlog::info("Model [{}] loaded successfully", model_dir);
}

static void ServerLogger(const httplib::Request& req, const httplib::Response& res) {
  if (spdlog::get_level() == spdlog::level::debug) {
    std::ostringstream ostr;
    ostr << "Request: " << req.path << " " << req.body;
    spdlog::debug(ostr.str());
  }
}

static void SetupServer(httplib::Server& svr) {
  svr.set_logger(ServerLogger);

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
  app.add_option("-m,--model", svr_config.model_dir, "Model directory")->required(true);
  app.add_option("-n,--hostname", svr_config.host, "Hostname to listen on");
  app.add_option("-p,--port", svr_config.port, "Port number to listen on");
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
  if (svr_config.verbose_mode) spdlog::set_level(spdlog::level::level_enum::debug);

  LoadModel(svr_config.model_dir, svr_state);

  httplib::Server svr;
  SetupServer(svr);
  SetupEndpoints(svr, svr_state);
  RunServer(svr_config, svr);

  return 0;
}
