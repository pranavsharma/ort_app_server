// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "spdlog/spdlog.h"
#include <json.hpp>
#include "utils.h"
#include <experimental/filesystem>

#include "ort_genai.h"
#include "model_downloader.h"

using json = nlohmann::json;
namespace fs = std::experimental::filesystem;

namespace oas {
class ModelManager {
 public:
  ModelManager(const std::string& manifest_file, const std::string& downloaded_models_path0);
  struct ModelRunner {
    std::unique_ptr<OgaModel> oga_model;
    std::unique_ptr<OgaTokenizer> oga_tokenizer;
    std::unique_ptr<OgaTokenizerStream> oga_tokenizer_stream;
  };

  bool WasModelDownloaded(const std::string& model_id);
  std::pair<Status, std::string> DownloadModel(const std::string& model_id);
  ModelRunner* GetModelRunner(const std::string& model_id);
  Status LoadModel(const std::string& model_id);
  std::vector<std::string> GetLoadedModelsList();

 private:
  Status InitializeModelManifestRegistry(const std::string& manifest_file);
  Status LoadModelsFromDisk(const std::string& downloaded_models_path);
  Status LoadModelImpl(const std::string& model_path, ModelRunner& model_runner);
  struct ModelInfo {
    std::string model_id;
    std::string model_path;
  };
  struct ModelRegistry {
    bool WasModelDownloaded(const std::string& model_id) {
      return model_info_registry.count(model_id);
    }
    const std::string& GetModelPath(const std::string& model_id) {
      return model_info_registry[model_id].model_path;
    }
    void AddModelRunner(const std::string& model_id, ModelRunner&& model_runner) {
      model_runner_registry[model_id] = std::move(model_runner);
    }
    void AddModelInfo(const std::string& model_id, const std::string& model_path) {
      model_info_registry[model_id] = {model_id, model_path};
    }
    void AddModel(const std::string& model_id, const std::string& model_path, ModelRunner&& model_runner) {
      model_runner_registry[model_id] = std::move(model_runner);
      model_info_registry[model_id] = {model_id, model_path};
    }
    // TODO returns a ptr to an internal member, not good fix it later
    ModelRunner* GetModelRunner(const std::string& model_id) {
      if (!model_runner_registry.count(model_id)) return nullptr;
      return &model_runner_registry[model_id];
    }
    std::unordered_map<std::string, ModelRunner> model_runner_registry;
    std::unordered_map<std::string, ModelInfo> model_info_registry;
  };

  enum class ModelSource {
    kHuggingFace,
    kLocal,
    kUnknown
  };
  struct ModelManifest {
    std::string model_id;
    std::string include_filter;
    std::string base_path;
    ModelSource model_source = ModelSource::kUnknown;
  };
  struct ModelManifestRegistry {
    std::unordered_map<std::string, ModelManifest> reg;
  };
  std::unordered_map<ModelSource, ModelDownloader> model_hub_type_downloader_map;
  ModelManifestRegistry model_manifest_registry;
  ModelRegistry model_registry;
  std::string downloaded_models_path;
};
}  // namespace oas