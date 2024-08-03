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
  ModelManager(const std::string& downloaded_models_path0);
  struct ModelRunner {
    std::unique_ptr<OgaModel> oga_model;
    std::unique_ptr<OgaTokenizer> oga_tokenizer;
    std::unique_ptr<OgaTokenizerStream> oga_tokenizer_stream;
  };

  Status InitializeModelManifestRegistry(const std::string& manifest_file);
  bool WasModelDownloaded(const std::string& model_id);
  std::pair<Status, std::string> DownloadModel(const std::string& model_id);
  ModelRunner* GetModelRunner(const std::string& model_id);
  Status LoadModel(const std::string& model_id);
  void AddModelMetadata(const std::string& model_id, const std::string& model_path);
  std::vector<std::string> GetLoadedModelsList();
  std::vector<std::string> GetModelsFromManifest();

 private:
  Status LoadModelsFromDisk(const std::string& downloaded_models_path);
  Status LoadModelImpl(const std::string& model_path, ModelRunner& model_runner);
  struct ModelMetadata {
    std::string model_id;
    std::string model_path_on_disk;
  };
  struct ModelRegistry {
    using ModelRunnerRegistry = std::unordered_map<std::string, ModelRunner>;
    using ModelMetadataRegistry = std::unordered_map<std::string, ModelMetadata>;

    bool WasModelDownloaded(const std::string& model_id) const {
      return model_metadata_registry.count(model_id);
    }
    const std::string& GetModelPath(const std::string& model_id) const {
      return model_metadata_registry.at(model_id).model_path_on_disk;
    }
    void AddModelRunner(const std::string& model_id, ModelRunner&& model_runner) {
      model_runner_registry[model_id] = std::move(model_runner);
    }
    void AddModelMetadata(const std::string& model_id, const std::string& model_path) {
      model_metadata_registry[model_id] = {model_id, model_path};
    }
    // TODO returns a ptr to an internal member, not good fix it later, should ideally be a const function
    ModelRunner* GetModelRunner(const std::string& model_id) {
      if (!model_runner_registry.count(model_id)) return nullptr;
      return &model_runner_registry.at(model_id);
    }
    const ModelRunnerRegistry& GetModelRunnerRegistry() const {
      return model_runner_registry;
    }

    ModelRunnerRegistry model_runner_registry;      // stores data about the models loaded in memory
    ModelMetadataRegistry model_metadata_registry;  // stores data about the models pulled/downloaded
    std::mutex mtx;
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
  std::unordered_map<ModelSource, ModelDownloader> model_hub_type_downloader_map;
  using ModelManifestRegistry = std::unordered_map<std::string, ModelManifest>;
  ModelManifestRegistry model_manifest_registry;
  ModelRegistry model_registry;
  std::string downloaded_models_path;
};
}  // namespace oas