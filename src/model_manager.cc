// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include "model_manager.h"

// static const std::string kCmdLineModel = "cmdline_model";

namespace oas {
std::vector<std::string> ModelManager::GetLoadedModelsList() {
  std::vector<std::string> ret;
  for (auto& [model_id, _] : model_registry.model_runner_registry) {
    ret.push_back(model_id);
  }
  return ret;
}

ModelManager::ModelManager(const std::string& manifest_file, const std::string& downloaded_models_path0)
    : downloaded_models_path(downloaded_models_path0) {
  model_hub_type_downloader_map[ModelSource::kHuggingFace] = DownloadHuggingFaceModel;
  model_hub_type_downloader_map[ModelSource::kLocal] = DownloadLocalModel;
  auto rc = InitializeModelManifestRegistry(manifest_file);
  if (rc != Status::kOk) {
    throw OasException("Failed to initialize model manifest");
  }
  rc = LoadModelsFromDisk(downloaded_models_path0);
  if (rc != Status::kOk) {
    throw OasException("Failed to load models from the disk");
  }
}

ModelManager::ModelRunner* ModelManager::GetModelRunner(const std::string& model_id) {
  return model_registry.GetModelRunner(model_id);
}

bool ModelManager::WasModelDownloaded(const std::string& model_id) {
  return model_registry.WasModelDownloaded(model_id);
}

std::pair<Status, std::string> ModelManager::DownloadModel(const std::string& model_id) {
  auto it = model_manifest_registry.reg.find(model_id);
  if (it == model_manifest_registry.reg.end()) {
    return {Status::kModelNotRecognized, ""};
  }
  std::string dest_folder = downloaded_models_path + "/" + model_id;
  if (model_registry.WasModelDownloaded(model_id)) {
    return {Status::kModelAlreadyDownloaded, ""};
  }
  auto callback = [](const std::string&) {};
  auto& manifest = it->second;
  DownloadRequest dreq{model_id, dest_folder, manifest.base_path, manifest.include_filter, callback};
  auto dresult = model_hub_type_downloader_map.at(manifest.model_source)(dreq);
  Status rc = Status::kOk;
  std::string err_str{};
  if (dresult.failures.empty()) {
    model_registry.AddModelInfo(model_id, dest_folder);
  } else {
    rc = Status::kFail;
    for (auto& ferror : dresult.failures) {
      err_str += ferror + "\n";
    }
  }
  return {rc, err_str};
}

Status ModelManager::LoadModelsFromDisk(const std::string& downloaded_models_path) {
  spdlog::info("Loading info for models that were downloaded before");
  for (auto const& dir_entry : fs::directory_iterator{fs::path(downloaded_models_path)}) {
    fs::path fs_model_path = dir_entry.path();
    model_registry.AddModelInfo(fs_model_path.filename().string(), fs_model_path.string());
  }
  spdlog::info("Loaded info for [{}] models", model_registry.model_info_registry.size());
  return Status::kOk;
}

Status ModelManager::LoadModelImpl(const std::string& model_path, ModelRunner& model_runner) {
  model_runner.oga_model = OgaModel::Create(model_path.c_str());
  if (!model_runner.oga_model) {
    spdlog::error("could not create model for [{}]", model_path);
    return Status::kFail;
  }
  model_runner.oga_tokenizer = OgaTokenizer::Create(*model_runner.oga_model);
  if (!model_runner.oga_tokenizer) {
    spdlog::error("could not create tokenizer for [{}]", model_path);
    return Status::kFail;
  }
  model_runner.oga_tokenizer_stream = OgaTokenizerStream::Create(*model_runner.oga_tokenizer);
  if (!model_runner.oga_tokenizer_stream) {
    spdlog::error("could not create tokenizer stream for [{}]", model_path);
    return Status::kFail;
  }
  spdlog::info("Model [{}] loaded successfully", model_path);
  return Status::kOk;
}

Status ModelManager::LoadModel(const std::string& model_id) {
  if (!model_registry.WasModelDownloaded(model_id)) {
    spdlog::error("Model [{}] was not pulled before.", model_id);
    return Status::kModelNotDownloaded;
  }
  if (model_registry.GetModelRunner(model_id)) {
    return Status::kModelAlreadyLoaded;
  }
  ModelRunner model_runner;
  auto rc = LoadModelImpl(model_registry.GetModelPath(model_id), model_runner);
  if (rc != Status::kOk) {
    spdlog::error("Loading model [{}] failed", model_id);
    return rc;
  }
  model_registry.AddModelRunner(model_id, std::move(model_runner));
  return Status::kOk;
}

Status ModelManager::InitializeModelManifestRegistry(const std::string& mf_file) {
  spdlog::info("Reading manifest file [{}]", mf_file);
  std::ifstream f(mf_file);
  if (!f.good()) {
    spdlog::error("Could not read file [{}]", mf_file);
    return Status::kFail;
  }
  json obj = json::parse(f);
  for (auto& model : obj["models"]) {
    ModelManifest mf;
    mf.model_id = model["model_id"].get<std::string>();
    mf.base_path = model["base_path"].get<std::string>();
    auto hub_type = model["model_source"].get<std::string>();
    mf.model_source = ModelSource::kUnknown;
    if (hub_type == "HuggingFace") {
      mf.model_source = ModelSource::kHuggingFace;
    } else if (hub_type == "Local") {
      mf.model_source = ModelSource::kLocal;
    }
    mf.include_filter = ContainsJsonKey(model, "include_filter") ? model["include_filter"].get<std::string>() : "";
    model_manifest_registry.reg[mf.model_id] = mf;
  }
  spdlog::info("Read manifest for [{}] models", model_manifest_registry.reg.size());
  return Status::kOk;
}

}  // namespace oas