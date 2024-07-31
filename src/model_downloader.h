// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <string>
#include <unordered_map>

namespace oas {
struct DownloadResult {
  std::vector<std::string> failures;
};

struct DownloadRequest {
  std::string model_id;
  std::string download_dir;
  std::string base_path;
  std::string include_filter;
  std::function<void(const std::string&)> download_status_callback;
};
using ModelDownloader = std::function<DownloadResult(const DownloadRequest&)>;

DownloadResult DownloadHuggingFaceModel(const DownloadRequest& dreq);
DownloadResult DownloadLocalModel(const DownloadRequest& dreq);

// add other downloaders here
};  // namespace oas