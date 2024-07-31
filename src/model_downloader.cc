// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>
#include <json.hpp>
#include <fstream>
#include <experimental/filesystem>
#include <vector>
#include <thread>
#include <future>
#include <mutex>
#include <functional>
#include <string>
#include <unordered_map>
#include "spdlog/spdlog.h"

#include "model_downloader.h"

using json = nlohmann::json;
namespace fs = std::experimental::filesystem;

namespace oas {
void DownloadFile(const std::string& base_url, const std::string& file_url, const std::string& dest_path,
                  std::mutex& mtx, std::function<void(const std::string&)> download_status_callback,
                  DownloadResult& result) {
  httplib::Client cli(base_url);
  cli.set_follow_location(true);
  cli.set_bearer_token_auth("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");
  cli.set_ca_cert_path("", "/etc/ssl/certs");
  httplib::Result res;
  {
    std::ofstream ofs(dest_path, std::ios::binary | std::ios::app);
    res = cli.Get(file_url.c_str(), [&](const char* data, size_t data_length) {
      ofs.write(data, data_length);
      return true;
    });
  }

  std::lock_guard<std::mutex> lock(mtx);
  if (res && (res->status == 200 || res->status == 302 || res->status == 301)) {
    download_status_callback("Downloaded: " + dest_path);
  } else {
    std::string failure_reason = "Failed to download: " + file_url + " (Status: " + std::to_string(res ? res->status : -1) + ")";
    download_status_callback(failure_reason);
    result.failures.push_back(failure_reason);
  }
}

DownloadResult DownloadLocalModel(const DownloadRequest& dreq) {
  const auto& model_id = dreq.model_id;
  std::error_code ec;
  fs::copy(fs::path(dreq.base_path), fs::path(dreq.download_dir), ec);
  DownloadResult dr;
  if (ec) {
    dr.failures.push_back(ec.message());
    return dr;
  }
  return dr;
}

DownloadResult DownloadHuggingFaceModel(const DownloadRequest& dreq) {
  const auto& model_id = dreq.model_id;
  const auto& download_status_callback = dreq.download_status_callback;
  DownloadResult result;
  httplib::Client cli("https://huggingface.co");
  cli.set_follow_location(true);
  cli.set_ca_cert_path("", "/etc/ssl/certs");

  std::string api_url = "/api/models/" + dreq.base_path;
  auto res = cli.Get(api_url.c_str());

  if (res && res->status == 200) {
    json model_info = json::parse(res->body);
    std::vector<std::string> files_to_download;

    for (const auto& file : model_info["siblings"]) {               // TODO: validate if siblings key is present
      const std::string fn = file["rfilename"].get<std::string>();  // TODO: validate if rfilename is present
      if (!dreq.include_filter.empty() && fn.find(dreq.include_filter) != std::string::npos) {
        files_to_download.push_back(fn);
      }
    }

    std::vector<std::future<void>> futures;
    std::mutex mtx;

    std::string tmp_dir = dreq.download_dir + ".tmp";
    std::error_code ec;
    if (!fs::exists(tmp_dir) && !fs::create_directories(tmp_dir, ec)) {
      std::string failure_reason = "Failed to create temporary directory: " + tmp_dir + " : " + ec.message();
      download_status_callback(failure_reason);
      result.failures.push_back(failure_reason);
      return result;
    }

    for (const auto& file_url : files_to_download) {
      std::string url = "/" + dreq.base_path + "/resolve/main/" + file_url;
      std::string dest_path = tmp_dir + "/" + fs::path(file_url).filename().string();
      ec.clear();
      if (!fs::exists(fs::path(dest_path).parent_path()) && !fs::create_directories(fs::path(dest_path).parent_path(), ec)) {
        std::string failure_reason = "Failed to create directories for: " + dest_path + " : " + ec.message();
        download_status_callback(failure_reason);
        result.failures.push_back(failure_reason);
        continue;
      }

      if (fs::exists(dest_path)) {
        download_status_callback("File already exists: " + dest_path);
        continue;
      }

      futures.push_back(std::async(std::launch::async, DownloadFile, "https://huggingface.co", url, dest_path,
                                   std::ref(mtx), download_status_callback, std::ref(result)));
    }

    for (auto& future : futures) {
      future.get();
    }

    if (result.failures.empty()) {
      ec.clear();
      fs::rename(tmp_dir, dreq.download_dir, ec);
      if (ec) {
        std::string failure_reason = "Failed to rename temporary directory to destination: " + dreq.download_dir + " : " + ec.message();
        download_status_callback(failure_reason);
        result.failures.push_back(failure_reason);
      } else {
        download_status_callback("All files downloaded successfully and moved to destination.");
      }
    } else {
      ec.clear();
      if (!fs::remove_all(tmp_dir, ec)) {
        std::string failure_reason = "Failed to remove temporary directory: " + tmp_dir + " : " + ec.message();
        download_status_callback(failure_reason);
        result.failures.push_back(failure_reason);
      } else {
        download_status_callback("Some files failed to download. Temporary directory has been removed.");
      }
    }
  } else {
    std::string failure_reason = "Failed to retrieve model info: " + api_url + " ";
    failure_reason += "(Status: " + std::to_string(res ? res->status : -1) + ")";
    download_status_callback(failure_reason);
    result.failures.push_back(failure_reason);
  }

  return result;
}  // namespace ops
};  // namespace oas