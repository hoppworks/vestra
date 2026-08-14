// Fair multi-view model-only timing companion for Vestra's Rust benchmark.
// Model loading and PPM decoding happen once before timing; every sample runs
// the pinned PR #2 depth+confidence+pose workload over the same 12/3 windows.
#include "engine.hpp"
#include "image_io.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {
bool positive(const char* value, int& result) {
    try {
        result = std::stoi(value);
        return result > 0;
    } catch (...) {
        return false;
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "usage: " << argv[0]
                  << " MODEL.gguf FRAME_DIRECTORY THREADS CHUNK OVERLAP WARMUP REPEAT\n";
        return 2;
    }
    int threads = 0, chunk = 0, overlap = 0, warmup = 0, repeat = 0;
    if (!positive(argv[3], threads) || !positive(argv[4], chunk) ||
        !positive(argv[5], overlap) || overlap >= chunk || !positive(argv[6], warmup) ||
        !positive(argv[7], repeat)) {
        std::cerr << "positive threads/chunk/overlap/warmup/repeat required; overlap < chunk\n";
        return 2;
    }
    std::vector<std::string> paths;
    for (const auto& entry : std::filesystem::directory_iterator(argv[2])) {
        if (entry.is_regular_file() && entry.path().extension() == ".ppm") {
            paths.push_back(entry.path().string());
        }
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) {
        std::cerr << "no .ppm frames in " << argv[2] << '\n';
        return 2;
    }
    std::vector<da::Image> frames(paths.size());
    for (std::size_t index = 0; index < paths.size(); ++index) {
        if (!da::load_image_rgb(paths[index], frames[index])) {
            std::cerr << "could not load " << paths[index] << '\n';
            return 1;
        }
    }
    std::vector<std::vector<da::Image>> windows;
    const std::size_t step = static_cast<std::size_t>(chunk - overlap);
    for (std::size_t start = 0; start < frames.size(); start += step) {
        const std::size_t end = std::min(frames.size(), start + static_cast<std::size_t>(chunk));
        windows.emplace_back(frames.begin() + static_cast<std::ptrdiff_t>(start),
                             frames.begin() + static_cast<std::ptrdiff_t>(end));
        if (end == frames.size()) break;
    }
    auto engine = da::Engine::load(argv[1], threads);
    if (!engine) {
        std::cerr << "could not load model\n";
        return 1;
    }
    double checksum = 0.0;
    std::vector<double> samples;
    for (int iteration = 0; iteration < warmup + repeat; ++iteration) {
        const auto started = std::chrono::steady_clock::now();
        for (const auto& images : windows) {
            std::vector<da::ViewResult> views;
            int height = 0, width = 0;
            if (!engine->depth_pose_multi(images, views, height, width) || views.empty() ||
                views.front().depth.empty() || views.front().conf.empty()) {
                std::cerr << "depth_pose_multi failed\n";
                return 1;
            }
            checksum += views.front().depth.front() + views.front().conf.front();
        }
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        if (iteration >= warmup) samples.push_back(elapsed);
    }
    std::cout << "{\"schema\":\"vestra.pr2-multiview-model-bench/v1\",\"frames\":"
              << frames.size() << ",\"windows\":" << windows.size() << ",\"samples_ms\":[";
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << samples[index];
    }
    std::cout << "],\"checksum\":" << checksum << "}\n";
}
