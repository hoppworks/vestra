// Records the pinned C++ PR #2 multi-view model boundary. It deliberately
// stops before confidence selection, backprojection, Sim(3), or TSDF fusion.
#include "engine.hpp"
#include "image_io.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
constexpr char kMagic[4] = {'M', 'V', 'O', '1'};
constexpr std::uint32_t kVersion = 1;

template <typename T>
bool write_exact(std::ofstream& output, const T& value) {
    return static_cast<bool>(output.write(reinterpret_cast<const char*>(&value), sizeof(value)));
}

template <typename T>
bool write_vector(std::ofstream& output, const std::vector<T>& values) {
    return values.empty() || static_cast<bool>(output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T))));
}

bool parse_positive(const char* value, int& output) {
    try {
        output = std::stoi(value);
        return output > 0;
    } catch (...) {
        return false;
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: " << argv[0]
                  << " MODEL.gguf FRAME_DIRECTORY OUTPUT.mvo THREADS CHUNK OVERLAP\n";
        return 2;
    }
    int threads = 0, chunk = 0, overlap = 0;
    if (!parse_positive(argv[4], threads) || !parse_positive(argv[5], chunk) ||
        !parse_positive(argv[6], overlap) || overlap >= chunk) {
        std::cerr << "threads/chunk must be positive and overlap must be smaller than chunk\n";
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
    auto engine = da::Engine::load(argv[1], threads);
    if (!engine) {
        std::cerr << "could not load model\n";
        return 1;
    }
    std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
    const std::uint32_t frames = static_cast<std::uint32_t>(paths.size());
    const std::uint32_t chunk_u = static_cast<std::uint32_t>(chunk);
    const std::uint32_t overlap_u = static_cast<std::uint32_t>(overlap);
    std::uint32_t windows = 0;
    for (std::size_t start = 0; start < paths.size(); start += static_cast<std::size_t>(chunk - overlap)) {
        ++windows;
        if (start + static_cast<std::size_t>(chunk) >= paths.size()) break;
    }
    if (!output.write(kMagic, sizeof(kMagic)) || !write_exact(output, kVersion) ||
        !write_exact(output, frames) || !write_exact(output, chunk_u) ||
        !write_exact(output, overlap_u) || !write_exact(output, windows)) {
        std::cerr << "could not write MVO1 header\n";
        return 1;
    }
    const std::size_t step = static_cast<std::size_t>(chunk - overlap);
    for (std::size_t start = 0; start < paths.size(); start += step) {
        const std::size_t end = std::min(paths.size(), start + static_cast<std::size_t>(chunk));
        std::vector<da::Image> images(end - start);
        for (std::size_t index = start; index < end; ++index) {
            if (!da::load_image_rgb(paths[index], images[index - start])) {
                std::cerr << "could not load " << paths[index] << '\n';
                return 1;
            }
        }
        std::vector<da::ViewResult> views;
        int h = 0, w = 0;
        if (!engine->depth_pose_multi(images, views, h, w) || views.size() != images.size() || h <= 0 || w <= 0) {
            std::cerr << "depth_pose_multi failed for window at " << start << '\n';
            return 1;
        }
        const std::uint32_t start_u = static_cast<std::uint32_t>(start);
        const std::uint32_t view_count = static_cast<std::uint32_t>(views.size());
        const std::uint32_t h_u = static_cast<std::uint32_t>(h);
        const std::uint32_t w_u = static_cast<std::uint32_t>(w);
        if (!write_exact(output, start_u) || !write_exact(output, view_count) ||
            !write_exact(output, h_u) || !write_exact(output, w_u)) {
            std::cerr << "could not write MVO1 window header\n";
            return 1;
        }
        const std::size_t pixels = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
        for (const da::ViewResult& view : views) {
            if (view.depth.size() != pixels || view.conf.size() != pixels ||
                !write_vector(output, view.depth) || !write_vector(output, view.conf) ||
                !output.write(reinterpret_cast<const char*>(view.ext.data()), sizeof(float) * view.ext.size()) ||
                !output.write(reinterpret_cast<const char*>(view.intr.data()), sizeof(float) * view.intr.size())) {
                std::cerr << "could not write MVO1 view\n";
                return 1;
            }
        }
        if (end == paths.size()) break;
    }
    if (!output) {
        std::cerr << "could not finalize MVO1 output\n";
        return 1;
    }
    std::cout << "{\"schema\":\"vestra.cpp-pr2-multiview/v1\",\"frames\":" << frames
              << ",\"windows\":" << windows << "}\n";
}
