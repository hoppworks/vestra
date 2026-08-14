// End-to-end differential harness for the pinned PR #2 C API. The caller
// supplies Vestra's already-decoded RGB24 PPM frames, so video decoding and
// frame selection cannot contaminate a Rust/C++ geometry comparison.
#include "da_capi.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
constexpr char kMagic[4] = {'C', 'P', 'S', '1'};
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
    if (argc != 8) {
        std::cerr << "usage: " << argv[0]
                  << " MODEL.gguf FRAME_DIRECTORY OUTPUT.cps THREADS CHUNK OVERLAP FUSE(0|1)\n";
        return 2;
    }
    int threads = 0, chunk = 0, overlap = 0, fuse = 0;
    if (!parse_positive(argv[4], threads) || !parse_positive(argv[5], chunk) ||
        !parse_positive(argv[6], overlap)) {
        std::cerr << "threads, chunk, and overlap must be positive integers\n";
        return 2;
    }
    try { fuse = std::stoi(argv[7]); } catch (...) { return 2; }
    if (fuse != 0 && fuse != 1) {
        std::cerr << "fuse must be 0 or 1\n";
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
    std::vector<const char*> raw_paths;
    raw_paths.reserve(paths.size());
    for (const auto& path : paths) raw_paths.push_back(path.c_str());

    da_ctx* context = da_capi_load(argv[1], threads);
    if (!context) {
        std::cerr << "could not load model\n";
        return 1;
    }
    int points = 0;
    std::vector<int> counts(paths.size());
    float* xyz = nullptr;
    unsigned char* rgb = nullptr;
    float* radius = nullptr;
    const int result = da_capi_points_stream(
        context, raw_paths.data(), static_cast<int>(raw_paths.size()), chunk, overlap,
        55.0, 1.0f, 0, 1, 1, fuse, 0, 0.0, &points, counts.data(), &xyz, &rgb, &radius);
    if (result != 0 || points < 0 || !xyz || !rgb || !radius) {
        std::cerr << "points stream failed: " << da_capi_last_error(context) << '\n';
        da_capi_free_floats(xyz);
        da_capi_free_bytes(rgb);
        da_capi_free_floats(radius);
        da_capi_free(context);
        return 1;
    }

    float* position = nullptr;
    float* forward = nullptr;
    int pose_frames = 0;
    const int pose_result = da_capi_stream_last_poses(context, &position, &forward, &pose_frames);
    std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
    const std::uint32_t frame_count = static_cast<std::uint32_t>(paths.size());
    const std::uint32_t point_count = static_cast<std::uint32_t>(points);
    const std::uint32_t recorded_pose_frames = pose_result == 0 ? static_cast<std::uint32_t>(pose_frames) : 0;
    const bool wrote = output.write(kMagic, sizeof(kMagic)) && write_exact(output, kVersion) &&
        write_exact(output, frame_count) && write_exact(output, point_count) &&
        write_exact(output, recorded_pose_frames) && write_vector(output, counts) &&
        write_vector(output, std::vector<float>(xyz, xyz + static_cast<size_t>(points) * 3)) &&
        static_cast<bool>(output.write(reinterpret_cast<const char*>(rgb), static_cast<std::streamsize>(points) * 3)) &&
        write_vector(output, std::vector<float>(radius, radius + points)) &&
        (recorded_pose_frames == 0 ||
         (write_vector(output, std::vector<float>(position, position + static_cast<size_t>(pose_frames) * 3)) &&
          write_vector(output, std::vector<float>(forward, forward + static_cast<size_t>(pose_frames) * 3))));
    da_capi_free_floats(position);
    da_capi_free_floats(forward);
    da_capi_free_floats(xyz);
    da_capi_free_bytes(rgb);
    da_capi_free_floats(radius);
    da_capi_free(context);
    if (!wrote) {
        std::cerr << "could not write CPS1 output\n";
        return 1;
    }
    std::cout << "{\"schema\":\"vestra.cpp-pr2-capi-stream/v1\",\"frames\":" << frame_count
              << ",\"points\":" << point_count << ",\"pose_frames\":" << recorded_pose_frames << "}\n";
}
