// Deterministic, model-free closed-trajectory fixture for the PR #2 stream
// oracle. The camera/raycast/noise recipe is intentionally copied from the
// pinned test_stream_loop.cpp, but this tool emits recorded ViewResults so the
// Rust and C++ geometry pipelines receive exactly the same evidence.
#include "synth_scene.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kFixtureVersion = 3;
constexpr std::uint32_t kBranchIcpRefine = 1u << 0;
constexpr std::uint32_t kBranchLoopClose = 1u << 1;

template <typename T>
bool write_exact(std::ostream& out, const T& value) {
    return static_cast<bool>(out.write(reinterpret_cast<const char*>(&value), sizeof(T)));
}

template <typename T>
bool write_vector(std::ostream& out, const std::vector<T>& values) {
    return values.empty() || static_cast<bool>(out.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T))));
}

// Apply the exact overlap-only drift from PR #2's test_stream_loop.cpp.
static void perturb_ext(std::array<float, 12>& ext, const double rt[9], const double tt[3]) {
    double r[9] = {ext[0], ext[1], ext[2], ext[4], ext[5], ext[6], ext[8], ext[9], ext[10]};
    double t[3] = {ext[3], ext[7], ext[11]};
    double c[3] = {-(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
                   -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
                   -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2])};
    double cn[3] = {rt[0] * c[0] + rt[1] * c[1] + rt[2] * c[2] + tt[0],
                    rt[3] * c[0] + rt[4] * c[1] + rt[5] * c[2] + tt[1],
                    rt[6] * c[0] + rt[7] * c[1] + rt[8] * c[2] + tt[2]};
    double rn[9];
    for (int row = 0; row < 3; ++row) for (int col = 0; col < 3; ++col)
        rn[row * 3 + col] = r[row * 3] * rt[col * 3] + r[row * 3 + 1] * rt[col * 3 + 1]
                          + r[row * 3 + 2] * rt[col * 3 + 2];
    double tn[3] = {-(rn[0] * cn[0] + rn[1] * cn[1] + rn[2] * cn[2]),
                    -(rn[3] * cn[0] + rn[4] * cn[1] + rn[5] * cn[2]),
                    -(rn[6] * cn[0] + rn[7] * cn[1] + rn[8] * cn[2])};
    ext = {(float)rn[0], (float)rn[1], (float)rn[2], (float)tn[0],
           (float)rn[3], (float)rn[4], (float)rn[5], (float)tn[1],
           (float)rn[6], (float)rn[7], (float)rn[8], (float)tn[2]};
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: vestra_cpp_closed_loop_fixture OUTPUT.vps [--icp-refine]\n";
        return 64;
    }
    const bool icp_refine = argc == 3 && std::string(argv[2]) == "--icp-refine";
    if (argc == 3 && !icp_refine) {
        std::cerr << "unknown option: " << argv[2] << '\n';
        return 64;
    }
    constexpr int width = 160, height = 120, frame_count = 60, orbit_period = 48;
    constexpr int chunk_size = 12, overlap = 4;
    constexpr double fx = 140, fy = 140, two_pi = 6.283185307179586;
    constexpr double depth_noise = 0.005;
    const double yaw = 0.6 * 3.14159265358979323846 / 180.0;
    const double rt[9] = {std::cos(yaw), -std::sin(yaw), 0,
                          std::sin(yaw),  std::cos(yaw), 0,
                          0,              0,             1};
    const double tt[3] = {0.02, 0.0, 0.0};
    const int step = chunk_size - overlap;

    std::vector<da::synth::Camera> cameras;
    cameras.reserve(frame_count);
    for (int frame = 0; frame < frame_count; ++frame) {
        const double a = two_pi * frame / orbit_period;
        const da::synth::V3 eye{2.5 * std::cos(a), 2.5 * std::sin(a), 1.3};
        const da::synth::V3 target{5.0 * std::cos(a), 5.0 * std::sin(a), 1.04};
        cameras.push_back(da::synth::look_at(eye, target, {0, 0, 1}, fx, fy, width, height));
    }
    const da::synth::Scene room = da::synth::make_room(4.0, 3.0);
    std::ofstream out(argv[1], std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "cannot create fixture\n";
        return 65;
    }
    const std::uint32_t branch_flags = kBranchLoopClose | (icp_refine ? kBranchIcpRefine : 0);
    const std::uint32_t window_count = 7;
    const double confidence_percentile = 20.0;
    const float point_size = 1.0f;
    const std::uint32_t minimum_overlap_points = 50;
    if (!out.write("VPS1", 4) || !write_exact(out, kFixtureVersion) ||
        !write_exact(out, static_cast<std::uint32_t>(frame_count)) ||
        !write_exact(out, static_cast<std::uint32_t>(height)) ||
        !write_exact(out, static_cast<std::uint32_t>(width)) ||
        !write_exact(out, static_cast<std::uint32_t>(chunk_size)) ||
        !write_exact(out, static_cast<std::uint32_t>(overlap)) ||
        !write_exact(out, confidence_percentile) || !write_exact(out, point_size) ||
        !write_exact(out, minimum_overlap_points) || !write_exact(out, branch_flags) ||
        !write_exact(out, window_count)) {
        std::cerr << "cannot write fixture header\n";
        return 66;
    }
    for (int start = 0, window = 0; start < frame_count; start += step, ++window) {
        const int views = std::min(chunk_size, frame_count - start);
        if (!write_exact(out, static_cast<std::uint32_t>(views))) return 66;
        for (int local = 0; local < views; ++local) {
            const int frame = start + local;
            da::synth::View view = da::synth::render(
                room, cameras[frame], depth_noise,
                static_cast<unsigned long long>(1000 * (window + 1) + frame + 1));
            if (window > 0 && local < overlap) perturb_ext(view.ext, rt, tt);
            std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3, 128);
            if (!write_vector(out, std::vector<float>(view.intr.begin(), view.intr.end())) ||
                !write_vector(out, std::vector<float>(view.ext.begin(), view.ext.end())) ||
                !write_vector(out, view.depth) || !write_vector(out, view.conf) ||
                !write_vector(out, rgb)) {
                std::cerr << "cannot write fixture payload\n";
                return 66;
            }
        }
        if (start + chunk_size >= frame_count) break;
    }
    std::cout << "VPS1 closed-loop frames=" << frame_count << " windows=" << window_count
              << " icp=" << (icp_refine ? "on" : "off") << " loop=on\n";
    return 0;
}
