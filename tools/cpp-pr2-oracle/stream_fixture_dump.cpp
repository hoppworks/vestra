// Model-free oracle for the exact PR #2 streaming stitcher.  It reads
// precomputed window-scoped DA3 outputs and invokes da::stream_points_core, so a
// Vestra fixture can distinguish geometry/stitching differences from inference
// differences.  The binary format is documented in README.md next to this file.
#include "stream.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kFixtureVersion = 2;
constexpr std::uint32_t kOutputVersion = 1;
constexpr char kFixtureMagic[4] = {'V', 'P', 'S', '1'};
constexpr char kOutputMagic[4] = {'V', 'P', 'O', '1'};

template <typename T>
bool read_exact(std::istream& in, T& value) {
    return static_cast<bool>(in.read(reinterpret_cast<char*>(&value), sizeof(T)));
}

template <typename T>
bool read_vector(std::istream& in, std::vector<T>& values, std::size_t count) {
    if (count > values.max_size()) return false;
    values.resize(count);
    return count == 0 || static_cast<bool>(in.read(reinterpret_cast<char*>(values.data()),
                                                    static_cast<std::streamsize>(count * sizeof(T))));
}

template <typename T>
bool write_exact(std::ostream& out, const T& value) {
    return static_cast<bool>(out.write(reinterpret_cast<const char*>(&value), sizeof(T)));
}

template <typename T>
bool write_vector(std::ostream& out, const std::vector<T>& values) {
    return values.empty() || static_cast<bool>(out.write(reinterpret_cast<const char*>(values.data()),
                                                          static_cast<std::streamsize>(values.size() * sizeof(T))));
}

struct FixtureFrame {
    da::ViewResult view;
    std::vector<std::uint8_t> rgb;
};

bool checked_plane(std::uint32_t h, std::uint32_t w, std::size_t& plane) {
    if (h == 0 || w == 0) return false;
    const auto hh = static_cast<std::size_t>(h);
    const auto ww = static_cast<std::size_t>(w);
    if (hh > std::numeric_limits<std::size_t>::max() / ww) return false;
    plane = hh * ww;
    return plane <= std::numeric_limits<std::size_t>::max() / 3;
}

bool read_fixture(const std::string& path, std::uint32_t& frames, std::uint32_t& h, std::uint32_t& w,
                  da::StreamParams& params, std::vector<std::vector<FixtureFrame>>& output, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { error = "cannot open fixture"; return false; }
    char magic[4]{};
    std::uint32_t version = 0;
    std::uint32_t chunk = 0, overlap = 0, min_overlap = 0, window_count = 0;
    double conf_pct = 0;
    float point_size = 0;
    if (!in.read(magic, 4) || std::string(magic, 4) != std::string(kFixtureMagic, 4) ||
        !read_exact(in, version) || version != kFixtureVersion ||
        !read_exact(in, frames) || !read_exact(in, h) || !read_exact(in, w) ||
        !read_exact(in, chunk) || !read_exact(in, overlap) || !read_exact(in, conf_pct) ||
        !read_exact(in, point_size) || !read_exact(in, min_overlap) || !read_exact(in, window_count)) {
        error = "invalid VPS1 header"; return false;
    }
    std::size_t plane = 0;
    if (frames == 0 || !checked_plane(h, w, plane) || chunk < 2 || overlap >= chunk ||
        !std::isfinite(conf_pct) || !std::isfinite(point_size)) {
        error = "invalid VPS1 dimensions or parameters"; return false;
    }
    params.chunk_size = static_cast<int>(chunk);
    params.overlap = static_cast<int>(overlap);
    params.conf_pct = conf_pct;
    params.point_size = point_size;
    params.min_overlap_pts = static_cast<int>(min_overlap);
    params.global_budget = 0;
    params.icp_refine = false;
    params.loop_close = false;
    const std::uint32_t step = chunk - overlap;
    std::uint32_t expected_windows = 0;
    for (std::uint32_t w0 = 0; w0 < frames; w0 += step) {
        ++expected_windows;
        if (w0 + chunk >= frames) break;
    }
    if (window_count != expected_windows) { error = "VPS1 window count does not match schedule"; return false; }
    output.resize(window_count);
    for (std::uint32_t window = 0; window < window_count; ++window) {
        std::uint32_t view_count = 0;
        if (!read_exact(in, view_count)) { error = "truncated VPS1 window header"; return false; }
        const std::uint32_t w0 = window * step;
        const std::uint32_t expected_views = std::min(chunk, frames - w0);
        if (view_count != expected_views) { error = "VPS1 view count does not match schedule"; return false; }
        std::vector<FixtureFrame>& views = output[window];
        views.resize(view_count);
        for (FixtureFrame& frame : views) {
            if (!in.read(reinterpret_cast<char*>(frame.view.intr.data()), sizeof(float) * frame.view.intr.size()) ||
                !in.read(reinterpret_cast<char*>(frame.view.ext.data()), sizeof(float) * frame.view.ext.size()) ||
                !read_vector(in, frame.view.depth, plane) || !read_vector(in, frame.view.conf, plane) ||
                !read_vector(in, frame.rgb, plane * 3)) {
                error = "truncated VPS1 view payload"; return false;
            }
        }
    }
    if (in.peek() != std::char_traits<char>::eof()) { error = "trailing VPS1 bytes"; return false; }
    return true;
}

bool write_output(const std::string& path, const da::StreamCloud& cloud, std::uint32_t frames,
                  std::uint32_t h, std::uint32_t w, std::string& error) {
    const std::size_t points = cloud.xyz.size() / 3;
    if (cloud.xyz.size() % 3 != 0 || cloud.rgb.size() != points * 3 || cloud.radius.size() != points ||
        cloud.counts.size() != frames || cloud.window_pos.size() % 3 != 0 ||
        cloud.window_mid_frame.size() != cloud.window_pos.size() / 3 || cloud.frame_pos.size() != frames * 3 ||
        cloud.frame_fwd.size() != frames * 3) {
        error = "invalid StreamCloud shape"; return false;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) { error = "cannot create oracle output"; return false; }
    const std::uint32_t point_count = static_cast<std::uint32_t>(points);
    const std::uint32_t window_count = static_cast<std::uint32_t>(cloud.window_mid_frame.size());
    const std::int32_t warnings = cloud.warnings;
    const std::int32_t loops = cloud.loops_found;
    if (!out.write(kOutputMagic, 4) || !write_exact(out, kOutputVersion) || !write_exact(out, frames) ||
        !write_exact(out, h) || !write_exact(out, w) || !write_exact(out, point_count) ||
        !write_exact(out, window_count) || !write_exact(out, warnings) || !write_exact(out, loops) ||
        !write_exact(out, cloud.metric_scale) || !write_vector(out, cloud.xyz) || !write_vector(out, cloud.rgb) ||
        !write_vector(out, cloud.radius) || !write_vector(out, cloud.counts) ||
        !write_vector(out, cloud.window_pos) || !write_vector(out, cloud.window_mid_frame) ||
        !write_vector(out, cloud.frame_pos) || !write_vector(out, cloud.frame_fwd)) {
        error = "failed writing VPO1 output"; return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: vestra_cpp_stream_fixture_dump INPUT.vps OUTPUT.vpo\n";
        return 64;
    }
    std::uint32_t frames = 0, h = 0, w = 0;
    da::StreamParams params;
    std::vector<std::vector<FixtureFrame>> fixture;
    std::string error;
    if (!read_fixture(argv[1], frames, h, w, params, fixture, error)) {
        std::cerr << "fixture error: " << error << '\n'; return 65;
    }
    const da::WindowSource source = [&](int w0, int w1, std::vector<da::ViewResult>& views,
                                         std::vector<std::vector<std::uint8_t>>& rgb,
                                         int& source_h, int& source_w, std::string&) {
        views.clear(); rgb.clear();
        source_h = static_cast<int>(h); source_w = static_cast<int>(w);
        const int step = params.chunk_size - params.overlap;
        const int window = w0 / step;
        const std::vector<FixtureFrame>& supplied = fixture.at(static_cast<std::size_t>(window));
        if (supplied.size() != static_cast<std::size_t>(w1 - w0)) return false;
        for (const FixtureFrame& frame : supplied) {
            views.push_back(frame.view);
            rgb.push_back(frame.rgb);
        }
        return true;
    };
    da::StreamCloud cloud;
    if (!da::stream_points_core(static_cast<int>(frames), params, source, cloud, error)) {
        std::cerr << "stream error: " << error << '\n'; return 66;
    }
    if (!write_output(argv[2], cloud, frames, h, w, error)) {
        std::cerr << "output error: " << error << '\n'; return 67;
    }
    std::cout << "VPO1 points=" << cloud.radius.size() << " windows=" << cloud.window_mid_frame.size()
              << " warnings=" << cloud.warnings << " loops=" << cloud.loops_found << '\n';
    return 0;
}
