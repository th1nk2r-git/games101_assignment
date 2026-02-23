// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>

rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f>& positions) {
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return { id };
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i>& indices) {
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return { id };
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f>& cols) {
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return { id };
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f) {
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(float x, float y, const Vector3f* _v) {
    Vector2f p(x, y);
    Vector2f p0(_v[0].x(), _v[0].y());
    Vector2f p1(_v[1].x(), _v[1].y());
    Vector2f p2(_v[2].x(), _v[2].y());

    Vector2f v0 = p0 - p;
    Vector2f v1 = p1 - p;
    Vector2f v2 = p2 - p;
    float area1 = (fabs(v0.cross(v1)) + fabs(v1.cross(v2)) + fabs(v2.cross(v0))) / 2;

    Vector2f v3 = p1 - p0;
    Vector2f v4 = p2 - p0;
    float area2 = fabs(v3.cross(v4)) / 2;
    return fabs(area1 - area2) < 1e-2;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v) {
    float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) / (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() - v[2].x() * v[1].y());
    float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) / (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() - v[0].x() * v[2].y());
    float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) / (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() - v[1].x() * v[0].y());
    return { c1,c2,c3 };
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type) {
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind) {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto& vert : v) {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i) {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();

    float x_min = std::numeric_limits<float>::infinity();
    float x_max = -std::numeric_limits<float>::infinity();
    float y_min = std::numeric_limits<float>::infinity();
    float y_max = -std::numeric_limits<float>::infinity();

    for (auto& vec : t.v) {
        x_min = fmin(x_min, vec.x());
        x_max = fmax(x_max, vec.x());
        y_min = fmin(y_min, vec.y());
        y_max = fmax(y_max, vec.y());
    }

    float offset[16][2] = {
        {0.125f, 0.125f}, {0.125f, 0.375f}, {0.125f, 0.625f}, {0.125f, 0.875f},
        {0.375f, 0.125f}, {0.375f, 0.375f}, {0.375f, 0.625f}, {0.375f, 0.875f},
        {0.625f, 0.125f}, {0.625f, 0.375f}, {0.625f, 0.625f}, {0.625f, 0.875f},
        {0.875f, 0.125f}, {0.875f, 0.375f}, {0.875f, 0.625f}, {0.875f, 0.875f}
    };

    for (int x = floor(x_min); x <= ceil(x_max); x++) {
        for (int y = floor(y_min); y <= ceil(y_max); y++) {
            for (int i = 0; i < 16; i++) {
                float dx = x + offset[i][0], dy = y + offset[i][1];
                if (insideTriangle(dx, dy, t.v)) {
                    auto [alpha, beta, gamma] = computeBarycentric2D(dx, dy, t.v);
                    float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;

                    int index = get_sample_index(x, y, i);
                    if (z_interpolated < sample_depth_buf[index]) {
                        sample_depth_buf[index] = z_interpolated;
                        sample_frame_buf[index] = t.getColor();
                    }
                }
            }

            Eigen::Vector3f averaged_color(0, 0, 0);
            for (int i = 0; i < 16; i++) {
                int index = get_sample_index(x, y, i);
                averaged_color += sample_frame_buf[index];
            }
            averaged_color /= 16.0;

            int index = get_pixel_index(x, y);
            frame_buf[index] = averaged_color;
        }
    }
    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle

    // If so, use the following code to get the interpolated z value.
    //auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
    //float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    //float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    //z_interpolated *= w_reciprocal;

    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m) {
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v) {
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p) {
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff) {
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color) {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f { 0, 0, 0 });
        std::fill(sample_frame_buf.begin(), sample_frame_buf.end(), Eigen::Vector3f(0, 0, 0));
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth) {
        std::fill(sample_depth_buf.begin(), sample_depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h) {
    frame_buf.resize(w * h);
    sample_frame_buf.resize(w * h * 16);
    sample_depth_buf.resize(w * h * 16);
}

int rst::rasterizer::get_pixel_index(const int& x, const int& y) {
    return (height - 1 - y) * width + x;
}

int rst::rasterizer::get_sample_index(const int& x, const int& y, const int& id) {
    return id * (height * width) + (height - 1 - y) * width + x;
}

void rst::rasterizer::set_pixel(const int& index, const Eigen::Vector3f& color) {
    frame_buf[index] = color;
}

// clang-format on