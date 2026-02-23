#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos) {
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1, -eye_pos[2],
        0, 0, 0, 1;

    view = translate * view;

    return view;
}

// 作业1: 构造绕z轴旋转某个弧度的变换矩阵
Eigen::Matrix4f get_model_matrix(float rotation_angle) {
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f rotation;
    
    float angle = rotation_angle / 180 * MY_PI;

    rotation << cos(angle), -sin(angle), 0, 0,
                sin(angle),  cos(angle), 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;

    model = rotation * model;

    return model;
}

// 作业1: 根据视锥体参数构造透视矩阵
Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar) {
    Eigen::Matrix4f squeeze = Eigen::Matrix4f::Identity();

    squeeze(0, 0) = zNear;
    squeeze(1, 1) = zNear;
    squeeze(2, 2) = zNear + zFar;
    squeeze(2, 3) = zNear * zFar * (-1);
    squeeze(3, 2) = 1;
    squeeze(3, 3) = 0;

    float top = tan(eye_fov / 2) * fabs(zNear);
    float bottom = -top;
    float right = top * aspect_ratio;
    float left = -right;

    Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f scale = Eigen::Matrix4f::Identity();
    scale(0, 0) = 2.0 / (right - left);
    scale(1, 1) = 2.0 / (top - bottom);
    scale(2, 2) = 2.0 / (zNear - zFar);

    Eigen::Matrix4f projection = scale * translation * squeeze;

    return projection;
}

// 提高项: 构造绕任意轴旋转的变换矩阵
Eigen::Matrix4f get_rotation(Vector3f axis, float angle) {
    float radian = angle / 180 * MY_PI;
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f M;
    Eigen::Matrix3f Rk;
    Rk << 0, -axis[2], axis[1],
        axis[2], 0, -axis[0],
        -axis[1], axis[0], 0;
    
    M = I + (1 - cos(radian)) * Rk * Rk + sin(radian) * Rk;

    model << M(0, 0), M(0, 1), M(0, 2), 0,
        M(1, 0), M(1, 1), M(1, 2), 0,
        M(2, 0), M(2, 1), M(2, 2), 0,
        0, 0, 0, 1;
    return model;
}

int main(int argc, const char** argv) {
    float angle = 0;
    Vector3f axis(0, 0, 1);
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = { 0, 0, 5 };

    std::vector<Eigen::Vector3f> pos {
        {2, 0, -2},
        {0, 2, -2},
        {-2, 0, -2}
    };

    std::vector<Eigen::Vector3i> ind {
        {0, 1, 2}
    };

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_rotation(axis, angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        }
        else if (key == 'd') {
            angle -= 10;
        }
    }

    return 0;
}
