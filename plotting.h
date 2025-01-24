#pragma once
#include <iostream>
#include <fstream>
#include <gnuplot-iostream.h>
#include <string>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <vector>
#include "utils.h"

void plotPeriodicGrid(double *periodic_grid, int n, int m) {
    std::ofstream data_file_x("periodic_grid_data_x.dat");
    std::ofstream data_file_y("periodic_grid_data_y.dat");
    for (int y_i = 0; y_i < m; ++y_i) {
        for (int i = 1; i < 2 * n; i += 2) {
            int u_i = i - 1;
            int v_i = i;
            double x = periodic_grid[periodic_linear_Idx(u_i, y_i)];
            double y = periodic_grid[periodic_linear_Idx(v_i, y_i)];
            data_file_x << i / 2 << " " << y_i << " " << x << "\n";
            data_file_y << i / 2 << " " << y_i << " " << y << "\n";
        }
        data_file_x << "\n";
        data_file_y << "\n";
    }
    data_file_x.close();
    data_file_y.close();

    Gnuplot gp;
    gp << "set terminal png size 800,800\n"; // Use PNG terminal with specified size
    gp << "set xrange [0:" << n-1 << "]\n"; // Set x-axis range
    gp << "set yrange [0:" << m-1 << "]\n"; // Set y-axis range
    gp << "set view map \n";
    gp << "set pm3d at b\n"; 
    gp << "set output 'periodic_grid_plot_x.png'\n"; // Output file
    gp << "plot 'periodic_grid_data_x.dat' using 1:2:3 with image\n";

    gp << "set output 'periodic_grid_plot_y.png'\n"; // Output file
    gp << "plot 'periodic_grid_data_y.dat' using 1:2:3 with image\n";
}

void plotVelocityGrid(
    double *periodic_grid, 
    double *velocity_grid,
    int n,
    int m,
    double periodic_start,
    double periodic_end,
    const std::string& plotname,
    const std::string& dirName)
 {
    std::string data_file_path(std::string(dirName + "/uv_" + plotname + ".dat"));
    std::ofstream data_file(data_file_path);
    double max_magnitude = 0.0;

    // find maximum magnitude
    std::vector<double> magnitudes;
    for (int y_i = 0; y_i < m; ++y_i) {
        for (int i = 1; i < 2 * n; i += 2) {
            int u_i = i - 1;
            int v_i = i;
            double u = velocity_grid[periodic_linear_Idx(u_i, y_i)];
            double v = velocity_grid[periodic_linear_Idx(v_i, y_i)];
            double magnitude = sqrt(u * u + v * v);
            magnitudes.push_back(magnitude);
            if (magnitude > max_magnitude) {
                max_magnitude = magnitude;
            }
        }
    }

    // Write normalized vectors and magnitudes to the data file
    for (int y_i = 0; y_i < m; ++y_i) {
        for (int i = 1; i < 2 * n; i += 2) {
            int u_i = i - 1;
            int v_i = i;
            double x = periodic_grid[periodic_linear_Idx(u_i, y_i)];
            double y = periodic_grid[periodic_linear_Idx(v_i, y_i)];
            double u = velocity_grid[periodic_linear_Idx(u_i, y_i)];
            double v = velocity_grid[periodic_linear_Idx(v_i, y_i)];
            double magnitude = magnitudes[(y_i * n) + (i / 2)];
            double norm_u = u / max_magnitude;
            double norm_v = v / max_magnitude;
            data_file << x << " " << y << " " << norm_u << " " << norm_v << " " << magnitude << "\n";
        }
        data_file << "\n";
    }
    data_file.close();

    Gnuplot gp;
    gp << "set terminal png size 800,800\n"; // Use PNG terminal with specified size
    gp << "set xrange [" << periodic_start << ":" << periodic_end << "]\n"; // Set x-axis range
    gp << "set yrange [" << periodic_start << ":" << periodic_end << "]\n"; // Set y-axis range
    gp << "set cbrange [0:" << max_magnitude << "]\n"; // Set color range
    gp << "set output '" << dirName << "/" << plotname << ".png'\n"; // Output file
    gp << "plot '"<< data_file_path <<"' using 1:2:3:4:5 with vectors head filled lt palette\n";
}
std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

std::string createTimestampedDirectory() {
    std::string timestamp = getCurrentTimestamp();
    std::string dirName = "plots/" + timestamp;
    std::filesystem::create_directory(dirName);
    return dirName;
}

void createGifFromPngs(const std::string& dirName, const std::string& outputGif, double periodic_start, double periodic_end)
{ 
    std::stringstream command;
    command << "ffmpeg -framerate 10 -i " 
    << dirName << "/velocity_%04d.png -vf \"scale=800:-1:flags=lanczos\" -c:v gif -y " 
    << dirName << "/" << outputGif
    << " -loglevel error";
    
    // Execute the command
    int ret = std::system(command.str().c_str());
    
    if (ret == 0) {
        std::cout << "GIF created successfully.\n";
    } else {
        std::cerr << "Failed to execute FFmpeg command.\n";
    }
}

