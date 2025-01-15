#ifndef PLOTTING_H
#define PLOTTING_H

#include <iostream>
#include <fstream>
#include <gnuplot-iostream.h>

void plotPeriodicGrid(float *periodic_grid, int N, int M) {
    std::ofstream data_file_x("periodic_grid_data_x.dat");
    std::ofstream data_file_y("periodic_grid_data_y.dat");
    for (int y_i = 0; y_i < M; ++y_i) {
        for (int i = 0; i < 2 * N; i += 2) {
            float x = periodic_grid[y_i * (2 * N) + i - 1];
            float y = periodic_grid[y_i * (2 * N) + i];
            data_file_x << i / 2 << " " << y_i << " " << x << "\n";
            data_file_y << i / 2 << " " << y_i << " " << y << "\n";
        }
        data_file_x << "\n";
        data_file_y << "\n";
    }
    data_file_x.close();
    data_file_y.close();

    Gnuplot gp;
    gp << "set terminal png size 800,600\n"; // Use PNG terminal with specified size
    gp << "set xrange [0:" << N-1 << "]\n"; // Set x-axis range
    gp << "set yrange [0:" << M-1 << "]\n"; // Set y-axis range
    gp << "set view map \n";
    gp << "set pm3d at b\n"; 
    gp << "set output 'periodic_grid_plot_x.png'\n"; // Output file
    gp << "plot 'periodic_grid_data_x.dat' using 1:2:3 with image\n";

    gp << "set output 'periodic_grid_plot_y.png'\n"; // Output file
    gp << "plot 'periodic_grid_data_y.dat' using 1:2:3 with image\n";
}

void plotVelocityGrid(float *periodic_grid, float *velocity_grid, int N, int M,float periodic_start, float periodic_end, std::string plot_name) {
    std::ofstream data_file(plot_name + ".dat");
    for (int y_i = 0; y_i < M; ++y_i) {
        for (int i = 0; i < 2 * N; i += 2) {
            float x = periodic_grid[y_i * (2 * N) + i];
            float y = periodic_grid[y_i * (2 * N) + i + 1];
            float u = velocity_grid[y_i * (2 * N) + i];
            float v = velocity_grid[y_i * (2 * N) + i + 1];
            data_file << x << " " << y << " " << u << " " << v << "\n";
        }
        data_file << "\n";
    }
    data_file.close();

    Gnuplot gp;
    gp << "set terminal png size 800,600\n"; // Use PNG terminal with specified size
    gp << "set xrange ["<<periodic_start<<":" << periodic_end << "]\n"; // Set x-axis range
    gp << "set yrange ["<<periodic_start<<":" << periodic_end << "]\n"; // Set y-axis range
    gp << "set output '"<< plot_name << ".png'\n"; // Output file
    gp << "plot '"<< plot_name <<".dat' using 1:2:3:4 with vectors head filled lt 2\n";
}

#endif // PLOTTING_H