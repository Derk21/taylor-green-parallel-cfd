#pragma once
#ifndef PLOTTING_H
#define PLOTTING_H

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
#include "utils.cuh"

void plotPeriodicGrid(double *periodic_grid, int n, int m);

void plotVelocityGrid(double *periodic_grid, double *velocity_grid, int n, int m, double periodic_start, double periodic_end, const std::string& plotname, const std::string& dirName);

std::string createTimestampedDirectory();

void createGifFromPngs(const std::string& dirName, const std::string& outputGif, double periodic_start, double periodic_end);

void plotErrors(const std::string& dir1, const std::string& dir2);
#endif // PLOTTING_H