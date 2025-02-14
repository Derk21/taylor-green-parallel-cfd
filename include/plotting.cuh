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

/**
 * @brief plot periodic grid values for debugging
 * 
 * @param periodic_grid 
 * @param n 
 * @param m 
 */
void plotPeriodicGrid(double *periodic_grid, int n, int m);

/**
 * @brief outputs .dat of velocity data and .png of plotted vectors
 * 
 * @param periodic_grid  
 * @param velocity_grid 
 * @param n 
 * @param m 
 * @param periodic_start 
 * @param periodic_end 
 * @param plotname name of file
 * @param dirName output dir
 */
void plotVelocityGrid(double *periodic_grid, double *velocity_grid, int n, int m, double periodic_start, double periodic_end, const std::string &plotname, const std::string &dirName);

/**
 * @brief Create a Timestamped Directory object
 * 
 * @return std::string with directory path
 */
std::string createTimestampedDirectory();

/**
 * @brief Create a velocity gif from pngs in directory 
 * 
 * @param dirName output directory
 * @param outputGif gif name
 * @param periodic_start 
 * @param periodic_end 
 */
void createGifFromPngs(const std::string &dirName, const std::string &outputGif, double periodic_start, double periodic_end);

/**
 * @brief plots rsme and relative error into simulation directory
 * 
 * @param reference directory 
 * @param simulation directory (also where plot is saved)
 */
void plotErrors(const std::string &reference, const std::string &simulation);

/**
 * @brief read .dat file to double vector
 * 
 * @param file_path 
 * @return std::vector<double> with all values 
 */
std::vector<double> readDataFile(const std::string &file_path);
#endif // PLOTTING_H