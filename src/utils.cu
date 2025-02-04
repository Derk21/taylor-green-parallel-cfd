#include "utils.cuh"

void print_matrix_row_major(const int &m, const int &n, const double *A, const int &lda) {
    //adapted from https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/utils/cusolver_utils.h
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}
void print_matrix(const int &m, const int &n, const double *A, const int &lda) {
    //is column major (column by column)
    //copied from https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/utils/cusolver_utils.h
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

void switchRowColMajor(double *A, const int &m, const int &n)
{
    //converts A from row major to column major 
    double * temp = (double *)malloc(m * n * sizeof(double));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            temp[j * m + i] = A[i * n + j];
        }
    }
    memcpy(A, temp, m * n * sizeof(double));
    free(temp);
}

__host__ __device__ 
int periodic_linear_Idx(const int &x, const int &y, const int bound_x , const int bound_y )
{   
    int mod_x = (x + bound_x) % bound_x; // ensures non-negative result
    int mod_y = (y + bound_y) % bound_y;
    return mod_y * bound_x + mod_x;
}

void setClosestGridPointIdx(double x, double y, int n, int m, int &closest_x_i, int &closest_y_i)
{
    //sets index to y-value,(v_i) 
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);

    closest_x_i = round((x - PERIODIC_START) / dx);
    closest_y_i = round((y - PERIODIC_START) / dy);

    //periodic boundary conditions
    closest_x_i = (closest_x_i + n) % n;
    closest_y_i = (closest_y_i + m) % m;

    //convert to v_i coordinate
    closest_x_i = closest_x_i * 2 + 1;
}


void taylorGreenGroundTruth(double* periodic_grid,double *velocity_grid_next, int iteration, int n , int m){
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    int nn = 2 * n;
    double t = iteration * TIMESTEP;
    double F = exp(-2.0 * DIFFUSIVITY * t);

    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < nn; i+=2)
        {   
            int u_i = i-1;
            int v_i = i;

            double x = periodic_grid[periodic_linear_Idx(u_i,y_i)];
            double y = periodic_grid[periodic_linear_Idx(v_i,y_i)];

            velocity_grid_next[periodic_linear_Idx(u_i,y_i)] =  sin(x) * cos(y) * F;
            velocity_grid_next[periodic_linear_Idx(v_i,y_i)] = -1.0 * cos(x) * sin(y) * F;
        }
    }
}

void setPressureGroundTruth(double *pressure_grid,double * periodic_grid,int iteration, int n ,int m)
{
    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    double t = iteration * TIMESTEP;
    double F = exp(-2.0 * DIFFUSIVITY * t);
    double rho = 0.1;
    for (int y_i = 0; y_i < m; y_i++)
    {
        for (int i = 1; i < n; i++)
        {

            int u_i = 2 * (i-1);
            int v_i = 2 * i;

            double x = periodic_grid[periodic_linear_Idx(u_i,y_i,n,m)];
            double y = periodic_grid[periodic_linear_Idx(v_i,y_i,n,m)];
            pressure_grid[y_i * n + i] = (rho / 4 )* (cos(2*x)+cos(2*y))*pow(F,2); 
        }
    }
}

double get_max_diff(double* a, double* b, int n)
{
    //mean reduction
    double maximum = 0.0;
    for (int i = 0; i < n; i++)
    {
        maximum = max(abs(a[i]-b[i]),maximum);
    }
    return maximum;
}

bool is_close(const double &a, const double &b, const double &tolerance) 
{
    return std::fabs(a - b) < tolerance;
}

void clip(double &v,const double min, const double max)
{
    v = std::max(min,std::min(v,max));
}

double calculateRMSE(const std::vector<double>& reference, const std::vector<double>& simulation)
{
    double sum = 0.0;
    size_t n = reference.size();
    for (size_t i = 0; i < n; ++i) {
        sum += std::pow(reference[i] - simulation[i],2);
    }
    return sum / n;
}

double calculateRelativeErr(const std::vector<double>& simulation, double rmse)
{
    double sum = 0.0;
    size_t n = simulation.size();
    for (size_t i = 0; i < n; ++i) {
        sum += std::abs(simulation[i]);
    }
    double abs_mean = sum / n;
    return rmse / abs_mean;
} 