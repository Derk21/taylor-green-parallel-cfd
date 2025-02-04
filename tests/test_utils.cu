#include "utils.cuh"
#include <cassert>
void test_setClosestGridPointIdx()
{
    int closest_x_i, closest_y_i;
    int n = NUM_N;
    int m = M;

    double dx = (PERIODIC_END - PERIODIC_START) / (n - 1);
    double dy = (PERIODIC_END - PERIODIC_START) / (m - 1);
    // Test case 1: Point within bounds
    setClosestGridPointIdx(dx, dx, n, m, closest_x_i, closest_y_i);
    assert(closest_x_i == 3);
    assert(closest_y_i == 1);

    // Test case 2: Point at the boundary
    setClosestGridPointIdx(PERIODIC_END, PERIODIC_END, n, m, closest_x_i, closest_y_i);
    assert(closest_x_i == 2*n - 1);
    assert(closest_y_i == m - 1);

    // Test case 3: Point outside the boundary (positive)
    setClosestGridPointIdx(PERIODIC_END + dx/2, PERIODIC_END + dy/2, n, m, closest_x_i, closest_y_i);
    assert(closest_x_i == 1);
    assert(closest_y_i == 0);

    // Test case 4: Point outside the boundary (negative)
    setClosestGridPointIdx(PERIODIC_START - dx/2, PERIODIC_START - dy/2, n, m, closest_x_i, closest_y_i);
    assert(closest_x_i == 2*n - 1);
    assert(closest_y_i == m - 1);

    // Test case 5: Point exactly at the start
    setClosestGridPointIdx(PERIODIC_START, PERIODIC_START, n, m, closest_x_i, closest_y_i);
    assert(closest_x_i == 1);
    assert(closest_y_i == 0);

    // Test case 6: Point exactly at the middle

    setClosestGridPointIdx((PERIODIC_START + PERIODIC_END) / 2 + 1e-10, (PERIODIC_START + PERIODIC_END) / 2 +1e-10, n, m, closest_x_i, closest_y_i);
    assert(closest_x_i == 2*(n / 2)+1);
    assert(closest_y_i == m / 2);

    std::cout << "All test cases passed!" << std::endl;
}

int main(){
    test_setClosestGridPointIdx();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}