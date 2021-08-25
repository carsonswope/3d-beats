#include "grouping.h"

#include <vector>

using namespace std;

int main() {

    const int DIM_X = 7, DIM_Y = 4;

    vector<uint16_t> d(DIM_X * DIM_Y, 0);

    d[10] = 1;
    d[11] = 1;
    d[12] = 1;

    auto g = CppGrouping();

    g.make_groups((void*)d.data(), DIM_X, DIM_Y);

    return 0;
}