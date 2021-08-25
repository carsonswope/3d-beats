#include "grouping.h"

#include "stdio.h"
#include "stdint.h"

CppGrouping::CppGrouping() {}

void CppGrouping::make_groups(void* _d, int dim_x, int dim_y) {
    uint16_t* d = (uint16_t*)_d;

    printf("%i x %i\n", dim_x, dim_y);
    // printf("%u\n", d[0]);

    for (int y = 0; y < dim_y; y++) {
        for (int x = 0; x < dim_x; x++) {

            const int i = d[y*dim_x + x];
            printf("%u ", i);

        }

        printf("\n");
    }
}
