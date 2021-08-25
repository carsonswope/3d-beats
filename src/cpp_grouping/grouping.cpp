#include "grouping.h"

#define MAX_UINT16 65535

#include <stdio.h>
#include <stdint.h>
// #include <algorithm>
#include <cstring>
#include <vector>
#include <queue>
#include <utility>

// Wrapper for a 2d array, like the one in numpy!
template <typename T>
class Array2d {
private:
    const int dim_x, dim_y;
    T* data;
    bool data_owner;
    T default_value; // default return for [] if out of bounds!
    const int max_idx;
public:

    explicit Array2d(void* _data, int _dim_x, int _dim_y, T _default=0) :
        data((T*)_data),
        dim_x(_dim_x),
        dim_y(_dim_y),
        default_value(_default),
        max_idx(_dim_x * _dim_y) {
            if (!data) {
                data_owner = true;
                data = new T[dim_x * dim_y];
            } else {
                data_owner = false;
            }
        }
    
    ~Array2d() {
        if (data_owner) {
            delete[] data;
        }
    }

    T get(const int x, const int y) {
        const int idx = get_idx(x, y);
        if (idx < 0 || idx >= max_idx) { return default_value; }
        return data[idx];
    }

    T* get_ptr(const int x, const int y) {
        const int idx = get_idx(x, y);
        if (idx < 0 || idx >= max_idx) { return nullptr; }
        return data + idx;
    }

    void set(const int x, const int y, T val) {
        const int idx = get_idx(x, y);
        if (idx < 0 || idx >= max_idx) { printf("erorr in array2d\n"); }
        data[idx] = val;
    }

    int get_idx(const int x, const int y) {
        const bool out_of_bounds = 
            x < 0 || x >= dim_x ||
            y < 0 || y >= dim_y;
        if (out_of_bounds) return -1;
        return (x * dim_y) + (y);
    }

    void empty() {
        memset(data, 0, sizeof(T) * max_idx);
    }

};

typedef std::pair<int, int> int2;

CppGrouping::CppGrouping() {}

void CppGrouping::make_groups(void* _img, int dim_x, int dim_y, void* _c, void* _g_info, float pct_thresh) {

    const std::vector<int2> DIRS = {
        {-1, 0},
        { 1, 0},
        { 0,-1},
        { 0, 1}
    };

    auto img = Array2d<uint16_t>(_img, dim_y, dim_x);
    auto c = Array2d<int32_t>(_c, dim_y * dim_x, 3);
    auto g_info = Array2d<float>(_g_info, 2, 3);

    auto seen = Array2d<uint8_t>(nullptr, dim_y, dim_x);
    seen.empty();

    std::vector<int2> r_group;
    float r_group_x;
    float r_group_y;

    std::vector<int2> l_group;
    float l_group_x;
    float l_group_y;

    std::vector<int2> current_group;
    std::queue<int2> to_visit;

    for (int y = 0; y < dim_y; y++) {
        for (int x = 0; x < dim_x; x++) {

            if (seen.get(y, x)) continue;
            const uint16_t d = img.get(y, x);
            if (!d) continue;

            seen.set(y, x, 1);
            if (!to_visit.empty()) { printf("huh??\n"); }
            to_visit.emplace(int2{y,x});
            current_group.clear();

            while (!to_visit.empty()) {
                const auto c = to_visit.front();
                to_visit.pop();
                current_group.push_back(c);

                for (auto& d : DIRS) {
                    int2 new_c{
                        c.first + d.first,
                        c.second + d.second
                    };

                    if  (new_c.first >= 0 && new_c.second >= 0 && new_c.first < dim_y && new_c.second < dim_x
                            && !seen.get(new_c.first, new_c.second)) {
                        seen.set(new_c.first, new_c.second, 1);
                        if (img.get(new_c.first, new_c.second)) { to_visit.push(new_c); }
                    }
                }

            }

            if (current_group.size() * 1.f / (dim_x * dim_y) <= pct_thresh) continue;

            int p_sum_y = 0;
            int p_sum_x = 0;

            for (auto& c : current_group) {
                p_sum_y += c.first;
                p_sum_x += c.second;
            }

            const float c_y = (p_sum_y * 1.f) / current_group.size();
            const float c_x = (p_sum_x * 1.f) / current_group.size();

            if (c_x < (dim_x / 2.f)) {
                if (current_group.size() > r_group.size()) {
                    r_group = std::move(current_group);
                    r_group_x = c_x;
                    r_group_y = c_y;
                }
            } else {
                if (current_group.size() > l_group.size()) {
                    l_group = std::move(current_group);
                    l_group_x = c_x;
                    l_group_y = c_y;
                }

            }
        }
    }

    g_info.set(0, 0, r_group.size() * 1.f);
    g_info.set(0, 1, r_group_x);
    g_info.set(0, 2, r_group_y);

    for (int i = 0; i < r_group.size(); i++) {
        const auto _c = r_group[i];
        c.set(i, 0, _c.first);
        c.set(i, 1, _c.second);
        c.set(i, 2, 1); // group 1
    }

    g_info.set(1, 0, l_group.size() * 1.f);
    g_info.set(1, 1, l_group_x);
    g_info.set(1, 2, l_group_y);

    for (int _i = 0; _i < l_group.size(); _i++) {
        const int i = _i + r_group.size();
        const auto _c = l_group[_i];
        c.set(i, 0, _c.first);
        c.set(i, 1, _c.second);
        c.set(i, 2, 2); // group 2
    }
}
