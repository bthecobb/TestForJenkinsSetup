#include <gtest/gtest.h>
#include <vector>
#include <algorithm>

using namespace std;

vector<int> sortArray(vector<int> arr) {
    sort(arr.begin(), arr.end());
    return arr;
}

// ✅ Unit Test Cases
TEST(SortArrayTest, HandlesSortedArray) {
    vector<int> arr = {1, 2, 3};
    EXPECT_EQ(sortArray(arr), vector<int>({1, 2, 3}));
}

TEST(SortArrayTest, HandlesUnsortedArray) {
    vector<int> arr = {3, 1, 2};
    EXPECT_EQ(sortArray(arr), vector<int>({1, 2, 3}));
}
