#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;

bool customSort(int a, int b) {
    return a > b;  // Sort descending
}

void sortingExample() {
    vector<int> arr = {3, 1, 4, 1, 5, 9};
    sort(arr.begin(), arr.end());  // Default sort (ascending)
    sort(arr.begin(), arr.end(), customSort);  // Custom sort (descending)
}
