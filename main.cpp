#include <iostream>
#include <string>

bool isPalindrome(const std::string& str) {
    int left = 0, right = str.length() - 1;
    while (left < right) {
        if (str[left] != str[right]) {
            return false;
        }
        left++;
        right--;
    }
    return true;
}

int main() {
    std::string test = "racecar";
    std::cout << "Is palindrome: " << (isPalindrome(test) ? "Yes" : "No") << std::endl;
    return 0;
}
