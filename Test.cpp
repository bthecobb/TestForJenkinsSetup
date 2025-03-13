#include <gtest/gtest.h>
#include "main.cpp"

TEST(PalindromeTest, BasicTests) {
    EXPECT_TRUE(isPalindrome("racecar"));
    EXPECT_TRUE(isPalindrome("madam"));
    EXPECT_FALSE(isPalindrome("hello"));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
