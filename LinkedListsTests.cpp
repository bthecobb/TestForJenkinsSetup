#include <gtest/gtest.h>

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    while (head) {
        ListNode* nextNode = head->next;
        head->next = prev;
        prev = head;
        head = nextNode;
    }
    return prev;
}

// ✅ Unit Test Cases
TEST(ReverseListTest, HandlesEmptyList) {
    EXPECT_EQ(reverseList(nullptr), nullptr);
}

TEST(ReverseListTest, HandlesSingleElement) {
    ListNode* node = new ListNode(1);
    EXPECT_EQ(reverseList(node)->val, 1);
}

TEST(ReverseListTest, HandlesMultipleNodes) {
    ListNode* node1 = new ListNode(1);
    ListNode* node2 = new ListNode(2);
    node1->next = node2;
    ListNode* reversed = reverseList(node1);
    EXPECT_EQ(reversed->val, 2);
    EXPECT_EQ(reversed->next->val, 1);
}
