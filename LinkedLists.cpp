#include <iostream>  l
using namespace std;

struct Node {
    int value;
    Node* next;
    Node(int val) : value(val), next(nullptr) {}  // Constructor
};

void printList(Node* head) {
    while (head) {
        cout << head->value << " -> ";
        head = head->next;
    }
    cout << "nullptr" << endl;
}
