#include <iostream>  
using namespace std;  // Ensures cout, endl are accessible

struct Node {
    int value;
    Node* next;
    Node(int val) : value(val), next(nullptr) {}
};

void printList(Node* head) {
    while (head) {
        cout << head->value << " -> ";  // Now works
        head = head->next;
    }
    cout << "nullptr" << endl;  // Now works
}
#Testing
