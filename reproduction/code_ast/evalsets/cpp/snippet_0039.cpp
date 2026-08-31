#include <stack>
#include <string>
using namespace std;

class Solution {
  public:
    int longestValidParentheses(string s) {
        stack<int> st;
        int maxLength = 0, curLength = 0;

        for (int i = 0; i < s.size(); i++) {
            if (!st.empty() && s[i] == ')' && s[st.top()] == '(')
                st.pop();
            else
                st.push(i); // push the index

            if (!st.empty())
                curLength = i - st.top();
            else
                curLength = i + 1;
            maxLength = maxLength > curLength ? maxLength : curLength;
        }
        return maxLength;
    }
};