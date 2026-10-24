#include <iostream>

using namespace std;

int main() {
	int ret = 0, ans = 1;
	int n = 1000000;
	for (int i = 1; i < n; ++ i) {
		long long count = 1, temp = i;
		while (temp != 1) {
			if (temp % 2 == 0)
				temp /= 2;
			else
				temp = 3 * temp + 1;
			++ count;
		}
		if (ret < count) {
			ret = count;
			ans = i;
		}
		cout << i << endl;
	}
	cout << ans << endl;
	cout << ret << endl;
}