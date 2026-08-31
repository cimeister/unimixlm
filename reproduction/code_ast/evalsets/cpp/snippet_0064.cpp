//
// Created by yangtao on 20-4-26.
//

#include <iostream>
#include <cstdio>
#include <algorithm>
using namespace std;
const int N = 200000 ;
int a[N], b[N], c[N], d[N * 3 + 5], k[N * 3];
int n, m, idx, idx1;
int ans, ans1, ans2;

int query(int x) {
    return lower_bound(d, d + idx1, x) - d;
}

int main() {

    cin >> n;

    for(int i = 0; i < n; i++) {
        scanf("%d", &a[i]);
        d[idx++] = a[i];
    }

    cin >> m;

    for(int i = 0; i < m; i++) {
        scanf("%d", &b[i]);
        d[idx++] = b[i];
    }

    for(int i = 0 ; i < m; i++) {
        scanf("%d", &c[i]);
        d[idx++] = c[i];
    }

    sort( d, d + idx) ;

    for(int i = 0; i < idx; i++) {
        if(i == 0 || d[i] != d[i-1])
            d[idx1++] = d[i];
    }

    for(int i = 0 ; i < n; i++) k[query(a[i])]++;

    for(int i = 0; i < m; i++) {
        int x = k[query(b[i])], y = k[query(c[i])];
        if( x > ans1 || x == ans1 && y > ans2) {
            ans1 = x, ans2 = y, ans = i;
        }
    }

    cout << ans + 1;

    return 0;
}