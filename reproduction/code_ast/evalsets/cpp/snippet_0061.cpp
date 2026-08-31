#include<bits/stdc++.h>
using namespace std;

int main()
{
	long long t;
	scanf("%lld",&t);
	while(t--)
	{
		char str[100005]={0},ch;
		scanf("%s",str);
		long long i,j,count1=0,count0=0;
		long long len=strlen(str);
		for(i=0;i<len;i++)
		{
			if(str[i]=='0')
			++count0;
			else if(str[i]=='1')
			++count1;
		}
		if(count0==1 || count1==1)
		printf("Yes\n");
		else
		printf("No\n");
	}
	return 0;
}