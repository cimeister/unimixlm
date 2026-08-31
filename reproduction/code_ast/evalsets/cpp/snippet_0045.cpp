//Name : <NAME>
//Roll no : BITF09M038

#include "NDArray.h"

void main ( )
{
	int upper[2] = { 1 , 1  };
	int l[2] = { -2 , -3  };

	int indexSet[2] = { 0 , 0  };

	try
	{
		NDArray arr ( 2 , l , upper );
		arr.setValue ( indexSet , 56 );

		cout << arr.getValue ( indexSet ) << endl ;
		cout << arr.getSize ( ) << endl;
		
	}

	catch ( exception & e )
	{
		cout << e.what ( ) << endl;
	}
}