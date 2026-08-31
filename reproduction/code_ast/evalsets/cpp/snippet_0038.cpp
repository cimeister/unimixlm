#include "queue.cpp"
#include "player.cpp"

void quiz(){
    Queue<string> *a = new Queue<string>();
    ifstream file;
    Player p1, p2;
    int k=0,player,quiz,catagory,sportScore=0,Score=0,z=0,category=0;

    string SPORTS="BDCDADBCCCBABBD",GK="BDBADBAABBAACAA",INVENTIONS="BCBDABDBDCABCCB",
    COMPUTER="CBCABBCBAACCBBB",C="BDDADBAABBCBDAB", psl="BDAACADCBDCDCBA",ANS;


    system("cls");
    system("color e");

    do{
        cout<<"\nEnter Number of player(max 2):";
        while(!(cin>>player)){
            cin.clear();
            cin.ignore(99,'\n');
            cout<<"Invalid! Try again: ";
        }
    }while(player<1||player>2);

    do{
        cout<<"Enter Number of quiz(max 15):";
        while(!(cin>>quiz)){
            cin.clear();
            cin.ignore(99,'\n');
            cout<<"Invalid! Try again: ";
        }
    }while(quiz<1||quiz>15);

    do{
        cout<<"\n\n\t\t1 for Sports\n\t\t2 for Computer Science\n\t\t3 for Inventions\n\t\t4 for General\n\t\t5 for C++\n\t\t6 for PSL(Pakistan Super League)\n";
        while(!(cin>>category)){
            cin.clear();
            cin.ignore(99,'\n');
            cout<<"Invalid! Try again: ";
        }
    }while(category>6 || catagory<1);

    switch(category){
        case 1:file.open("sports.txt");
                ANS=SPORTS;
                break;

        case 2:file.open("Computer.txt");
                ANS=COMPUTER;
                break;

        case 3:file.open("Inventions-and-Discoveries.txt");
                ANS=INVENTIONS;
                break;

        case 4:file.open("GK.txt");
                ANS=GK;
                break;

        case 5:file.open("c++.txt");
                ANS=C;
                break;

        case 6:file.open("psl.txt");
                ANS=psl;
                break;
    }

    char stop='.',ans;
    string first,b;

    for(int i=0;i<15;i++){
        getline(file, first, stop);
        a->enque(first);
    }

    system("cls");
    cout<<"\n\n\t\tPLAYER 1 TURN\n";
    cout<<"\nPLAYER 1 NAME:";
    cin>>first; p1.setName(first);


    cout<<"\n\nPress any key to Start...";
    system("pause>NULL");

    system("cls");
    for(int i=0;i<quiz;i++){
        a->display();
        b=a->deque();

        int valid = 0;
        while(valid != 1)
        {
            while(!(cin>>ans)){
				cin.clear();
				cin.ignore(99,'\n');
				cout<<"Invalid! Try again: ";
			}
			
            if((ans=='A')||(ans=='a')||(ans=='b')||(ans=='B')||(ans=='c')||(ans=='C')||(ans=='d')||(ans=='D'))
            {
                valid=1;
                if(ans == ANS[i]||ans == ANS[i]+32)
                    sportScore++;
            }
            else
                cout<<"\nInvalid!!\nEnter: ";
        }

        a->enque(b);
        k++;
    }

    if(player == 2){
        system("cls");
        cout<<"\n\n\t\tPLAYER 2 TURN\n";
        cout<<"\nPLAYER 2 NAME:";
        cin>>first; p2.setName(first);

        cout<<"\n\nPress any key to Start...";
        system("pause>NULL");

        system("cls");
        for( int i=k,j=k; i<k+quiz; i++,j++){
            if(j>14&&z==0){
                j=0;
                z++;
            }
            a->display();
            b=a->deque();

            int isValid = 0;
            while(isValid != 1)
            {
                while(!(cin>>ans)){
					cin.clear();
					cin.ignore(99,'\n');
					cout<<"Invalid! Try again: ";
				}
				
                if((ans=='A')||(ans=='a')||(ans=='b')||(ans=='B')||(ans=='c')||(ans=='C')||(ans=='d')||(ans=='D'))
                {
                    isValid=1;
                    if(ans == ANS[j]||ans == ANS[j]+32)
                        Score++;
                }
                else
                    cout<<"\nInvalid!!\nEnter: ";
            }

            a->enque(b);
        }
    }

    p1.setScore(sportScore);
    system("cls");
    cout<<endl;
    cout<<p1.getName()<<" got: "<<p1.getScore()<<endl;

    if(player==2){
        p2.setScore(Score);
        cout<<p2.getName()<<" got: "<<p2.getScore()<<endl;
    }

    (sportScore>Score)?p1.highscore(p1):p2.highscore(p2);
}