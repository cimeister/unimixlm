#include<iostream>
using namespace std;
#define RED 0
#define GREEN 1
#define BLANK -1


typedef struct Node{
    int capacity;
    int color;
    int current_count;
}Node;


class ChainReaction{
    public:
    Node grid[10][6];

    ChainReaction(){

        for(int i=1;i<9;i++){
            for(int j=1; j<5;j++){
                grid[i][j].capacity = 3;
                grid[i][j].current_count = 0;
                grid[i][j].color = BLANK;
            }
        }

        for(int i = 0; i<10;i++){
            grid[i][0].current_count = 0;
            grid[i][0].color = BLANK;
            grid[i][0].capacity = 2;

            grid[i][5].current_count = 0;
            grid[i][5].color = BLANK;
            grid[i][5].capacity = 2;
        }

        for(int i = 0; i<6;i++){
            grid[0][i].current_count = 0;
            grid[0][i].color = BLANK;
            grid[0][i].capacity = 2;

            grid[9][i].current_count = 0;
            grid[9][i].color = BLANK;
            grid[9][i].capacity = 2;
        }

        grid[0][0].capacity = 1;
        grid[0][5].capacity = 1;
        grid[9][0].capacity = 1;
        grid[9][5].capacity = 1;

    }


    bool isGameOver(int color){
        int opponent = (color + 1) %2;

        for(int i=0;i<10;i++){
            for(int j=0;j<6;j++){
                if (grid[i][j].color == opponent){
                    return false;
                }
            }
        } 

        return true;
    }

    
    


    bool play(int x,int y,int color,bool recur = false){
        
        if(  x < 0 || x > 9 || y < 0 || y > 5 ){
            return false;
        }
        if (  !recur && grid[x][y].color != color && grid[x][y].color != BLANK){
            return false;
        }
        grid[x][y].color = color;
        grid[x][y].current_count+=1;

        if (grid[x][y].capacity < grid[x][y].current_count){
            
            grid[x][y].current_count = 0;
            grid[x][y].color = BLANK;
            //capacity 2
            if(x == 0 && y == 0){
                play(x+1,y,color,true);
                play(x,y+1,color,true);
            } else if ( x == 0 && y == 5){
                play(x+1,y,color,true);
                play(x,y-1,color,true);
            } else if ( x == 9 && y == 0){
                play(x-1,y,color,true);
                play(x,y+1,color,true);
            } else if ( x == 9 && y == 5){
                play(x-1,y,color,true);
                play(x,y-1,color,true);
            } 
            //capacity 3
            else if ( x == 0 ){
                play(x+1,y,color,true);
                play(x,y+1,color,true);
                play(x,y-1,color,true);
            } else if ( x == 9 ){
                play(x,y+1,color,true);
                play(x,y-1,color,true);
                play(x-1,y,color,true);
            } else if ( y == 0 ){
                play(x+1,y,color,true);
                play(x,y+1,color,true);
                play(x-1,y,color,true);
            } else if ( y == 5 ){
                play(x+1,y,color,true);
                play(x,y-1,color,true);
                play(x-1,y,color,true);
            } 
            //capacity 4
            else{
                play(x+1,y,color,true);
                play(x,y-1,color,true);
                play(x-1,y,color,true);
                play(x,y+1,color,true);
            }
            return true;
        }
    }

    void print(){
        cout<<"  ";
        for(int i =0;i<6;i++){
            cout<<i<<"\t";
        }
        cout<<endl;
        for(int i=0;i<10;i++){
            cout<<i<<" ";
            for(int j=0;j<6;j++){
                char x ;

                if(grid[i][j].color == RED){
                    x = 'x';
                } else if( grid[i][j].color == GREEN){
                    x = 'o';
                } else {
                    x = '-';
                    cout<<x;
                }

                for(int k=0; k<grid[i][j].current_count ; k++ ){
                    cout<<x;
                }
                cout<<"\t";
                
            }
            cout<<endl;
        }
    }
};


int main(){

    ChainReaction game;
    //game.grid[2][1].current_count = 2;
    //game.grid[2][1].color = RED;
    game.print();
    int color = 0;
    int count = 0;
    while(1){
        
        while ( 1 ){
            int x,y;
            cin>>x>>y;
            bool ret = game.play(x,y,color);
            cout<<ret<<endl;
            if (ret) break;
        }
        game.print();
        count++;

        if(game.isGameOver(color) && count > 2){
            break;
        }
        color = (color+1)%2;
        
    }
    return 0;
}