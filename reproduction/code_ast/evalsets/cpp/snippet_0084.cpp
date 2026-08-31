class Solution {
public:
  int maxArea(vector<int> &heights) {
    int begin,end,ret;
    begin=0;
    end=(int)(heights.size())-1;
    ret = min(heights[begin],heights[end])*(end-begin);
    while(begin<end){
      if(heights[begin]<heights[end]){
        ++begin;
      }else if(heights[begin]>heights[end]){
        --end;
      }else{
        ++begin;
        --end;
      }
      ret = max(ret, min(heights[begin],heights[end])*(end-begin));
    }
    return ret;
  }
};