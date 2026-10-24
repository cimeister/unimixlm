
#pragma once
namespace itertools{
    typedef struct{
        template <typename T>
        T operator ()(T a, T b) const{
            return a+b;
        }
    } Add;

    template <typename CONTAINER, typename  FUNCTION = Add>
    class accumulate{
        CONTAINER con;
        FUNCTION fun;
        typedef typename CONTAINER::value_type value_type;
    public:
        explicit accumulate(CONTAINER cont, FUNCTION func = Add()):
         con(cont), fun(func){}

        class iterator{
            //set the type by the firs argument
            typename CONTAINER::value_type value;
            //2 arguments for first and last variables we got 
            typename CONTAINER::iterator _start;
            typename CONTAINER::iterator _end;
            FUNCTION fun;
        public:
            explicit iterator(typename CONTAINER::iterator iter, typename CONTAINER::iterator end, FUNCTION func):
             _start(iter), _end(end), fun(func){
                 if(_start != _end) value = *iter;
             };
            iterator(const iterator& a) = default;
            iterator& operator=(const iterator& a){
                if (this != &a) {
                    this->value = a.value;
                    this->_start = a._start;
                    this->_end = a._end;
                    this->fun = a.fun;
                }
                return *this;
            };
            bool operator ==(const iterator& other) {
                return (_start == other._start);
            }
            bool operator !=(const iterator& other) {
                return (_start != other._start);
                //maybe (it != other.it)
            }
            iterator& operator ++(){
                ++_start;
                if(_start != _end)
                    value = fun(value, *_start);
                return *this;
            }
            iterator operator ++(int){
                iterator tmp = *this;
                ///make default const so we could writeiterator temp(*this);
                ++(*this);
                return tmp;
            }
            auto operator *(){
                return value;
            }
        };

        //maybe without cons
        iterator begin(){
            return iterator(con.begin(), con.end(), fun);
        }
        iterator end(){
            return iterator(con.end(), con.end(), fun);
        }

    };
}