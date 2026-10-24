from random import uniform, randint, choice
from math import floor

# generates all 4 choices
def genRandom(ans, **kwargs):
    """
    ans is the answer of the problem
    sort= toggles sorting list at the end, default False
    canzero = if the resulting number can equal 0 or not
    """

    ans = float(ans)

    l = []
    r = _getRange(ans)
    if (ans.is_integer()):
        ans = floor(ans)
        # if ans is an integer, it creates an array of all numbers
        # from ans-15 to ans+15 except ans
        # then picks random ones to add into the list
        select = list(range(ans-r, ans+r+1))
        select.pop(select.index(ans))

        print(f's {select}')
        if (not kwargs.get('canzero') and 0 in select):
          select.pop(select.index(0))

        # appends wrong answers to the array
        for i in range(0, 3):
            c = choice(select)
            l.append(c)
            select.pop(select.index(c)) # to make sure that there are no repeats
        
        l.append(ans)

        # sorts the list and returns the list of choices
        if (not kwargs.get('sort')):
            pass
        else:
            l.sort()
        return l
    else:
        for i in range(0, 3):
            # choose if random answer is going to be greater or less
            g = choice([True, False])

            if g:
                num = round(uniform(ans+0.25, ans+r), 2)
                while True:
                    num = round(uniform(ans+0.25, ans+r), 2)
                    try:
                        ind = l.index(ans)
                    except ValueError:
                        break
            else:
                num = round(uniform(ans-0.25, ans-r), 2)
                while True:
                    num = round(uniform(ans-0.25, ans-r), 2)
                    try:
                        ind = l.index(ans)
                    except ValueError:
                        break
            l.append(num)

        l.append(ans)
        if (not kwargs.get('sort')):
            pass
        else:
            l.sort()
        return l

# get the range of answers given the answer
def _getRange(ans):
    ans = float(ans)
    ans = abs(ans)
    r = 0.0

    # print(f"{ans} {ans <= 18.0}")
    if (ans <= 18.0):
        r = 3.0
    elif (ans >= 90.0):
        r = 15.0
    else:
        r = ans/6
    
    if ans.is_integer():
        return floor(r)
    else:
        return round(r, 2)