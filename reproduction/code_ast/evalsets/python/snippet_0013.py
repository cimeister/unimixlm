import requests
from bs4 import BeautifulSoup
import sys
import re


args = sys.argv
page = None
soup = None

def main():
    global page, soup
    if(not(len(args) in {2,  4, 7})):
        #exit with error, if invalid arg length
        error(1)
        return
    
    ip = args[1]
    
    if(len(args) == 2):
        if(args[1] == 'help'):
            print_help()
            return
        
        if(load_URL('http://{}/sw?s=0'.format(ip))):
            get_all()
        return
    
    elif(len(args) == 4 and args[2] == 'get'):
        if(load_URL('http://{}/sw?s=0'.format(ip))):
            get(int(args[3]))
        return

    elif(len(args) == 7 and args[2] == 'set'):
        out = args[3]
        function = args[4]
        user = args[5]
        passwd = args[6]

        if(load_URL('http://{ip}/sw?u={user}&p={passwd}&o={out}&f={function}'.format(ip = ip, out = out, function = function, user = user, passwd = passwd))):
            set(out)
        return

    else:
        error(1)
        return

##
# @var Errorcode
# @return void
#
# Error codes:
# 1:    wrong arguments
# 400+: network connection error
##
def error(code):
    if(code == 1):
        print('Syntaxrror, enter "{} help" for more info.'.format(args[0]))
    elif(400 <= code < 600):
        print('Networking Error: Exited with code {}'.format(code))
    else:
        print('An unknown Error occured. Error {}'.format(code))

def load_URL(URL):
    global page, soup
    page = requests.get(URL)

    if(page.status_code == 200):
        soup = BeautifulSoup(page.content, 'html.parser')
        return True
    else:
        error(page.status_code)
        return False

def print_help():
    global page, soup
    print('help')
    #TODO help

def get_all():
    global page, soup
    for out in find_outs():
        if(out):
            print('on')
        else:
            print('off')


def get(out):
    global page, soup
    if(find_outs()[out - 1]):
        print('on')
    else:
        print('off')

def set(out):
    global page, soup
    get(int(out))

def find_outs():
    global page, soup
    stati = re.findall(r"Out [0-9]+: (0|1)", soup.prettify())

    res = []

    for status in stati:
        res.append(bool(int(status[-1])))

    return res

if __name__ == "__main__":
    main()