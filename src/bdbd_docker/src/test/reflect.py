# test of python reflection
import inspect
from bdbd_common.utils import fstr, sstr
'''
def reach(name) :
    # https://stackoverflow.com/questions/15608987/access-variables-of-caller-function-in-python
    for f in inspect.stack() :
        if name in f[0].f_locals : return f[0].f_locals[name]
    return None 
def sstr(vs):
    dd = dict()
    for v in vs.split(' '):
        dd[v] = reach(v)
    print(fstr(dd))
'''

# fmat='6.3f'

def p1(**kwargs):
    return p2(**kwargs)

def p2(a=1):
    return(a)

a = 1.23456789
b = [2, 3]

print(sstr('a b', fmat='8.4f'))
exit(0)

print(sstr('a b'))
exit(0)

i_global = 1
def sub():
    i_local = 2
    def sub2():
        i_sub2 = 3
        frame = inspect.currentframe()
        back = frame.f_back
        #print('\nback.f_locals.keys:', dir(back.f_locals.keys))
        print('\n2frame:', dir(frame))
        #print('\n2globals: ', globals())
        print('\n2locals:', locals())
        print('\n2dir:', dir() )
        print('\n2vars:', vars())
        print('\ni_local:', reach('i_local'))
    sub2()
    #print('\nglobals: ', globals())
    print('\nlocals:', locals())
    print('\ndir:', dir() )


sub()
print('\nin main globals: ', globals())
print('\nin main locals:', locals())
print('\nin main dir:', dir())
print('')
