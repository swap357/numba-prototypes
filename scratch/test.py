global_double = 1.234

def foo(x):
  return x * 2

def bar(a):
  return int(foo(a) + global_double)
