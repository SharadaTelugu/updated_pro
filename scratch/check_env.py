import sys, importlib.metadata as m
print('Interpreter:', sys.executable)
print('Version:', sys.version)
for dist in m.distributions():
    name = dist.metadata.get('Name')
    version = dist.version
    print(f'{name} {version}')
