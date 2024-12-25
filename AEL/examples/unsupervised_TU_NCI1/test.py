import importlib
heuristic_module = importlib.import_module("add")
eva = importlib.reload(heuristic_module)


def sum(eva,a,b,c):
    print(eva(a,b,c)+10)
    return eva(a,b,c)+10
sum(eva.add_num,1,2,3)