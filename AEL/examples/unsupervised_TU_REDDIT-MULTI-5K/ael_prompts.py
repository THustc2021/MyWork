
class GetPrompts():
    def __init__(self):
        self.prompt_task = "Given edge_index,x graph data, \
                            You need to design a function to randomly delete a portion of nodes and update edge information accordingly, \
                           returning updated node, edge data"
        self.prompt_func_name = "drop_nodes"
        self.prompt_func_inputs = ["edge_index,x"]
        self.prompt_func_outputs = ["edge_index,x"]
        self.prompt_inout_inf = ''''edge_index=[2, 247581]: This section describes the edges in the graph. Each column represents an edge, with the first row being the index of the starting node and the second row being the index of the ending node. There are 247581 edges here.
            x=[74233, 1]: This section represents the characteristics of a node and is typically used to describe its attributes. In this example, there are 74233 nodes, each with 1 features.'''

        self.prompt_other_inf = "return edge_index,x:The changed value"

    def get_task(self):
        return self.prompt_task
    
    def get_func_name(self):
        return self.prompt_func_name
    
    def get_func_inputs(self):
        return self.prompt_func_inputs
    
    def get_func_outputs(self):
        return self.prompt_func_outputs
    
    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf

if __name__ == "__main__":
    getprompts = GetPrompts()
    print(getprompts.get_task())
