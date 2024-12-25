import openai
def get_response(prompt_content):
    BASE_URL = "https://api.chatgptid.net/v1"
    OPENAI_API_KEY = "fk221426-8c6wYuZWYR5wrztZey4YxiRSgTeCGAbg"
    client = openai.OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL,
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt_content}
        ]
    )

    return response.choices[0].message.content

print(get_response("1+1=?"))


import torch
import numpy as np
def drop_nodes(edge_index, x):
    node_num, _ = x.size()
    _, edge_num = edge_index.size()
    drop_num = int(node_num * 0.2)
    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    edge_index = edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()
    # Randomly reconnect nodes to maintain connectivity\n
    for i in idx_nondrop:
        for j in idx_nondrop:
            if i != j and adj[i, j] == 0:
               if np.random.rand() < 0.1:  # Reconnect with 10% probability
                          adj[i, j] = 1
                          adj[j, i] = 1
                          edge_index = adj.nonzero().t()
    return edge_index,x
