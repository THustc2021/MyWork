import numpy as np
import importlib
import time
import pickle
import os.path as osp
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import random
from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *
from arguments import arg_parse

class simclr(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, dataset_num_features,alpha=0.5, beta=1., gamma=.1):
        super(simclr, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = False

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):

        if x is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)

        y = self.proj_head(y)

        return y

    def loss_cal(self, x, x_aug):

        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss

class Evaluation():
    def __init__(self) -> None:
        print("begin evaluate")
    def evaluate(self):
        time.sleep(1)
        try:
            heuristic_module = importlib.import_module("ael_alg")
            eva = getattr(importlib.reload(heuristic_module), (heuristic_module.__dir__()[-1])) # 获取图像增强方法
            fitness = train(eva)
            return fitness
        except Exception as e:
            print("Error:",str(e))
            return None

def train(eva):
    args = arg_parse()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

    accuracies = {'val': [], 'test': []}
    epochs = 1
    log_interval = 1
    batch_size = 128
    # batch_size = 512
    lr = args.lr
    DS = args.DS
    print(DS)
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    dataset = TUDataset(path, name=DS, aug=args.aug, eva=eva).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug='none', eva=eva).shuffle()
    print(len(dataset))
    print(dataset.get_num_feature())
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = simclr(args.hidden_dim, args.num_gc_layers,dataset_num_features).to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #print('================')
    #print('lr: {}'.format(lr))
    #print('num_features: {}'.format(dataset_num_features))
    #print('hidden_dim: {}'.format(args.hidden_dim))
    #print('num_gc_layers: {}'.format(args.num_gc_layers))
    #print('================')

    model.eval()
    emb, y = model.encoder.get_embeddings(dataloader_eval)

    aug_P = np.ones(5) / 5
    for epoch in range(1, epochs + 1):
        dataloader.dataset.aug_P = aug_P
        loss_all = 0
        model.train()
        for data in dataloader:
            #print(data)
            data, data_aug = data
            optimizer.zero_grad()

            node_num, _ = data.x.size()
            data = data.to(device)
            x = model(data.x, data.edge_index, data.batch, data.num_graphs)

            data_aug = data_aug.to(device)
            x_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)

            loss = model.loss_cal(x.detach().cpu().to("cpu"), x_aug.detach().cpu().to("cpu"))
            loss = model.loss_cal(x, x_aug)
            print(loss)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))
        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            acc_val, acc = evaluate_embedding(emb, y)
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)
        print(accuracies)
    wrong = 1-accuracies['test'][0]
    return wrong