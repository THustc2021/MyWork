import torch
import torch.nn.functional as F

def info_nce_loss(features, n_views=2, temperature=0.05):
    # 这里的labels用来做mask，方便后面与矩阵做逐元素相乘的时候筛选正样本和负样本，以batchsize=3为例，
    # 经过数据增强后一个batch的大小实际上为6，输入的features = [6, 128]
    # 最后生成的labels：tensor([[1., 0., 0., 1., 0., 0.],
    #                        [0., 1., 0., 0., 1., 0.],
    #                        [0., 0., 1., 0., 0., 1.],
    #                        [1., 0., 0., 1., 0., 0.],
    #                        [0., 1., 0., 0., 1., 0.],
    #                        [0., 0., 1., 0., 0., 1.]])
    labels = torch.cat([torch.arange(features.shape[0] // n_views) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    features = F.normalize(features, dim=1)

    # 计算相似度矩阵，即如果一个batch的输入样本为[ a1, a2
    #                                       b1, b2
    #                                       c1, c2]
    # 经过网络特征提取之后为：[a1 b1 c1 a2 b2 c2]
    # 相应地相似度矩阵为：[a1a1 a1b1 a1c1 a1a2 a1b2 a1c2
    #                  b1a1 b1b1 b1c1 b1a2 b1b2 b1c2
    #                  c1a1 c1b1 c1c1 c1a2 c1b2 c1c2
    #                  a2a1 a2b1 a2c1 a2a2 a2b2 a2c2
    #                  b2a1 b2b1 b2c1 b2a2 b2b2 b2c2
    #                  c2a1 c2b1 c2c1 c2a2 c2b2 c2c2]
    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)

    labels = labels[~mask].view(labels.shape[0], -1)
    # 此时的labels为：
    # tensor([[0., 0., 1., 0., 0.],
    #         [0., 0., 0., 1., 0.],
    #         [0., 0., 0., 0., 1.],
    #         [1., 0., 0., 0., 0.],
    #         [0., 1., 0., 0., 0.],
    #         [0., 0., 1., 0., 0.]])
    # 相比原来的labels删除了对角线上锚样本与自己做乘积的情况，
    # 对应在原相似度矩阵的位置上只保留label为1的数，相当于只保留了正样本与锚样本的乘积，即a1a2,b1b2,c1c2...
    # mask为：tensor([[ True, False, False, False, False, False],
    #               [False,  True, False, False, False, False],
    #               [False, False,  True, False, False, False],
    #               [False, False, False,  True, False, False],
    #               [False, False, False, False,  True, False],
    #               [False, False, False, False, False,  True]])
    # 相应地，在相似度矩阵上面排除锚样本与自己相乘的情况
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    # positives 保留正样本与锚样本的乘积：[a1a2
    #                                 b1b2
    #                                 c1c2
    #                                 a2a1
    #                                 b2b1
    #                                 c2c1]
    # negatives 保留锚样本与负样本的乘积:[a1b1 a1c1 a1b2 a1c2
    #                                b1a1 b1c1 b1a2 b1c2
    #                                c1a1 c1b1 c1a2 c1b2
    #                                a2b1 a2c1 a2b2 a2c2
    #                                b2a1 b2c1 b2a2 b2c2
    #                                c2a1 c2b1 c2a2 c2b2]
    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    # 将positives堆在negatives的前面，形如[a1a2 a1b1 a1c1 a1b2 a1c2
    #         #                        b1b2 b1a1 b1c1 b1a2 b1c2
    #         #                        c1c2 c1a1 c1b1 c1a2 c1b2
    #         #                        a2a1 a2b1 a2c1 a2b2 a2c2
    #         #                        b2b1 b2a1 b2c1 b2a2 b2c2
    #         #                        c2c1 c2a1 c2b1 c2a2 c2b2]
    # 最左边一列为infoloss的分子，右边为分子
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)
    # labels = [0, 0, 0, 0, 0, 0]，这里相当于交叉熵损失函数里面样本的真实标签为0
    # 因为对比损失函数跟交叉熵损失的计算形式是一样的，所以如果类别全部为0，表示的对于logits的每一行，都使用索引为0（也就是第一个）的元素作为分子
    logits = logits / temperature
    return logits, labels

if __name__ == '__main__':

    info_nce_loss(torch.rand((16, 256)))
