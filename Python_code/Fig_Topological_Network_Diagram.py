# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 18:02:13 2022

    Topological Network Diagram Mining

@author: wenjie
"""
import operator
import numpy as np
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt

# 更新全局参数，设置图形大小
plt.rcParams.update({
    'figure.figsize':(10,10)
})


# 导入自己的数据
dat=np.load('../npz_itemset_AR/Result_BCdat_MpEGS2000_top30_time30000.npz',
            allow_pickle=True)
item_list = dat['lst_topkid']
perf = dat['max_perf']
acc = perf[:,0]     # 0:acc; 1:mcc; 2:auc

# 提取出 mcc>0.5 的特征子集(交易数据)
# item_list = item_list[acc>=0.9]
print('共有{}个特征子集(交易数据)'.format(len(item_list)))

lst_1d =[x for item in item_list for x in item.tolist()]   # 拉直成一维列表
dict_count = Counter(lst_1d)    #
# 按计数进行排序
d_sorted = sorted(dict_count.items(),key=operator.itemgetter(1),reverse=True)

# 创建空的无向图
G = nx.Graph() 


# 生成并添加前 12 个高频的节点， 
tmp12 = d_sorted[:12]
nodes = [x[0] for x in tmp12]
weight_node = [x[1] for x in tmp12]
G.add_nodes_from(nodes) # 添加多个节点

# 生成并添加 边， 
edges = [[node1,node2] for node1 in nodes for node2 in nodes if node1<node2]
wedges = [] # 加权的边， 如(node1,node2,weight)
for edge in edges:
    tmp_count = 0
    for item in item_list:
        if set(edge) < set(item):
            tmp_count += 1
    wedge = (edge[0],edge[1],tmp_count/item_list.shape[0])  # 权重为两基因共现频率
    wedges.append(wedge)
G.add_weighted_edges_from(wedges)

pos=nx.circular_layout(G)  # 节点在同心圆上分布,shell_layout, circular_layout

# 绘制图的节点
nx.draw_networkx_nodes(G,pos,node_size=3800,label=True,
                       node_color='#cccccc',alpha=0.8)

#把节点的标签画出来
labels = {x:'#'+str(x+1) for (x,y) in G.nodes(data=True)}
pos1 = {k:v-[0.0,-0.0] for (k,v) in pos.items()}  # 调整节点标签显示位置
nx.draw_networkx_labels(G,pos1,labels=labels,ax=None,font_size=20,
                        font_weight='bold',font_family='Times New Roman')

# 绘制以权重为线的宽度的图
weight_edge = [float(d['weight']) for (u,v,d) in G.edges(data=True)]  # /28742*20
nx.draw_networkx_edges(G,pos,
    width=[x*25 if x>0.115 else 0 for x in weight_edge], #调节边数，边宽
                       # np.array(weight_edge)/1000,
                       alpha=np.array(weight_edge)*1.2,
                       edge_color='#252525')  #灰色#808080



#获取graph中的边权重
# edge_labels = nx.get_edge_attributes(G,'weight')
# print('weight of all edges:',edge_labels)
#把边权重画出来
# nx.draw_networkx_edge_labels(G, pos, edge_labels)

# plt.savefig('../temp/figpy_网络拓扑图3.svg', format='svg') 
plt.show()
