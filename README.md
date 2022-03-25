# YOLOv5-6.0 train.py

## COCO label format

One image may has several different labels for several bboxs.

X_center, Y_center, width, height are all normalized.

> [ class, x_center, Y_center, width, height ]


## Out images what means

✅ plot_labels( )

> labels_correlogram.jpg

    seaborn.pairplot()
    #绘制所有标签的 ['x', 'y', 'width', 'height'] 的多变量联合分布直方图，即两两变量之间点的分布，以及单个变量的直方图
> labels.jpg

    1. ax[0].hist   # 绘制 class label 直方图分布
    2. seaborn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)    # 绘制 x,y 直方图
    3. seaborn.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)   # 绘制 w,h 直方图
    4. ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls)) # 初始化一个窗口，按比例恢复归一化的 box 坐标尺寸，将所有 box 中心设为窗口中心，绘制所有 box

- [ ] plot.py 

## Anchor check and kmeans
✅ check_anchors()

    # 每个gt平均有多少个符合thr的anchors
    aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold
    # (x > 1 / thr).float()：获取x中大于1/thr的位置并标为1.和0.
    bpr = (best > 1 / thr).float().mean()  # best possible recall
    # bpr = 最多能被召回(通过thr)的gt框数量 / 所有gt框数量
    # bpr小于0.98 才会用k-means计算anchor

✅ kmean_anchors()

> 聚类

    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    # k:返回的n组anchor聚类中心[9,2]，以及各点与中心欧氏距离均值

> 遗传变异

    # Evolve
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc=f'{PREFIX}Evolving anchors with Genetic Algorithm:')  # progress bar
    for _ in pbar:
        v = np.ones(sh) #!创建全为1的[9,2]矩阵
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            #! 如果全为1
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
            #! (npr.random(sh) < mp)以90%变异，满足条件的置True(1.0)，乘以随机数random.random()
            #! ()*npr.randn(*sh)将数据分布变为正态分布，()* s + 1将均值调整为1
            #! .clip(0.3, 3.0)将anchor尺寸变换因子限制到0.3-3.0之间
        kg = (k.copy() * v).clip(min=2.0)
        #! .clip(min=2.0)将anchor尺寸大于2个像素
        fg = anchor_fitness(kg)#! 计算适应度
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k, verbose)

<b>Reference</b>

> [Sample assigment](https://flyfish.blog.csdn.net/article/details/119332396)

> [YOLOv3与YOLOv5自动生成anchor区别](https://flyfish.blog.csdn.net/article/details/119332396)

## Loss
✅ loss_init

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

> BCEWithLogitsLoss = Sigmoid + BCEloss

> $\ell_c(x, y) = L_c = \{l_{1,c},\dots,l_{N,c}\}^\top, \quad l_{n,c} = - w_{n,c} \left[ p_c y_{n,c} \cdot \log \sigma(x_{n,c}) + (1 - y_{n,c}) \cdot \log (1 - \sigma(x_{n,c})) \right]$

> Focal loss

    BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
> loss = -α * [ (1 - p_t) ** γ ] * log(p_t)

    p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
    alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)

✅ loss_build_tragets

    """
    筛选不同特征尺度下与anchor匹配的ground truth
    """
    # 选GT：根据anchor的长宽比来筛选属于这一层特征图的targets(GT) shape[m,7] 假设筛选得到m个
    t = t[j]  # filter
    
    # 扩充有效网格，即增加左上/右上/左下/右下网格计算offsets，得到True或False
    # (左上/右上/左下/右下互斥,只有1个True)
    j = torch.stack((torch.ones_like(j), j, k, l, m))
    
    # 先复制5次，shape[5,m,7],再用j筛选得到扩充后（3个网格）的GT，shape[n,7] 假设筛选得到n个
    t = t.repeat((5, 1, 1))[j]
    
    # gxy即GT的x和y，shape[m,2]，用None在第一维扩展后 shape[1,m,2]
    # off shape[5,2]，用None扩展第二维 shape[5,1,2]
    # 相加 shape[5,m,2]，用j筛选后 shape[k,2] 假设筛选得到k个
    offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
    
    # Define
    b, c = t[:, :2].long().T  # image, class
    gxy = t[:, 2:4]  # grid xy
    gwh = t[:, 4:6]  # grid wh
    #! 计算GT相对3个网格的坐标
    gij = (gxy - offsets).long()
    #! 拆解为横纵坐标
    gi, gj = gij.T  # grid xy indices
    
    # Append
    #! anchor idx
    a = t[:, 6].long()  # anchor indices
    #! gj.clamp_(0, gain[3] - 1) 在(0,h)范围内的坐标，即在特征图大小尺寸内的坐标
    indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
    #! gxy - gij即偏移量offset
    tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
    anch.append(anchors[a])  # anchors
    tcls.append(c)  # class

✅ loss_call

    pxy = ps[:, :2].sigmoid() * 2 - 0.5
    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]

对照 yolov3 中公式有异同

![yolov3](https://pic4.zhimg.com/80/v2-1229d5cfa3a57f06246eef447d5d32cf_1440w.jpg)


- [ ] bbox_iou 

<b>Reference</b>

> [BCE loss OR Binary Cross Entropy Loss](https://blog.csdn.net/Blankit1/article/details/119799222)

> [Pytorch_BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss)

>[anchor match](https://blog.csdn.net/ChuiGeDaQiQiu/article/details/116402281)

## Val
> NMS
```
"""x.shape[530,85] --> [800,6]"""
# x[:, 5:] 包含80个类别的各自score，哪一个大于阈值就属于哪一类
i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T    
# nonzero函数返回非零元素的idx
x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
# 返回的j从5开始重新计数,所以j+5
# x[i, j + 5, None]扩展为n行1列，即clas_conf;j[:, None]扩展为n行1列，即class

# Batched NMS
c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
#! NMS只在同一类之内使用，所以对每一类添加一个不同的偏移c，使得不同类之间不会重叠
boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
i = torchvision.ops.nms(boxes, scores, iou_thres)  #! NMS return filtered idx
```


## image weights

- [ ] how to do
## Other tips of yolov5
Use eval() trans str to class variable or get its values

    # yolo.py
    import torch
    
    Conv = torch.nn.Conv2d
    m = 'Conv'
    
    nc = 80
    
    out = eval(m)
    ncout = eval('nc')
    
    print(out)
    print(ncout)
    #get model type
    print(str(out)[8:-2].replace('__main__.', ''))

Use function glob.glob() match all the files (jpg) in this path 

    # datasets.py
    import glob
    
    f = [] # image file
    p = '../datasets/coco128/images/train2017'
    f += glob.glob(str(p / '**' / '*.*'), recursive=True)


> anchor assignment
![图片](https://user-images.githubusercontent.com/67272893/160047605-4c9df3ee-491d-4cd6-aa24-b1304908d364.png)




# Reference
> plots.py
[(here)](https://blog.csdn.net/qq_38253797/article/details/119324328#t16)

> aotoanchor.py
[(here)](https://blog.csdn.net/flyfish1986/article/details/117594265)

> loss.py
[(here)](https://blog.csdn.net/qq_38253797/article/details/119444854)
