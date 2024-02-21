##### 作者: 无名氏，某乎和小红薯同名，WX：无名氏的胡言乱语。
### 1、python手动实现二维卷积（一种丑陋但容易背的写法）
```python
import numpy as np 
def conv2d(img, in_channels, out_channels ,kernels, bias, stride=1, padding=0):
    N, C, H, W = img.shape 
    kh, kw = kernels.shape
    p = padding
    assert C == in_channels, "kernels' input channels do not match with img"

    if p:
        img = np.pad(img, ((0,0),(0,0),(p,p),(p,p)), 'constant') # padding along with all axis

    out_h = (H + 2*padding - kh) // stride + 1
    out_w = (W + 2*padding - kw) // stride + 1

    outputs = np.zeros([N, out_channels, out_h, out_w])
    # print(img)
    for n in range(N):
        for out in range(out_channels):
            for i in range(in_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        for x in range(kh):
                            for y in range(kw):
                                outputs[n][out][h][w] += img[n][i][h * stride + x][w * stride + y] * kernels[x][y]
                if i == in_channels - 1:
                    outputs[n][out][:][:] += bias[n][out]
    return outputs
```
### 2、pytorch手动实现自注意力和多头自注意力
from [（多头）自注意力机制的PyTorch实现 - Zzxn's Blog](https://zzxn.github.io/2020/11/03/multihead-attention-in-pytorch.html)
- 自注意力
``` python
from math import sqrt

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    # dim_in: int
    # dim_k: int
    # dim_v: int

    def __init__(self, dim_in, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        # x: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n

        att = torch.bmm(dist, v)
        return att
```

- 多头自注意力
```python
from math import sqrt

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    # dim_in: int  # input dimension
    # dim_k: int   # key and query dimension
    # dim_v: int   # value dimension
    # num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att

```
### 3、图像缩放
#### 步骤：
1. 通过原始图像和比例因子得到新图像的大小，并用零矩阵初始化新图像。
2. 由新图像的某个像素点（x，y）映射到原始图像(x’，y’)处。
3. 对x’,y’取整得到（xx，yy）并得到(xx，yy)、(xx+1，yy)、（xx，yy+1）和（xx+1，yy+1）的值。
4. 利用双线性插值得到像素点(x，y)的值并写回新图像。

双线性插值实现：将每个像素点坐标(x,y)分解为(i+u,j+v), i,j是整数部分，u,v是小数部分，则$ f(i+u,j+v) = (1-u)(1-v) \ast f(i,j)+uv \ast f(i+1,j+1)+u(1-v)f(i+1,j)+(1-u)v \ast f(i,j+1) $。

opencv实现细节：将新图像像素点映射回原图像时，$SrcX=(dstX+0.5) \ast (srcWidth/dstWidth) -0.5，SrcY=(dstY+0.5) \ast (srcHeight/dstHeight)-0.5$，使得原图像和新图像几何中心对齐。因为按原始映射方式，$5 \ast 5$图像缩放成$3 \ast 3$图像，图像中心点(1,1)映射回原图会变成(1.67，1.67)而不是(2,2)。
### 4、图像旋转实现
#### 旋转矩阵：

<center>
<img src="https://picx.zhimg.com/80/v2-71c5652ba1f236a963c717891b0dc538_720w.png?source=d16d100b" alt="图像旋转" title="图像旋转" width=60%/>
</center>

#### 实现思路：
1. 计算旋转后图像的min_x,min_y，将(min_x,min_y)作为新坐标原点(向下取整)，并变换原图像坐标到新坐标系，以防止旋转后图像超出图像边界。
2. 初始化旋转后图像的0矩阵，遍历矩阵中每个点(x,y)，根据旋转矩阵进行反向映射（旋转矩阵的逆，np.linalg.inv(a)），将(x,y)映射回原图(x0,y0)，同样将x0和y0拆分为整数和小数部分：i+u,j+v，进行双线性插值即可。从而得到旋转后图像每个像素（x,y）的值。

### 5、RoI Pooling实现细节
**RoI Pooling**需要经过两次量化实现pooling:
第一次是映射到feature map时，当位置是小数时，对坐标进行最近邻插值。

<center>
<img src="https://picx.zhimg.com/80/v2-b20991eed122c1d2069e72c6c46be207_720w.png?source=d16d100b" alt="RoI pooling第一次量化" title="RoI pooling第一次量化" width="50%" height="50%" align=center/>
</center>

第二次是在pooling时，当RoI size不能被RoI Pooling ouputsize整除时，直接舍去小数位。如4/3=1.33，直接变为1，则RoI pooling变成对每个1*2的格子做pooling，pooling方式可选max或者average。

<center>
<img src="https://picx.zhimg.com/80/v2-c158c576f109eae7d0fd51c112ca625a_720w.png?source=d16d100b" alt="RoI pooling第二次量化" title="RoI pooling第二次量化" width="80%" height="80%" align=center/>
</center>

### 6、RoIAlign实现细节
**RoIAlign**采用双线性插值避免量化带来的特征损失：
将RoI平分成outputsize*outputsize个方格，对每个方格取四个采样点，采样点的值通过双线性插值获得，最后通过对四个采样点进行max或average pooling得到最终的RoI feature。

<center>
<img src="https://pic1.zhimg.com/80/v2-b2760ff923b28435adf03c7ba8ba7bf1_720w.png?source=d16d100b" alt="RoI Align" title="RoI Align" width="80%" height="80%" align=center/>
</center>

### 7、2D/3D IoU实现
```python
#核心思路：
union_h = min(top_y) - max(bottom_y)
union_w = min(right_x) - max(left_x)
def 2d_iou(box1, box2):
    '''
    两个框（二维）的 iou 计算

    注意：左下和右上角点

    box:[x1, y1, x2, y2]
    '''
    # 计算重叠区域的长宽
    in_h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    in_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    inter = 0 if in_h<0 or in_w<0 else in_h*in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
    (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = inter / union
    return iou

# 思路类似，找到原点方向的角点以及斜对角处的角点
def 3d_iou(box1, box2):
    '''
   box:[x1,y1,z1,x2,y2,z2]
   '''
    area1 = (box1[3]-box1[0])*(box1[4]-box1[1])*(box1[5]-box1[2])
    area2 = (box2[3]-box2[0])*(box2[4]-box2[1])*(box2[5]-box2[2])
    area_sum = area1 + area2
    
    #计算重叠长方体区域的两个角点[x1,y1,z1,x2,y2,z2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    z1 = max(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    y2 = min(box1[4], box2[4])
    z2 = min(box1[5], box2[5])
    if x1 >= x2 or y1 >= y2 or z1 >= z2:
        return 0
    else:
        inter_area = (x2-x1)*(y2-y1)*(z2-z1)
    
    return inter_area/(area_sum-inter_area)
```
### 8、手撕NMS
```python
import numpy as np
# from https://github.com/luanshiyinyang/NMS，个人觉得很简洁的一种写法
def nms(bboxes, scores, iou_thresh):
    """
    :param bboxes: 检测框列表
    :param scores: 置信度列表
    :param iou_thresh: IOU阈值
    :return:
    """

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)

    # 结果列表
    result = []
    # 对检测框按照置信度进行从高到低的排序，并获取索引
    index = scores.argsort()[::-1]
    # 下面的操作为了安全，都是对索引处理
    while index.size > 0:
        # 当检测框不为空一直循环
        i = index[0]
        # 将置信度最高的加入结果列表
        result.append(i)

        # 计算其他边界框与该边界框的IOU
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        # 只保留满足IOU阈值的索引
        idx = np.where(ious <= iou_thresh)[0]
        index = index[idx + 1]  # 处理剩余的边框
    bboxes, scores = bboxes[result], scores[result]
    return bboxes, scores
```
### 9、手撕k-means
```python
import numpy as np


def kmeans(data, k, thresh=1, max_iterations=100):
  # 随机初始化k个中心点
  centers = data[np.random.choice(data.shape[0], k, replace=False)]

  for _ in range(max_iterations):
    # 计算每个样本到各个中心点的距离
    distances = np.linalg.norm(data[:, None] - centers, axis=2)

    # 根据距离最近的中心点将样本分配到对应的簇
    labels = np.argmin(distances, axis=1)

    # 更新中心点为每个簇的平均值
    new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])

    # 判断中心点是否收敛，多种收敛条件可选
    # 条件1：中心点不再改变
    if np.all(centers == new_centers):
      break
    # 条件2：中心点的阈值小于某个阈值
    # center_change = np.linalg.norm(new_centers - centers)
    # if center_change < thresh:
    #     break
    centers = new_centers

  return labels, centers


# 生成一些随机数据作为示例输入
data = np.random.rand(100, 2)  # 100个样本，每个样本有两个特征

# 手动实现K均值算法
k = 3  # 聚类数为3
labels, centers = kmeans(data, k)

# 打印簇标签和聚类中心点
print("簇标签:", labels)
print("聚类中心点:", centers)
```
### 10、手撕SoftNMS
```python
import numpy as np

# from github, author: OneDirection9
def soft_nms(dets, method='linear', iou_thr=0.3, sigma=0.5, score_thr=0.001):
    """Pure python implementation of soft NMS as described in the paper
    `Improving Object Detection With One Line of Code`_.

    Args:
        dets (numpy.array): Detection results with shape `(num, 5)`,
            data in second dimension are [x1, y1, x2, y2, score] respectively.
        method (str): Rescore method. Only can be `linear`, `gaussian`
            or 'greedy'.
        iou_thr (float): IOU threshold. Only work when method is `linear`
            or 'greedy'.
        sigma (float): Gaussian function parameter. Only work when method
            is `gaussian`.
        score_thr (float): Boxes that score less than the.

    Returns:
        numpy.array: Retained boxes.

    .. _`Improving Object Detection With One Line of Code`:
        https://arxiv.org/abs/1704.04503
    """
    if method not in ('linear', 'gaussian', 'greedy'):
        raise ValueError('method must be linear, gaussian or greedy')

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # expand dets with areas, and the second dimension is
    # x1, y1, x2, y2, score, area
    dets = np.concatenate((dets, areas[:, None]), axis=1)

    retained_box = []
    while dets.size > 0:
        max_idx = np.argmax(dets[:, 4], axis=0)
        # 将置信度最大的框放在首位
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        retained_box.append(dets[0, :-1])

        xx1 = np.maximum(dets[0, 0], dets[1:, 0])
        yy1 = np.maximum(dets[0, 1], dets[1:, 1])
        xx2 = np.minimum(dets[0, 2], dets[1:, 2])
        yy2 = np.minimum(dets[0, 3], dets[1:, 3])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)
        inter = w * h
        iou = inter / (dets[0, 5] + dets[1:, 5] - inter)

        # 根据IoU大小降低重叠框置信度，IoU越大，置信度减小程度越大
        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_thr] -= iou[iou > iou_thr]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_thr] = 0

        dets[1:, 4] *= weight
        retained_idx = np.where(dets[1:, 4] >= score_thr)[0]
        dets = dets[retained_idx + 1, :]

    return np.vstack(retained_box)


if __name__ == '__main__':
    boxes = np.array([[100, 100, 210, 210, 0.72],
                      [250, 250, 420, 420, 0.8],
                      [220, 220, 320, 330, 0.92],
                      [100, 100, 210, 210, 0.72],
                      [230, 240, 325, 330, 0.81],
                      [220, 230, 315, 340, 0.9]], dtype=np.float32)
    print('soft nms result:')
    print(soft_nms(boxes, method='gaussian'))
```
### 11、手撕Batch Normalization
```python
# 参考并更正自知乎（机器学习入坑者《Batch Normalization原理与python实现》）
class MyBN:
    def __init__(self, momentum=0.01, eps=1e-5, feat_dim=2):
        """
        初始化参数值
        :param momentum: 动量，用于计算每个batch均值和方差的滑动均值
        :param eps: 防止分母为0
        :param feat_dim: 特征维度
        """
        # 均值和方差的滑动均值
        self._running_mean = np.zeros(shape=(feat_dim, ))
        self._running_var = np.ones((shape=(feat_dim, ))
        # 更新self._running_xxx时的动量
        self._momentum = momentum
        # 防止分母计算为0
        self._eps = eps
        # 对应Batch Norm中需要更新的beta和gamma，采用pytorch文档中的初始化值
        self._beta = np.zeros(shape=(feat_dim, ))
        self._gamma = np.ones(shape=(feat_dim, ))

    def batch_norm(self, x):
        """
        BN向传播
        :param x: 数据
        :return: BN输出
        """
        if self.training:
            x_mean = x.mean(axis=0)
            x_var = x.var(axis=0)
            # 对应running_mean的更新公式
            self._running_mean = (1-self._momentum)*x_mean + self._momentum*self._running_mean
            self._running_var = (1-self._momentum)*x_var + self._momentum*self._running_var
            # 对应论文中计算BN的公式
            x_hat = (x-x_mean)/np.sqrt(x_var+self._eps)
        else:
            x_hat = (x-self._running_mean)/np.sqrt(self._running_var+self._eps)
        return self._gamma*x_hat + self._beta
```
整理这篇文章不易，喜欢的话可以关注我-->**无名氏，某乎和小红薯同名，WX：无名氏的胡言乱语。** 定期分享算法笔试、面试干货。

<center>
<img src="..\万字秋招算法岗深度学习八股文大全\公众号.png" width=50%/>
</center>
