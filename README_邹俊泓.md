# braincog作业-转换为SNN后的ViT与CNN在分类任务上的简单对比

通过BrainCog中已经给出的代码，发现BrainCog提供了VGGNet和ResNet这两种经典CNN网络转换为SNN的版本。并且通过代码实现可以发现，这种转换是直接通过将原来的激活函数用SNN的神经元替换来实现的。而这种转换方法，理论上来说同样可以套用到Transformer模型中，故而本次作业尝试将ViT转换为SNN，并与相近规模的VGGNet做了简单的比较。

受到算力与时间的限制，本次作业主要在cifar100数据集上进行训练，同时削减了网络的大小，削减的方面包括：VGGNet相比于默认的参数仅使用四分之一的通道数；ViT选取较小的embed_dim以及深度。

由于报名了单人组队，贡献应该就不用说明了。

## ViT_SNN实现

原ViT实现分类任务的前向过程大致上包括如下几步：
- 通过一个卷积层进行PatchEmbedding，卷积核大小与步长均设置成patch-size
- 将卷积后的输出reshape成序列形式，连接一个可训练的class token，并加入位置编码，获得最初的序列特征
- 通过若干TransformerEncoder层提取特征，每个TransformerEncoder层由一次自注意力计算以及一个前向网络组成。
- 取出class token，通过一个分类头以后获得分类结果

ViT中，激活函数主要出现在TransformerEncoder层中的前向网络，原来一般为GELU函数，这里将该激活函数替换为SNN的node。

此外，在PatchEmbedding层中，原实现仅由一层卷积层实现，这样如果转换为SNN的话就无法利用到时间信息，因此在本作业中，将其修改为Conv+Batchnorm+SNNnode的形式，便于其处理时间步长的输入。

由于是初步的尝试，因此仅在以上的两个地方使用了SNN神经元，而没有对自注意力的计算进行修改。

ViT的具体实现可见model_zoo文件夹下的vit_snn.py

## 参数设置

### 网络参数
由于修改了网络的默认参数，这里列举出网络的参数值，以及训练时的超参数设置。

VGG参数：相比于BrainCog提供的VGG实验，本作业中使用的网络将通道数削减了4倍，即4个阶段的特征通道数分别为16、32、64、128。其他设置保持不变。详细可以看vgg_snn.py。

ViT参数：ViT的patchsize设置为8(一般设置16，但考虑到cifar100的输入分辨率为32x32，16倍下采样可能过高，因此设置为8)。embed_dim设置为128，多头注意力的头数设置为4，ViT的深度设置为6。详细可以看vit_snn.py。

以32x32的输出分别使用python包thop测试VGG和ViT的参数量与计算量，结果如下：（该结果可以通过cd到model_zoo文件夹后python vgg_snn.py/vit_snn.py得到）

ViT：FLOPs = 0.164548608G ，Params = 1.212416M

VGG：FLOPs = 0.308931584G ，Params = 0.579098M

VGG的参数量大约为ViT的一般，而计算量为ViT的两倍，总的来说二者应该是规模相近的模型。

### 训练参数

两个网络的训练均采用了一样的设置：batchsize=64，step=8，layer_by_layer=False，其余参数设置使用BrainCog的默认参数，在cifar100上训练600个epoch。

vit和vgg的模型训练指令如下：

python examples\Perception_and_Learning\img_cls\bp\main.py --model VGG_SNN --dataset cifar100 --step 8 --batch-size 64 --act-fun QGateGrad --device 0

python examples\Perception_and_Learning\img_cls\bp\main.py --model VisionTransformer --dataset cifar100 --step 8 --batch-size 64 --act-fun QGateGrad --device 0

## 实验结果分析

log_vgg.txt和log_vit.txt分别记录了vgg和vit训练时的log；data文件夹中上传了性能最佳的模型。实验结果表明，在cifar100的分类任务上，vgg还是比目前实现的vit的性能强大。

其中，vgg_snn在验证集上最高达到了66.74%的准确率(在第591个epoch时达到)；而vit_snn仅能达到51.11%(在第489个epoch时达到)。

得到这个结果，可能有以下原因导致：

- Transformer可能不适合cifar100这样小规模数据集的拟合，且cifar100输入分辨率过低，难以发挥ViT全局感受野的优势；相比之下，CNN对小数据集拟合能力则较强。
- ViT所需的训练条件可能更为严格，而本作业中仅仅使用了BrainCog默认的参数进行训练，且没有使用DropPath等ViT训练时常用的正则化方法。
- ViT转变为SNN可能有更合理的结合方式，而非仅仅简单地把激活函数替换为SNN神经元。

总之，目前看来ViT在SNN上的初步实现性能还不够强大，还需要进一步的探索。
