# 当前状态
现在你已经把数据准备和好了，你可以@data里面里面查看dataloader的数据结构，现在你需要设计模型。
你需要@data 里面的代码进行查看dataloader是什么，以及target是什么，以及一些背景。
# 模型
## Decoder 架构
输入的[B, T, V]需要先转化成[B,T*V,dim],这个dim就是类似于768这种模型，应该是通过预设的一些config传入的，预设的参数分成模型config，和训练还有测试的config，你分别先创建对应的模型config typedict类再在模型里面解析。训练的一些参数先不管，这是后面的事情。
给你的数据已经归一化好了，你直接reshape维度，但是id_mask还需要处理，因为T时间的看不到后面的mask，然后T+1时刻的数据可以看到T时刻的数据和T+1时刻的其它所有数据。这个mask是attention_mask: [B, T, V]，这个是在计算注意力的时候用的。你需要进来改成[B, T*V, T*V]的形式。

1.我现在要做的就是做这个最简单的纯decoder的模型，就是输入的Boundary.csv的数据在最左边，相当于systerm prompt，然后因为可以输入第一瞬时的所有要预测的数据的状态，这里就最好是直接吧第一瞬时的数据作为输出数据。然后最后反向传播就是只传播需要预测的下一分钟的数据，并且是一一对应的。
2.暂时现不考虑拓扑关系预定义，让模型自己反向传播学习内化到拓扑关系。
3.现在暂时使用最基础的位置编码正弦–余弦的固定位置编码。
4.一次decoder的前向传播是这一个分钟预测到下一个分钟。
5.使用multihead self attention
6.其它正常的decoder结构，比如残差连接，norm层
7.输入数据都是浮点数，先归一化然后线性映射，可以使用更小 MLP（如 1→128→256，GELU+LayerNorm），对非线性关系更友好。

## 已完成
1.不要引入checkpoint增加代码复杂度
2.输入输出投影一定会有，不要增加复杂度
3.# Input projection
        x = self.input_projection(x)  # [B, T, d_model] 这里 应该是[B, T*V, d_model]
因为我不想变量混淆，T一般长度比较短，就是T=3，你可以增加这个默认
4.不要引入causal mask 增加复杂度
5.positional encoding 分成时间维度的，T的值是1-10，所以可以使用固定的正弦–余弦位置编码，V的值也是固定的正弦余弦，但是可以是学习的编码，或者你思考下用什么编码比较好。最后是一个d-model既要有加上时间的编码，也要加上原本6712的变量表征编码。


## 已完成
帮我删除所有transformer相关的结构，因为已经是用decoder了。并且在这里models写测试文件，就是验证输入输出的维度

## 已完成
现在你需要把整个模型完全兼容transformers仓库的Trainer和TrainingArguments，也就是意味着你需要返回loss，并且forward也需要兼容transformers仓库的要求的格式。LOSS用MSELoss】

## 已完成
现在我发现这个模型的config我需要几个基础的文件大小，所以你需要帮我设置一个config的typedict类，并且在models/config/*.json里面设置各种模型大小，比如模型的隐藏层大小，attention的头的维度，以及其他config.py里面的参数。并且这个参数路径文件需要在根目录下面的configs里面体现，这里是训练的config文件目录，那么模型的大小也应该被包含在这个目录下面。

## 已完成
现在整个模型已经可以跑起来了，但是我需要你实现别的对比的模型，并且把这个结构改成models\decoder\当前的一些文件目录，另外还需要对比CNN，LSTM,GCN三种实现。你当前先完成decoder的迁移工作，并且把别的模型的接口保留出来。