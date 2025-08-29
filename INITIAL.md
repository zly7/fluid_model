## FEATURE:
你不需要读取data路径，相关的数据格式已经整理到@CLAUDE.md，具体的细节也整理到里面了。
 - 数据处理模块：依据data的数据读取，预处理，数据归一化，使用pytorch配套的dataloader进行数据加载
 - 模型模块：设计并实现Decoder模型，使用PyTorch构建模型架构，包含位置编码、残差连接、LayerNorm等模块
 - 训练和推理模块：这里训练和推理一般是同一套代码，能用transformers的相关工具就使用，不要重复造轮子。用tensorboard进行实验管理和可视化


