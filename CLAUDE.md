## 调用MCP
不要进行错误的try和catch，这会显著增加debug的麻烦，就直接让错误暴露出来。而不是总是在最外层增加try catch函数来让错误被隐藏。

## 问题描述
附件1为我国某长输干线管网拓扑示意图,该管网由7个气源、122个分输点、262个管道(长输管道和小管段)、572个节点、23台压缩机、343个球阀及10个调节阀组成,各个设备的简称与图例如表1所示。各设备的具体信息如附件2所示。

表1 拓扑示意图说明

  简称:名称  	图例  	数量  
  N:节点   	Ο   	572 
  E:用户/分输	    	122 
  T:气源   	    	7   
  P:管道   	    	230 
  H:管段   	    	32  
  C:压缩机  	    	23  
  B:球阀   	DO  	343 
  R:调节阀  	Π   	10  
  连接线    	    	    

模型训练、验证所使用的数据如附件3所示。其中训练集中包括264组完整算例,包括各设备、气源、分输点处的边界条件,在对应边界条件下各管道、设备、节点、边界处的水热力仿真结果;验证集中包括30组各设备、气源、分输点处的边界条件。

### 输入数据：
2. 附件3中边界文件“Boundary.csv"各边界条件均通过“设备编号:参数名称”表征,如“T_001:SNQ”表示T_001的设定流量,具体参数名称的含义如表2所示:

表2 边界参数名称说明

  参数名称	含义    
  SNQ 	设定标况流量
  SP  	设定压力  
  ST  	设定运行状态
  SPD 	下游设定压力
  FR  	阀门开度  

T 系列：大部分 SNQ ，一个 SP，只有T_005:SP，其它T_XXX都是SNQ
E 系列：全部 SNQ
C 系列：成对出现（ST 与 SP_out）
R 系列：成对出现（ST 与 SPD）
B 系列：全部 FR
还有一个时间轴：TIME，给的值间隔是半个小时，实际大部分情况是每3小时进行一次数值的更改(这里的原因就是这个数据是真实操作得出的因为原本方法的计算延迟不会每个时间点都更换数据)

### 要预测的数据

3. 附件3中输出文件各变量均通过“设备编号_参数名称”表征,如“B_001_p_in”表示B_001阀门的入口压力,具体参数名称的含义如表3所示:

表3 参数名称说明

  参数名称 	含义  
  p_in 	入口压力
  p_out	出口压力
  q_in 	入口流量
  q_out	出口流量
  t_in 	入口温度
  t_out	出口温度
  inv  	管存  
  pwr  	功率  
下面是要预测的数据，要预测的情况是分钟级别的。
Parameter Analysis by File Type
  B.csv (球阀/Ball Valves):
  - Parameters: p_in, p_out, q_in, q_out, t_in, t_out
  - Pattern: B_XXX_parameter (e.g., B_001_p_in, B_002_p_out)

  C.csv (压缩机/Compressors):
  - Parameters: p_in, p_out, q_in, q_out, t_in, t_out, pwr
  - Pattern: C_XXX_parameter (e.g., C_001_p_in, C_002_pwr)
  
  H.csv (管段/Pipeline Segments):
  - Parameters: p_in, p_out, q_in, q_out, t_in, t_out
  - Pattern: H_XXX_parameter (e.g., H_001_p_in, H_002_t_out)

  N.csv (节点/Nodes):
  - Parameters: p, q, t (simplified compared to others)
  - Pattern: N_XXX_parameter (e.g., N_001_p, N_002_q, N_003_t)

  P.csv (管道/Pipelines):
  - Parameters: p_in, p_out, q_in, q_out, t_in, t_out, inv
  - Pattern: P_XXX_parameter (e.g., P_001_p_in, P_002_inv)

  R.csv (调节阀/Control Valves):
  - Parameters: p_in, p_out, q, t_in, t_out (note: only single q, not q_in/q_out)
  - Pattern: R_XXX_parameter (e.g., R_001_p_in, R_002_q)

  T&E.csv (气源和分输点/Gas Sources and Distribution Points):
  - Parameters: p, q, t (simplified like nodes)
  - Pattern: T_XXX_parameter and E_XXX_parameter (e.g., T_001_p, E_001_q)

拓扑示意图是一个稀疏连接，就是一般一个节点只与少数几个其他节点相连。这里还有一个很重要的点是这里预测的维度是不变的，所以可以每分钟进行一次预测。




## 使用技术栈：
pytorch
可以使用transformers但是未必是必须的
如果没有对应的模型则从pytorch开始构建
暂时不用考虑分布式训练，采用单机单卡训练
调用SwanLab，而不是Wandb

# 规范
能够分到package 并且使用__init__.py就区分package，不要把代码全部写在根目录，但是package不宜太多，不应该超过5个。

