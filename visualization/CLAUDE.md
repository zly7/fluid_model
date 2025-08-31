## 测试规范
调用MCP 里面的playwright测试streamlit的运行，调用MCP里面的sequential-thinking进行深度思考，然后已经要至少增加到4个变量进行测试。多调用playwright MCP交互以测试完整性。当前的这个CLAUDE.md文件不要写入更改。你可以把新功能的介绍写在README.md里。


## 已完成
帮我重新写可视化的代码，我希望是一个可视化界面，一个是可以选择数据集的下拉框，这里是train路径的下面的路径名，另一个是可以选择变量名的下拉框，变量的名字是excel的第一行的名字。然后纵轴是值，横轴是随着TIME的值的变化。并且可以增加多个变量显示在同一个表，支持多条曲线同时显示。增加变量相当于增加一行可选项，就是选择数据集，然后选择变量名。

## 已完成
现在时间窗口也没有问题，就是我发现当我添加新的变量的时候，以前的老的变量不会保存。尤其是添加到第三个变量的时候，直接下拉框就失效了，帮我修复下

## 已完成
wanring：
D:\ml_pro_master\chroes\fluid_model\visualization\multi_variable_viewer.py:115: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  equipment_data[col] = df[col]
改进点：
我希望能够在选择参数的时候直接通过万能匹配，或者说正则表达式来进行筛选。

## 下面的功能待改进
现在\*的通用匹配不生效啊，我最常用的场景就是查看某个比如N_005\* 然后代表要查看N_005的全部变量的这24小时的变化。那么当我添加变量的时候，下面应该自动加很多框，这样同时有很多变量被添加下面的表格。并且不同变量的颜色应该默认不一样。


