1 LLM query

5种以上大模型

3种以上提示模板

视觉模态输入/文本模态输入

带有文本summary参考的视觉模态输入/文本模态输入


2 非均匀帧采样

师兄版本

我的版本

其他开源模型版本


3 相似度计算增强

不分段式

分段式
分段的帧聚合为视频描述


根据视频描述的来源方式分为，由视频LLM直接生成/由帧描述聚合生成


4 评估

对齐为师兄的版本








需要完成的事情
1 聚合帧caption提取一个视频的文本摘要（参考复用已有的参考文献代码）
2 不分段的情况下，完成整个pipeline，并完成评估，可以理解分段为1


3 针对同一的分段参数K，分段总结视频的文本摘要
实现，分段的文本摘要对比下的帧重要性分数查询
分段的文本摘要与该分段内的帧之间做相似度计算

对分段K做消融实验



最后注意，两个数据集的一些大参数技术选型一致，但是小参数可以不一致以寻找最优性能


之后

确定一个llm_query的分数（12种选择）

确定一个sim的分数（2种选择）

确定nfs分数加成（+0.5吧）

然后就可以计算该数据集的性能

之后再在5个split上跑分最终算一个平均



我们采用这种模式，首先把所有可能的实验设置组合列举出来（可能会有几百种上千种）
之后，针对每种组合，我们计算出该组合下的分数，以及实验参数
最终，我们编写一个脚本计算掉所有的组合，在输出文件夹里面生成很多很多json

最后，我们统一使用一个验证脚本去计算所有组合的f1score并从中找到最好的

一些中间变量的设置

输出的得分文件保存在：out/实验名字/final/ 下
文件名字为：scores_0.json、scores_1.json ...

文件结构为
{
    "exam_config":{
        "nfs": {
            "nfs_model" : ["lv_net"] # 多选一，我们使用的nfs方法，目前就这一种
        },
        "llm_query":{
            "model" : ["moonshot", "deepseek"], # 二选一，即我们选用的大模型类型
            "summary_source": ["moonshot", "deepseek", "none"], # 二选一，我们使用的文本摘要的来源，如果没有使用文本摘要，则是none
            "with_summary" : [true, false], # 二选一，是否包含文本摘要对比
            "with_explanation" : [true, false], # 二选一，是否包含解释性内容
        },
        "sim" : {
            "summary_source" : ["moonshot", "deepseek"] # 帧与文本摘要做相似度计算时，文本摘要的来源
        },
        "weight" : float , # 0 ~ 1 的数，以0.2为间隔，代表权重
    }

    "scores" : {
        "summe" : {
            "video_name" : [0.1,0.2,...],
            ...
        }
        "tvsum" : {
            "video_name" : [0.1,0.2,...].
            ...
        }
    }
}

out/exam01/llm_query文件夹下有若干json

其名字类似
moonshot-v1-128k_text_ws_we_tvsum_moonshot.json
其中前面moonshot-v1-128k表示model
ws表示with summary，即需要文本摘要（也可能是wos代表没有文本摘要对比）
we表示with explanation，即需要可解释性（也可能是woe）
最后面如果有moonshot或者deepseek，即代表其文本摘要的来源

每一个json文件里面是一个dict，以视频名索引得到一个list，每个list又是一个dict
如示例所示
{
    "z_6gVvQb2d0": [
            {
                "frame_idx": 0,
                "query": "You are an expert video summarizer. Given the caption of a single video frame, your task is to:\n\n1. Assign an importance score between 0 (not important at all) and 1 (extremely important) indicating how likely this frame should be included in the final video summary.\n2. Provide a brief explanation (1–2 sentences) justifying your score, focusing on why this frame is or isn’t key to understanding the video’s main events or narrative.\n\nCaption: a group of people walking down a street\n\nOutput in the following format (without any extra text):\n\nScore: <float between 0.00 and 1.00>  \nExplanation: <your concise justification>\n",
                "llm_output": "Score: 0.30  \nExplanation: This frame shows a generic scene without clear narrative significance or unique visual information that would be crucial for summarizing the video."
            },
            ...
    ],
    "video_name" : [
        ....
    ],
    ...
}
需要注意的是frame_idx指名了帧号而llm_output中会有一种输出格式Score: 0.30，你需要捕捉到0.30并记录

得到如下格式内容
{
    "exam_config" : {
        "llm_query" : # 记录参数
    }
    "scores" : {
        "summe" : {
            # summe 数据集下的所有内容
            "video_name" : {
                "picks" : [], # 帧号序列（按顺序从 0 开始）
                "llm_query_scores" : [], # 得分序列（需要和帧号对应上）
            },
            ...
        },
        "tvsum" : {
            # tvsum 数据集下的所有内容
        }
    }
}
作为中间变量

之后，nfs相关也要产生一个中间变量
out/exam01/nfs下有json，其picks代表nfs非均匀采样帧得到的帧号
默认我们会从一个视频中每隔15帧采样一帧
然后把这些帧送入nfs模块，从中选出一些帧，请注意这些帧是乱序的，你需要清楚其位置

nfs得到的picks需要你创建一个新的nfs_socre，那些选中帧为0.5，没有选中的为0
至于数组长度需要多长，需要nfs的输入对齐，请参考/root/autodl-tmp/datasets/数据集名字/frames/视频名/，里面的帧文件数量就是你需要的列表长度

pick记住，他是每隔15选取一帧，所以你需要除以15才能对应上连续的得分list的下标

sim请参考上面的步骤，都差不多
其原始json在out/实验名字/sim中



最后生成所有的得分组合 以一个 0~1的权重加权得到最终的分数，生成json保存在 final文件夹中







