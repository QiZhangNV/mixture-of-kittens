# 分享分支的验证范围

历史正式性能数据来自 OCI AGA GB300 EP64 job 811385，共12个模型/精度/合成 plan 组合、84条正式验证记录。所附原始 JSONL 保留原始数值；报告与图来自这些数据，没有将本次整理当成新的性能测量。

分享分支的 CUDA 源文件保持历史粗粒度 TMA-pull 版本的字节内容。原版对照由 pinned Git commit 和补丁生成后，逐文件与历史对照源码匹配；ThunderKittens 的115个 include 文件也匹配。分支整理只改变了代码存放、构建入口和 benchmark 加载对照扩展的路径。

本次整理会执行 CPU 源码/数据哈希检查、两个对照与 dependency 的干净生成检查、Python/Shell 语法检查及12组报告复算。检查结果见 [CPU_VALIDATION.json](CPU_VALIDATION.json)。这些检查不能替代目标环境中的 CUDA 构建与 GPU smoke。

另有从冻结分享包解压、重新编译全部扩展并运行 EP4 BF16/MXFP8 smoke 的 OCI AGA 检查 job 1357149。创建分支时仍在排队，未将其声明为通过。冻结包 SHA256 为 `8e69190f5441935a28a280f8ed0825f3dcfc7b09210a5807f63e7855e819024e`；该任务验证的 device 源码与本分支相同，但使用冻结包的构建入口，并非新分支路径本身的 GPU 验证。

本分支没有 MCore 源码、patch 或对 MCore 的运行依赖。这轮通信融合/消融/性能测试使用独立 kernel benchmark；更早的 MOK–MCore 集成工作属于另一条工作线。
