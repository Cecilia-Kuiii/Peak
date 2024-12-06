# Hill-Climbing Search 爬山搜索
爬山搜索算法（或最陡上升 steepest-ascent）从当前状态移动到目标值增加最多的相邻状态。该算法不维护搜索树，仅维护目标的状态和对应的值。

爬山的 “贪心” 策略使其很容易陷入局部最优（见上图），因为从局部上看，这些点对算法而言是全局最大值或平稳区域（plateaus）。平稳区域（Plateaus）可以分为没有方向使得目标值增长的（“flat local maxima”），或增长缓慢的（“shoulders”）平坦区域。

<img src="https://github.com/Cecilia-Kuiii/Peak/blob/d33ba2fcf449e91529635ea9f2087d53b142d175/pic/pseudo%20code/pseudocode_hill_climbing.png">
