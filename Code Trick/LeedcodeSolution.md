# 624.[Maximum Distance in Arrays](https://leetcode.cn/problems/maximum-distance-in-arrays/description/)

py.Solution:

    You are given m arrays, where each array is sorted in ascending order.

    You can pick up two integers from two different arrays (each array picks one) and calculate the distance. We define the distance between two integers a and b to be their absolute difference |a - b|.

    Return the maximum distance.

class Solution:

    def maxDistance(self, arrays: List[List[int]]) -> int:
        preMin, preMax = arrays[0][0], arrays[0][len(arrays[0]) - 1]
        ans = -inf
        for i in range(1, len(arrays)):
            x, y = arrays[i][0], arrays[i][len(arrays[i]) - 1]
            ans = max(ans, max(preMax - x, y - preMin))
            preMax = max(preMax, y)
            preMin = min(preMin, x)
        return ans
    
