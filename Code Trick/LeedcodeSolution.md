# 624.Maximum Distance in Arrays

py.Solution:
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
