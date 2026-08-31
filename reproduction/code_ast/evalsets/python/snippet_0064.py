"""
Given an array of non-negative integers, you are initially positioned at the
first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.

For example:
    A = [2,3,1,1,4], return true.

    A = [3,2,1,0,4], return false.
"""
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if nums is None or nums == []:
            return False
        current = 0
        maximum = nums[current]
        while maximum >= current:
            if maximum >= len(nums) - 1:
                return True
            current += 1
            if current > maximum:
                return False
            maximum = max(maximum, current + nums[current])

a = Solution()
print(a.canJump([2,3,1,1,4]))
print(a.canJump([3,2,1,0,4]))