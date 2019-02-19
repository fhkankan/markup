# 01两数之和

给定一个整数数组 `nums` 和一个目标值 `target`，请你在该数组中找出和为目标值的那 **两个** 整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。

**示例:**

```
给定 nums = [2, 7, 11, 15], target = 9

因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
```

解法

```python
# 方法一：暴力，时间复杂度o(n^2),空间复杂度O(1)
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]
        return None

# 方法二：哈希,时间复杂度O(n),空间复杂度O(n)
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        a = {}
        for i in range(len(nums)):
            another = target - nums[i]
            if another in a:
                return [a[another], i]
            a[nums[i]] = i
        return None
```

# 07整数反转

给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。

**示例 1:**

```
输入: 123
输出: 321
```

 **示例 2:**

```
输入: -123
输出: -321
```

**示例 3:**

```
输入: 120
输出: 21
```

**注意:**

假设我们的环境只能存储得下 32 位的有符号整数，则其数值范围为 [−231,  231 − 1]。请根据这个假设，如果反转后整数溢出那么就返回 0。

解法

```python
class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        ret = []
        res = ''
        ab_x = str(abs(x))
        for i in ab_x:
            ret.insert(0, i)
        for i in ret:
            res += str(i)
        if int(res) > (2**31-1):
            return 0
        if x > 0:
            return int(res)
        else:
            return int(res)*(-1)
```

解法2

```python
class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        INT_32_MIN = - 2**31
        INT_32_MAX = 2**31

        sign = 1
        if x < 0:
            sign = -1
            x *= -1

        x_reverse = 0.0
        while x > 0:
            x_reverse = x_reverse * 10 + x % 10
            x = x // 10
            
        x_reverse *= sign
        
        if x_reverse < INT_32_MIN or x_reverse > INT_32_MAX:
            return 0

        return int(x_reverse)
```

# 09回文数

判断一个整数是否是回文数。回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。

**示例 1:**

```
输入: 121
输出: true
```

**示例 2:**

```
输入: -121
输出: false
解释: 从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。
```

**示例 3:**

```
输入: 10
输出: false
解释: 从右向左读, 为 01 。因此它不是一个回文数。
```

解法

```python
class Solution:
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0:
            return False
        s = str(x)
        s_list = []
        r_list = []
        for i in s:
            s_list.append(i)
            r_list.insert(0, i)
        if s_list == r_list:
            return True
        else:
            return False
```

解法二

```python
class Solution:
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if (x < 0) or (x % 10 == 0 and x != 0):
            return False
        
        revertedNumber = 0
        # 防止溢出，转换一半
        while x > revertedNumber:
            revertedNumber = revertedNumber*10 + x % 10
            x //= 10
        return x == revertedNumber or x == revertedNumber //10   
```

解法三

```python
class Solution:
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0:
            return False
        s = str(x)
        r = s[::-1]
        return  s == r
```

