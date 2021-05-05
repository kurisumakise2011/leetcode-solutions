package com.edu;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;

import static com.edu.Asserts.assertThat;

@SuppressWarnings("unused")
public class Main {

	public static void main(String[] args) {
		new Main().process();
	}

	private void process() {
//		System.out.println(findDuplicate(new int[]{1, 3, 4, 2, 2}));
//		System.out.println(findDuplicate(new int[]{1,1}));
//		System.out.println(findDuplicate(new int[]{1,1,2}));
//		System.out.println(findDuplicate(new int[]{3,1,4,2,3}));
//		System.out.println(findDuplicate(new int[]{7, 8, 10, 11, 7}));
//		System.out.println(isRobotBounded("GGLLGG"));
//		System.out.println(isRobotBounded("GG"));
//		System.out.println(isRobotBounded("GL"));
		System.out.println(findKthLargest(new int[]{1,4,3,5,7,2}, 2));
	}

	public TreeNode invertTree(TreeNode root) {
		if (root == null) {
			return null;
		}
		Queue<TreeNode> queue = new LinkedList<>();
		queue.add(root);
		while (!queue.isEmpty()) {
			TreeNode node = queue.poll();
			TreeNode tmp = node.left;
			node.left = node.right;
			node.right = tmp;

			if (node.left != null) {
				queue.add(node.left);
			}
			if (node.right != null) {
				queue.add(node.right);
			}
		}
		return root;
	}

	public String minRemoveToMakeValid(String s) {
		char[] chars = s.toCharArray();
		for (int i = 0; i < chars.length; i++) {

		}
		return "";
	}

	public int findKthLargest(int[] nums, int k) {
		PriorityQueue<Integer> minHeap = new PriorityQueue<>(Comparator.comparingInt(o -> o));
		for (int num : nums) {
			minHeap.add(num);
			if (minHeap.size() > k) {
				minHeap.poll();
			}
		}
		return minHeap.poll();
	}

	public String addStrings(String num1, String num2) {
		int p = num1.length();
		int q = num2.length();
		int[] digits = new int[Math.max(p, q) + 1];
		for (int i = p - 1, j = q - 1, k = digits.length - 1; k > 0; k--) {
			digits[k] += (i >= 0 ? num1.charAt(i--) - '0' : 0) + (j >= 0 ? num2.charAt(j--) - '0' : 0);
			if (digits[k] > 9) {
				int tmp = digits[k];
				digits[k] = tmp % 10;
				digits[k - 1] = tmp / 10;
			}
		}

		StringBuilder answer = new StringBuilder();
		for (int digit : digits) {
			answer.append(digit);
		}

		int i = 0;
		while (i < answer.length() - 1 && answer.charAt(i) == '0') i++;
		return answer.substring(i);
	}

	public List<String> fizzBuzz(int n) {
		List<String> answer = new ArrayList<>(n);
		for (int i = 1; i <= n; i++) {
			String s = "";
			if (i % 3 == 0) {
				s += "Fizz";
			}
			if (i % 5 == 0) {
				s += "Buzz";
			}
			answer.add(s.isEmpty() ? String.valueOf(i) : s);
		}
		return answer;
	}

	public void merge(int[] nums1, int m, int[] nums2, int n) {
		int[] temp = new int[m];
		System.arraycopy(nums1, 0, temp, 0, m);
		for (int i = 0, j = 0; i + j < m + n;) {
			if (i == m) {
				System.arraycopy(nums2, j, nums1, i + j, n - j);
				return;
			}
			if (j == n) {
				System.arraycopy(temp, i, nums1, i + j, m - i);
				return;
			}

			if (i < m && temp[i] <= nums2[j]) {
				nums1[i + j] = temp[i];
				i++;
			} else {
				nums1[i + j] = nums2[j];
				j++;
			}
		}
	}

	public boolean isAlienSorted(String[] words, String order) {
		for (int i = 1; i < words.length; i++) {
			char[] first = words[i - 1].toCharArray();
			char[] second = words[i].toCharArray();

			int len = Math.min(first.length, second.length);
			int j;
			for (j = 0; j < len; j++) {
				if (order.indexOf(first[j]) < order.indexOf(second[j])) {
					break;
				}
				if (order.indexOf(first[j]) == order.indexOf(second[j])) {
					continue;
				}
				if (order.indexOf(first[j]) > order.indexOf(second[j])) {
					return false;
				}
			}
			if (j == len && first.length > second.length) {
				return false;
			}
		}
		return true;
	}

	public boolean validPalindrome(String s) {
		return validPalindrome(s, true);
	}


	public boolean validPalindrome(String s, boolean firstTime) {
		for (int i = 0; i < s.length() / 2; i++) {
			if (s.charAt(i) != s.charAt(s.length() - i - 1)) {
				return firstTime && (validPalindrome(s.substring(0, i) + s.substring(i + 1), false)
					|| validPalindrome(s.substring(0, s.length() - i - 1) + s.substring(s.length() - i), false));
			}
		}
		return true;
	}

	class BrowserHistory {
		private String[] trace = new String[16];
		private int pointer = 0;
		private int size = 1;

		public BrowserHistory(String homepage) {
			trace[pointer] = homepage;
		}

		public void visit(String url) {
			trace[++pointer] = url;
			size = pointer < size ? pointer + 1 : size + 1;
			if (size >= trace.length) {
				trace = Arrays.copyOf(trace, size << 2);
			}
		}

		public String back(int steps) {
			pointer = Math.max(pointer - steps, 0);
			return trace[pointer];
		}

		public String forward(int steps) {
			pointer = Math.min(size - 1, steps + pointer);
			return trace[pointer];
		}
	}

	public ListNode reverseList(ListNode head) {
		ListNode previous = null;
		ListNode current = head;
		while (current != null) {
			ListNode tmp = current.next;
			current.next = previous;
			previous = current;
			current = tmp;
		}
		return previous;
	}

	public int maxProfit(int[] prices) {
		if (prices.length == 1) {
			return 0;
		}
		int local = prices[1] - prices[0], global = local;
		for (int i = 2; i < prices.length; i++) {
			local = Math.max(local + prices[i] - prices[i - 1], prices[i] - prices[i - 1]);
			global = Math.max(local, global);
		}
		return Math.max(global, 0);
	}

	public boolean isRobotBounded(String instructions) {
		return false;
	}

	public int minMoves(int[] nums) {
		int moves = 0, min = Integer.MAX_VALUE;
		for (int num : nums) {
			moves += num;
			min = Math.min(min, num);
		}
		return moves - min * nums.length;
	}

	public int firstUniqChar(String s) {
		Map<Character, Integer> map = new HashMap<>();
		for (int i = 0; i < s.length(); i++) {
			map.merge(s.charAt(i), 1, (a, b) -> a + 1);
		}
		for (int i = 0; i < s.length(); i++) {
			if (map.get(s.charAt(i)) == 1) {
				return i;
			}
		}
		return -1;
	}

	public int findDuplicate(int[] nums) {
		int xor = 0;
		for (int i = 0; i < nums.length; i++) {
			xor ^= nums[i];
		}
		return xor;
	}

	class Logger {

		class Record {
			private String message;
			private int timestamp;
		}
		/** Initialize your data structure here. */
		private final Queue<Record> queue = new LinkedList<>();
		private final Set<String> set = new HashSet<>();

		public Logger() {

		}


		/** Returns true if the message should be printed in the given timestamp, otherwise returns false.
		 If this method returns false, the message will not be printed.
		 The timestamp is in seconds granularity. */
		public boolean shouldPrintMessage(int timestamp, String message) {
			while (!queue.isEmpty() && timestamp - queue.peek().timestamp >= 10) {
				set.remove(queue.remove().message);
			}
			boolean unique = set.add(message);
			if (unique) {
				Record record = new Record();
				record.message = message;
				record.timestamp = timestamp;
				queue.add(record);
			}
			return unique;
		}
	}

	// Definition for singly-linked list.
	public static class ListNode {
		int val;
		ListNode next;

		ListNode() {
		}

		ListNode(int val) {
			this.val = val;
		}

		ListNode(int val, ListNode next) {
			this.val = val;
			this.next = next;
		}

		@Override
		public String toString() {
			ListNode dummy = this;
			StringBuilder sb = new StringBuilder();
			sb.append("[");
			sb.append(dummy.val);
			dummy = dummy.next;
			while (dummy != null) {
				sb.append(",");
				sb.append(dummy.val);
				dummy = dummy.next;
			}
			sb.append("]");
			return sb.toString();
		}
	}

	static class MinStack {
		private int pointer = -1;
		private int min = Integer.MAX_VALUE;
		private int[] data = new int[16];

		public void push(int val) {
			if (pointer + 2 >= data.length) {
				data = Arrays.copyOf(data, data.length * 2);
			}
			if (min >= val) {
				data[++pointer] = min;
				min = val;
			}
			data[++pointer] = val;
		}

		public void pop() {
			if (pointer > 0 && min == data[pointer--]) {
					min = data[pointer--];
			}
		}

		public int top() {
			return data[pointer];
		}

		public int getMin() {
			return min;
		}
	}

	public static class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;

		TreeNode() {
		}

		TreeNode(int val) {
			this.val = val;
		}

		TreeNode(int val, TreeNode left, TreeNode right) {
			this.val = val;
			this.left = left;
			this.right = right;
		}
	}

	static class Solution {
		public int maxSubArray(int[] nums) {
			int local = nums[0], global = local;
			for (int i = 1; i < nums.length; i++) {
				local = Math.max(nums[i], local + nums[i]);
				global = Math.max(local, global);
			}
			return global;
		}

		// [0,1,0,3,12]
		public void moveZeroes(int[] nums) {
			int firstNoZero = 0;
			for (int i = 0; i < nums.length; i++) {
				if (nums[i] != 0) {
					nums[firstNoZero++] = nums[i];
				}
			}
			for (int i = firstNoZero; i < nums.length; i++) {
				nums[i] = 0;
			}
		}

		public boolean isPowerOfTwo(int n) {
			return n != 0 && ((long) n & ((long) n - 1)) == 0;
		}

		public boolean isPalindrome(String s) {
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < s.length(); i++) {
				char c = Character.toLowerCase(s.charAt(i));
				if (Character.isLetterOrDigit(c)) {
					sb.append(c);
				}
			}
			for (int i = 0; i < sb.length() / 2; i++) {
				if (sb.charAt(i) != sb.charAt(sb.length() - i - 1)) {
					return false;
				}
			}
			return true;
		}

		public boolean isSymmetric(TreeNode root) {
			return isMirror(root, root);
		}

		public boolean isMirror(TreeNode t1, TreeNode t2) {
			if (t1 == null && t2 == null) return true;
			if (t1 == null || t2 == null) return false;
			return (t1.val == t2.val)
				&& isMirror(t1.right, t2.left)
				&& isMirror(t1.left, t2.right);
		}

		public String longestPalindrome(String s) {
			int size = s.length() - 1;
			while (size > 0) {
				for (int from = 0, to = size; to < s.length(); from++, to++) {
					if (s.charAt(from) == s.charAt(to)) {
						boolean palindrome = isPalindrome(s, from, to);
						if (palindrome) {
							return s.substring(from, to + 1);
						}
					}
				}
				size--;
			}
			return s.substring(0, 1);
		}

		public boolean isPalindrome(String s, int from, int to) {
			for (int i = from; i <= to; i++) {
				if (s.charAt(i) != s.charAt((from +  to) - i)) {
					return false;
				}
			}
			return true;
		}

		// "mississippi", "mis*is*p*."
		public boolean isMatch(String s, String p) {
			int i = 0;
			int j = 1;
			while (i < s.length()) {
				if (j == p.length() + 1) {
					return false;
				}
				char first = p.charAt(j - 1);
				char second = Character.MIN_VALUE;
				if (j < p.length()) {
					second = p.charAt(j);
				}
				if (first == s.charAt(i) || first == '.') {
					i++;
					if (second != '*') {
						j++;
					}
				} else {
					if (second == '*') {
						j += 2;
					} else {
						return false;
					}
				}
			}
			return true;
		}

		public boolean checkOnesSegment(String s) {
			int i = 0;
			while (i < s.length() && s.charAt(i) == '1') i++;
			return s.length() == i || i > 1;
		}

		public int maxAscendingSum(int[] nums) {
			int local = nums[0], global = local;
			for (int i = 1; i < nums.length; i++) {
				local = nums[i - 1] < nums[i] ? local + nums[i] : nums[i];
				global = Math.max(local, global);
			}
			return global;
		}

		public int[] getOrder(int[][] tasks) {
			int[] order = new int[tasks.length];
			int[][] sortedTasks = new int[tasks.length][3];
			for (int i = 0; i < tasks.length; i++) {
				sortedTasks[i] = new int[]{tasks[i][0], tasks[i][1], i};
			}
			Arrays.sort(sortedTasks, (o1, o2) -> {
				// enqueue time
				if (o1[0] == o2[0]) {
					// processing time
					return o1[1] - o2[1];
				}
				// enqueue time
				return o1[0] - o2[0];
			});

			PriorityQueue<int[]> minHeap = new PriorityQueue<>(tasks.length, (o1, o2) -> {
				// processing time
				if (o1[0] == o2[0]) {
					// indexes
					return o1[1] - o2[1];
				}
				// processing time
				return o1[0] - o2[0];
			});

			int i = 0;
			int t = sortedTasks[0][0];
			int ti = 0;
			while (i < order.length) {
				while (ti < sortedTasks.length && sortedTasks[ti][0] <= t) {
					minHeap.add(new int[]{sortedTasks[ti][1], sortedTasks[ti][2]});
					ti++;
				}
				if (!minHeap.isEmpty()) {
					int[] task = minHeap.poll();
					t += task[0];
					order[i++] = task[1];
				} else {
					t = sortedTasks[ti][0];
				}
			}
			return order;
		}

		public int sumBase(int n, int k) {
			int sum = 0;
			while (n > 0) {
				sum += n % k;
				n /= k;
			}
			return sum;
		}

		private static final String VOWEL = "aeiou";
		public int longestBeautifulSubstring(String word) {
			int max = 0;
			int len = word.length();
			for (int i = 0; i < len; i++) {
				int subLen = 0;
				for (int j = 0; j < VOWEL.length(); j++) {
					int before = i;
					while (i < len && word.charAt(i) == VOWEL.charAt(j)) i++;
					if (i == before) {
						if (subLen != 0) {
							i--;
						}
						subLen = 0;
						break;
					}
					subLen += (i - before);
				}
				max = Math.max(subLen, max);
			}
			return max;
		}

		public boolean areAlmostEqual(String s1, String s2) {
			int[] idx = new int[2];
			int j = -1;
			for (int i = 0; i < s1.length(); i++) {
				if (s1.charAt(i) != s2.charAt(i)) {
					if (j > 0) {
						return false;
					}
					idx[++j] = i;
				}
			}
			return s1.charAt(idx[0]) == s2.charAt(idx[1]) && s1.charAt(idx[1]) == s2.charAt(idx[0]);
		}

		public int nearestValidPoint(int x, int y, int[][] points) {
			int idx = -1;
			int min = Integer.MAX_VALUE;
			for (int i = 0; i < points.length; i++) {
				if (x == points[i][0] && y == points[i][1]) {
					return i;
				}
				if (x == points[i][0] || y == points[i][1]) {
					int distance = Math.abs(x - points[i][0]) + Math.abs(y - points[i][1]);
					if (distance < min) {
						idx = i;
						min = distance;
					}
				}
			}
			return idx;
		}

		public int countMatches(List<List<String>> items, String ruleKey, String ruleValue) {
			final int idx = "color".equals(ruleKey) ? 1 : "name".equals(ruleKey) ? 2 : 0;
			return (int)items.stream().map(list -> list.get(idx)).filter(ruleValue::equals).count();
		}

		public int numDifferentIntegers(String word) {
			char[] chars = word.toCharArray();
			Set<String> strings = new HashSet<>();
			StringBuilder number = new StringBuilder();
			for (int i = 0; i < chars.length; i++) {
				if (chars[i] >= '0' && chars[i] <= '9') {
					while (i < chars.length && chars[i] == '0') i++;
					while (i < chars.length && chars[i] >= '0' && chars[i] <= '9') {
						number.append(chars[i]);
						i++;
					}
					strings.add(number.toString());
					number.setLength(0);
				}
			}
			return strings.size();
		}

		public int minOperations(int[] nums) {
			int ops = 0;
			for (int i = 1; i < nums.length; i++) {
				if (nums[i] <= nums[i - 1]) {
					ops += (nums[i - 1] - nums[i] + 1);
					nums[i] = nums[i - 1] + 1;
				}
			}
			return ops;
		}

		public String truncateSentence(String s, int k) {
			char[] chars = s.toCharArray();
			int counter = 0;
			for (int i = 0; i < chars.length; i++) {
				if (chars[i] == ' ') {
					counter++;
				}
				if (counter == k) {
					return s.substring(0, i);
				}
			}
			return s;
		}

		public int arraySign(int[] nums) {
			int minus = 0;
			for (int num : nums) {
				if (num == 0) {
					return 0;
				}
				if (num < 0) {
					minus++;
				}
			}
			return minus % 2 == 0 ? 1 : -1;
		}

		public boolean checkIfPangram(String sentence) {
			boolean[] hashset = new boolean[26];
			char[] chars = sentence.toCharArray();
			int size = 0;
			for (char c : chars) {
				if (!hashset[c - 'a']) {
					++size;
					hashset[c - 'a'] = true;
				}
				if (size == 26) {
					return true;
				}
			}
			return false;
		}

		public boolean squareIsWhite(String coordinates) {
			return ((coordinates.charAt(0) - '0') + (coordinates.charAt(1) - 96)) % 2 != 0;
		}

		public int addDigits(int num) {
			if (num < 10) {
				return num;
			}
			char[] chars = String.valueOf(num).toCharArray();
			int res = 0;
			for (char c : chars) {
				res += (c - '0');
			}
			return addDigits(res);
		}

		public List<List<Integer>> threeSum(int[] nums) {
			if (nums.length < 3) {
				return Collections.emptyList();
			}
			List<List<Integer>> result = new ArrayList<>();
			Arrays.sort(nums);
			int len = nums.length;

			for (int i = 0; i < nums.length; i++) {
				int low = i + 1, high = nums.length - 1, sum;
				while (low < high) {
					sum = nums[i] + nums[low] + nums[high];
					if (sum == 0) {
						result.add(List.of(nums[i], nums[low], nums[high]));
						while (low < len - 1 && nums[low + 1] == nums[low]) low++;
						while (high > 1 && nums[high - 1] == nums[high]) high--;
						low++;
						high--;
					} else if (sum < 0) {
						low++;
					} else {
						high--;
					}
				}
				while (i < len - 1 && nums[i + 1] == nums[i]) i++;
			}
			return result;
		}

		private List<Integer> sort(int num, int num1, int num2) {
			int temp;
			if (num > num1) {
				temp = num;
				num = num1;
				num1 = temp;
			}
			if (num > num2) {
				temp = num;
				num = num2;
				num2 = temp;
			}
			if (num1 > num2) {
				temp = num1;
				num1 = num2;
				num2 = temp;
			}

			return List.of(num, num1, num2);
		}


		public String convert(String s, int numRows) {

			if (numRows == 1) return s;

			StringBuilder ret = new StringBuilder();
			int n = s.length();
			int cycleLen = 2 * numRows - 2;

			for (int i = 0; i < numRows; i++) {
				for (int j = 0; j + i < n; j += cycleLen) {
					ret.append(s.charAt(j + i));
					if (i != 0 && i != numRows - 1 && j + cycleLen - i < n)
						ret.append(s.charAt(j + cycleLen - i));
				}
			}
			return ret.toString();
		}

		public ListNode deleteDuplicates(ListNode head) {
			ListNode dummy = head;
			while (dummy != null && dummy.next != null) {
				if (dummy.val == dummy.next.val) {
					dummy.next = dummy.next.next;
				} else {
					dummy = dummy.next;
				}
			}
			return head;
		}

		public boolean isSameTree(TreeNode p, TreeNode q) {
			if (p == q) {
				return true;
			}
			if (p == null || q == null) {
				return false;
			}
			return p.val == q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
		}

		public int arrangeCoins(int n) {
			int i = 1;
			while (n > 0) {
				n -= i;
				i++;
			}
			return n < 0 ? i - 2 : i - 1;
		}

		public int climbStairs(int n) {
			if (n == 1) {
				return 1;
			}
			int[] f = new int[n];
			f[0] = 1;
			f[1] = 2;
			for (int i = 2; i < n; i++) {
				f[i] = f[i - 1] + f[i - 2];
			}
			return f[n - 1];
		}

		public String addBinary(String a, String b) {
			int len = a.length() - b.length();
			if (len > 0) {
				b = "0".repeat(len) + b;
			} else if (len < 0) {
				a = "0".repeat(len * -1) + a;
			}
			len = a.length();
			int[] res = new int[len + 1];
			for (int i = len - 1; i >= 0; i--) {
				res[i + 1] += (a.charAt(i) - '0') + (b.charAt(i) - '0');
				if (res[i + 1] > 1) {
					res[i] = 1;
					res[i + 1] = res[i + 1] % 2;
				}
			}
			int i = 0;
			while (i < res.length && res[i] == 0) i++;
			StringBuilder answer = new StringBuilder();
			for (; i < res.length; i++) {
				answer.append(res[i]);
			}
			return (answer.length() == 0) ? "0" : answer.toString();
		}

		public int mySqrt(int x) {
			if (x == 1) {
				return 1;
			}
			double a = x + .0;
			double xk = (a / 2.0);
			while (xk * xk - x > 0.01) {
				xk = (xk + a / xk) / 2;
			}
			return (int) xk;
		}

		public int[] plusOne(int[] digits) {
			for (int i = digits.length - 1; i >= 0; i--) {
				if (digits[i] != 9) {
					digits[i]++;
					return digits;
				} else {
					digits[i] = 0;
				}
			}
			digits = Arrays.copyOf(digits, digits.length + 1);
			digits[0] += 1;
			return digits;
		}

		public int lengthOfLastWord(String s) {
			int i = s.length() - 1;
			int j = 0;
			while (i >= 0 && s.charAt(i) == ' ') i--;
			while (i >= 0 && s.charAt(i) != ' ') {
				j++;
				i--;
			}
			return j;
		}

		public int maxArea(int[] height) {
			int maxArea = Integer.MIN_VALUE;
			int area;
			int i = 0;
			int j = height.length - 1;
			while (i < j) {
				area = Math.min(height[i], height[j]) * (j - i);
				maxArea = Math.max(maxArea, area);
				if (height[i] > height[j]) {
					j--;
				} else {
					i++;
				}
			}
			return maxArea;
		}

		public int lengthOfLongestSubstring(String s) {
			if ("".equals(s)) {
				return 0;
			}
			if (s.length() == 1) {
				return 1;
			}
			Map<Character, Integer> seen = new HashMap<>();
			int len = 0;
			int start = 0;

			char[] chars = s.toCharArray();
			for (int end = 0; end < s.length(); end++) {
				if (seen.containsKey(chars[end])) {
					start = Math.max(start, seen.get(chars[end]) + 1);
				}

				seen.put(chars[end], end);
				len = Math.max(len, end - start + 1);
			}
			return len;
		}

		public String multiply(String num1, String num2) {
			if ("0".equals(num1) || "0".equals(num2)) {
				return "0";
			}
			int num1Len = num1.length();
			int num2Len = num2.length();
			int t;
			int[] res = new int[num1Len + num2Len];
			for (int i = num1Len - 1; i >= 0; i--) {
				for (int j = num2Len - 1, r = num1Len - 1 - i; j >= 0; j--, r++) {
					res[r] += (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
					if (res[r] > 9) {
						t = res[r];
						res[r] = t % 10;
						res[r + 1] += t / 10 % 10;
					}
				}
			}
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < res.length; i++) {
				if (sb.length() == 0 && res[res.length - 1 - i] == 0) {
					continue;
				}
				sb.append(res[res.length - 1 - i]);
			}
			return sb.toString();
		}

		public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
			ListNode dummyHead = new ListNode(0);
			ListNode p = l1, q = l2, curr = dummyHead;
			int carry = 0;
			while (p != null || q != null) {
				int x = (p != null) ? p.val : 0;
				int y = (q != null) ? q.val : 0;
				int sum = carry + x + y;
				carry = sum / 10;
				curr.next = new ListNode(sum % 10);
				curr = curr.next;
				if (p != null) p = p.next;
				if (q != null) q = q.next;
			}
			if (carry > 0) {
				curr.next = new ListNode(carry);
			}
			return dummyHead.next;
		}

		public void rotate(int[][] matrix) {
			int temp;
			for (int i = 0; i < matrix.length; i++) {
				for (int j = 0; j < matrix[i].length; j++) {
					temp = matrix[i][matrix.length - j - 1];
					matrix[i][matrix.length - j - 1] = matrix[matrix.length - i - 1][j];
					matrix[matrix.length - i - 1][j] = temp;
				}
			}
		}

		public double myPow(double x, int n) {
			if (n == 0) {
				return 1.0;
			}
			if (Double.compare(x, 1.0) == 0) {
				return x;
			}
			if (Double.compare(x, -1.0) == 0) {
				return n % 2 == 0 ? -x : x;
			}
			if (n == Integer.MIN_VALUE) {
				return 0;
			}
			double y = 1;
			for (int i = 0; i < Math.abs(n); i++) {
				y = y * x;
			}
			return n < 0 ? 1 / y : y;
		}

		public int strStr(String haystack, String needle) {
			if (needle.isEmpty()) {
				return 0;
			}
			int nlen = needle.length();
			int hlen = haystack.length();
			if (nlen > hlen) {
				return -1;
			}
			char[] hchars = haystack.toCharArray();
			char[] nchars = needle.toCharArray();
			int j;
			for (int i = 0; i <= hlen - nlen; i++) {
				j = 0;
				while (i + j < hlen && hchars[i + j] == nchars[j]) {
					j++;
					if (nlen == j) {
						return i;
					}
				}
			}
			return -1;
		}

		public int removeElement(int[] nums, int val) {
			if (nums.length == 0) {
				return 0;
			}
			int j = 0;
			for (int i = 0; i < nums.length; i++) {
				if (nums[i] != val) {
					nums[j++] = nums[i];
				}
			}
			return j;
		}

		public int removeDuplicates(int[] nums) {
			if (nums.length == 0) {
				return 0;
			}
			int j = 0;
			int prev = nums[0];
			for (int i = 1; i < nums.length; i++) {
				if (prev != nums[i]) {
					nums[j] = prev;
					j++;
				}
				prev = nums[i];
			}
			nums[j] = prev;
			return j + 1;
		}

		public int comparison = 0;

		public int searchInsert(int[] nums, int target) {
			if (nums.length == 0) {
				return 0;
			}
			return searchInsert(nums, 0, nums.length, target);
		}

		private int searchInsert(int[] nums, int low, int high, int target) {
			if (high == low) {
				if (target == nums[low]) {
					comparison++;
					return low;
				} else {
					comparison++;
					return target > nums[low] ? low + 1 : low;
				}
			}
			int mid = (high + low) / 2;
			if (target > nums[mid]) {
				comparison++;
				return searchInsert(nums, mid + 1, high, target);
			} else {
				comparison++;
				return searchInsert(nums, low, mid, target);
			}
		}

		public boolean isValid(String s) {
			if (s == null || s.isBlank()) {
				return false;
			}
			Deque<Character> stack = new LinkedList<>();
			char[] chars = s.toCharArray();
			// {()}, {{[]({)}
			for (char c : chars) {
				if (c == 40 || c == 91 || c == 123) {
					stack.push(c);
				} else {
					Character opening = stack.poll();
					if (opening == null || (opening != c - 2 && opening != c - 1)) {
						return false;
					}
				}
			}
			return stack.isEmpty();
		}

//		Map<Character, Integer> romans = Map.of(
//			'I', 1,
//			'V', 5,
//			'X', 10,
//			'L', 50,
//			'C', 100,
//			'D', 500,
//			'M', 1000,
//			'O', 0);

		public List<String> letterCombinations(String digits) {
			if (digits == null || digits.isBlank()) {
				return Collections.emptyList();
			}
			char[] chars = digits.toCharArray();
			List<String> strings = new ArrayList<>();
//			for (int i = 0; i < chars.length; i++) {
//			}
			return Collections.emptyList();
		}

		public int myAtoi(String s) {
			s = s.trim();
			char c;
			int sign = 1;
			int len = s.length();
			int rs = 0, val;
			for (int i = 0; i < len; i++) {
				c = s.charAt(i);
				if (i == 0 && (c == '+' || c == '-')) {
					sign = c == '-' ? -1 : 1;
					continue;
				}
				if (c > 47 && c < 58) {
					val = (c - '0');
					if (rs > (Integer.MAX_VALUE - val) / 10) {
						rs = sign == -1 ? Integer.MIN_VALUE : Integer.MAX_VALUE;
						return rs;
					}
					rs = rs * 10 + val;
				} else {
					break;
				}
			}
			return rs * sign;
		}

		private static int toDecimal(char ch) {
			switch (ch) {
				case 'I':
					return 1;
				case 'V':
					return 5;
				case 'X':
					return 10;
				case 'L':
					return 50;
				case 'C':
					return 100;
				case 'D':
					return 500;
				case 'M':
					return 1000;
				default:
					return 0;
			}
		}

		private static String toRomanNumerals(int digit, int numeral) {
			if (numeral == 4) {
				return "M".repeat(digit);
			}
			if (numeral == 3) {
				if (digit == 9) {
					return "CM";
				}
				if (digit <= 8 && digit >= 5) {
					return "D" + "C".repeat(digit - 5);
				}
				if (digit == 4) {
					return "CD";
				}
				if (digit <= 3) {
					return "C".repeat(digit);
				}
			}
			if (numeral == 2) {
				if (digit == 9) {
					return "XC";
				}
				if (digit <= 8 && digit >= 5) {
					return "L" + "X".repeat(digit - 5);
				}
				if (digit == 4) {
					return "XL";
				}
				if (digit <= 3) {
					return "X".repeat(digit);
				}
			}
			if (numeral == 1) {
				if (digit == 9) {
					return "IX";
				}
				if (digit <= 8 && digit >= 5) {
					return "V" + "I".repeat(digit - 5);
				}
				if (digit == 4) {
					return "IV";
				}
				if (digit <= 3) {
					return "I".repeat(digit);
				}
			}
			return "O";
		}

		public String intToRoman(int num) {
			List<Integer> digits = new ArrayList<>();
			while (num > 0) {
				digits.add(num % 10);
				num /= 10;
			}
			StringBuilder roman = new StringBuilder();
			for (int i = digits.size() - 1; i >= 0; i--) {
				roman.append(toRomanNumerals(digits.get(i), i + 1));
			}
			return roman.toString();
		}

		public int romanToInt(String s) {
			if (s.isEmpty()) {
				return 0;
			}
			s += "O";
			int n = 0;
			char[] rome = s.toCharArray();
			for (int i = 0; i < rome.length; i++) {
				if (rome[i] == 'O') {
					break;
				}
				int a = toDecimal(rome[i]);
				int b = toDecimal(rome[i + 1]);
				if (a < b) {
					n += b - a;
					i++;
				} else {
					n += a;
				}
			}
			return n;
		}

		public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
			if (l1 == null) {
				return l2;
			}
			if (l2 == null) {
				return l1;
			}

			if (l1.val < l2.val) {
				l1.next = mergeTwoLists(l1.next, l2);
				return l1;
			} else {
				l2.next = mergeTwoLists(l1, l2.next);
				return l2;
			}
		}

		public String longestCommonPrefix(String[] strs) {
			if (strs.length == 0 || strs.length > 200) {
				return "";
			}
			String prefix = strs[0];
			int idx = prefix.length();
			for (int i = 1; i < strs.length; i++) {
				for (int j = 0; j < idx; j++) {
					if (j == strs[i].length() || prefix.charAt(j) != strs[i].charAt(j)) {
						idx = j;
						break;
					}
				}
			}
			return prefix.substring(0, idx);
		}

		public boolean isPalindrome(int x) {
			if (x < 0) {
				return false;
			}
			String value = String.valueOf(x);
			int len = value.length() - 1;
			for (int i = 0; i < len / 2 + 1; i++) {
				if (value.charAt(i) != value.charAt(len - i)) {
					return false;
				}
			}
			return true;
		}

		public int reverse(int x) {
			// use long to manage possible out of range integer values
			// it's the simplest way, it takes additional memory, if you want to use only
			// integer, you have to verify each result of multiplication
			long y = 0;
			// remainder
			int r;
			// that it is way to expand decimal numbers
			// for example:
			// 123 = 100 * 1 + 10 * 2 + 3 * 1
			// 2839 = 1000 * 2 + 8 * 100 + 3 * 10 + 9 * 1
			while (x != 0) {
				// compute remainder
				// it will be, 123 % 10 = 3
				// it will be, 12 % 10 = 2
				// it will be, 1 % 10 = 1
				r = x % 10;
				// recover number
				y = y * 10 + r;
				// rid off counted digit
				x /= 10;
			}
			// validate if we are not over the limits
			if (y > Integer.MAX_VALUE || y < Integer.MIN_VALUE) {
				return 0;
			}
			return (int) y;
		}
	}
}