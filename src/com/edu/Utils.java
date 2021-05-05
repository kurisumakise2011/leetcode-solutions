package com.edu;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;

public class Utils {

	public static int[] toArray(String s) {
		return Arrays.stream(s.replace('[', Character.MIN_VALUE)
			.replace(']', Character.MIN_VALUE).split(","))
			.map(String::trim).mapToInt(Integer::parseInt).toArray();
	}

	public static Main.TreeNode tree(Integer... array) {
		return populate(new Main.TreeNode(), array, 0);
	}

	public static String toString(Main.TreeNode node) {
		StringBuilder sb = new StringBuilder();
		toString(node, sb);
		return "[" + sb.toString().substring(0, sb.length() - 1) + "]";
	}

	private static void toString(Main.TreeNode node, StringBuilder sb) {
		if (node != null) {
			toString(node.left, sb);
			sb.append(node.val).append(",");
			toString(node.right, sb);
		}
	}

	private static Main.TreeNode populate(Main.TreeNode root, Integer[] array, int i) {
		if (i < array.length && array[i] != null) {
			root = new Main.TreeNode(array[i]);
			root.left = populate(root.left, array, 2 * i + 1);
			root.right = populate(root.right, array, 2 * i + 2);
		}
		return root;
	}

}
