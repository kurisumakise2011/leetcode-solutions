package com.edu;

public class Asserts {

	public static void assertThat(Object actual, Object expected) {
		if (actual != null && !actual.equals(expected)) {
			throw new AssertionError("\nexpected: " + expected + "\nbut was:  " + actual);
		}
	}

}
