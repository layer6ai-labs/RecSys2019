package common.utils;

import java.util.NavigableMap;
import java.util.Random;
import java.util.TreeMap;

public class MLRandomUtils {

	public static class RandomCollection<E> {
		private final NavigableMap<Double, E> map;
		private final Random random;
		private double total;

		public RandomCollection(final Random random) {
			this.map = new TreeMap<>();
			this.random = random;
			this.total = 0;
		}

		public void add(double weight, E result) {
			if (weight <= 0) {
				throw new IllegalArgumentException(
						"only positive weights " + "allowed");
			}
			this.total += weight;
			this.map.put(this.total, result);
		}

		public E next() {
			double value = this.random.nextDouble() * this.total;
			return this.map.higherEntry(value).getValue();
		}
	}

	public static float nextFloat(final float min, final float max,
			final Random rng) {
		return min + rng.nextFloat() * (max - min);
	}

	public static void shuffle(int[] array, final Random rng) {
		for (int i = array.length - 1; i > 0; i--) {
			int index = rng.nextInt(i + 1);
			// swap
			int element = array[index];
			array[index] = array[i];
			array[i] = element;
		}
	}

	public static void shuffle(Object[] array, int startInclusive,
			int endExclusive, final Random rng) {
		final int len = endExclusive - startInclusive;

		for (int j = len - 1; j > 0; j--) {
			int index = rng.nextInt(j + 1) + startInclusive;
			int i = j + startInclusive;
			// swap
			Object element = array[index];
			array[index] = array[i];
			array[i] = element;
		}
	}

	public static void shuffle(Object[] array, final Random rng) {
		shuffle(array, 0, array.length, rng);
	}

	public static int[] shuffleCopy(int[] array, final Random rng) {

		int[] copy = array.clone();
		shuffle(copy, rng);
		return copy;
	}

	/**
	 * Generate i'th cartesian product (starting from 0'th) amongst n number with length L, without replacement assuming numbers change from the left.
	 * For example, the 7'th cartesian product of n=5 for length L=3 is [2,1,0], where the order of the permutation is [0,0,0], [1,0,0], [2,0,0], and so on.
	 * This is equivalent to converting a base 10 number i to base n, and reversing the digits.
	 *
	 * @param n the number of choices at each position
	 * @param L the length of permutation array
	 * @param i the index of permutation to return.
	 * @return
	 */
	public static int[] iterCartesianProduct(int n, int L, int i) {
		// should be the foll equation theoretically
		// but log suffers rounding error, so we use division instead
		// int maxL = (int) Math.ceil((Math.log((double)i + 1.0) / Math.log((double) n)));
		int maxL = 0;
		double iCopy = i;
		while (iCopy >= 1) {
			iCopy /= n;
			maxL++;
		}
		if (maxL > L) {
			throw new IllegalArgumentException(
					"provided index i cannot represent permutation with given legnth!");
		}
		int[] perm = new int[L];
		double power = Math.pow(n, maxL - 1);
		for (int k = 0; k < maxL; k++) {
			perm[maxL - k - 1] = (int) (i / power);
			i = i % (int) power;
			power /= n;
		}

		return perm;
	}
}
