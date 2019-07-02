package common.utils;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamClass;
import java.io.Serializable;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class MLIOUtils {

	private static class MLObjectInputStream extends ObjectInputStream {

		private ClassLoader classLoader;

		public MLObjectInputStream(InputStream in, ClassLoader classLoader) throws IOException {
			super(in);
			this.classLoader = classLoader;
		}

		// The default implementation uses sun.misc.VM.latestUserDefinedLoader(), which is
		// not necessarily the same class loader used by users of the MLIOUtils methods.
		// This discrepancy may happen when the current Java program is executed with a custom
		// user defined classpath.
		protected Class<?> resolveClass(ObjectStreamClass desc)
				throws IOException, ClassNotFoundException {
			try {
				return Class.forName(desc.getName(), false, classLoader);
			} catch (ClassNotFoundException ex) {
				return super.resolveClass(desc);
			}
		}
	}

	public static <T extends Serializable> T readObjectFromFile(
			final String file, Class<T> classType) throws Exception {
		if ((new File(file)).exists() == false) {
			throw new Exception("file doesn't exists " + file);
		}

		ObjectInputStream objectInputStream = null;
		try {
			BufferedInputStream fileInputStream = new BufferedInputStream(
					new FileInputStream(file));

			objectInputStream = new MLObjectInputStream(fileInputStream, classType.getClassLoader());
			Object o = objectInputStream.readObject();

			if (o.getClass().equals(classType) == true) {
				return classType.cast(o);
			} else {
				throw new Exception("failed to deserialize " + file
						+ " inti class " + classType.getSimpleName());
			}

		} finally {
			if (objectInputStream != null) {
				objectInputStream.close();
			}
		}
	}

	public static <T extends Serializable> T readObjectFromFileGZ(
			final String file, Class<T> classType) throws Exception {
		if ((new File(file)).exists() == false) {
			throw new Exception("file doesn't exists " + file);
		}

		ObjectInputStream objectInputStream = null;
		try {
			BufferedInputStream fileInputStream = new BufferedInputStream(
					new FileInputStream(file));

			GZIPInputStream gzInputStream = new GZIPInputStream(
					fileInputStream);

			objectInputStream = new MLObjectInputStream(gzInputStream, classType.getClassLoader());
			Object o = objectInputStream.readObject();

			if (o.getClass().equals(classType) == true) {
				return classType.cast(o);
			} else {
				throw new Exception("failed to serialize " + file
						+ " inti class " + classType.getSimpleName());
			}

		} finally {
			if (objectInputStream != null) {
				objectInputStream.close();
			}
		}
	}

	public static void writeObjectToFile(final Object object, final String file)
			throws IOException {
		ObjectOutputStream objectOutputStream = null;
		try {
			BufferedOutputStream fileOutputStream = new BufferedOutputStream(
					new FileOutputStream(file));

			objectOutputStream = new ObjectOutputStream(fileOutputStream);
			objectOutputStream.writeObject(object);

		} finally {
			if (objectOutputStream != null) {
				objectOutputStream.close();
			}
		}
	}

	public static void writeObjectToFileGZ(final Object object,
			final String file) throws IOException {
		ObjectOutputStream objectOutputStream = null;
		try {
			BufferedOutputStream fileOutputStream = new BufferedOutputStream(
					new FileOutputStream(file));

			GZIPOutputStream gzOutputStream = new GZIPOutputStream(
					fileOutputStream);

			objectOutputStream = new ObjectOutputStream(gzOutputStream);
			objectOutputStream.writeObject(object);

		} finally {
			if (objectOutputStream != null) {
				objectOutputStream.close();
			}
		}
	}

}
