package org.apache.sysds.performance.matrix;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.data.*;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Power;
import org.apache.sysds.runtime.matrix.data.*;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import sun.misc.Unsafe;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.foreign.*;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class FFMPerformance {

    private static final String BASE_PATH = "ffm_api_test/";
    private static final double EPSILON = 1E-9;

    // Load Unsafe
    private static final Unsafe UNSAFE;
    static {
        try {
            // Get Unsafe reference
            Field field = Unsafe.class.getDeclaredField("theUnsafe");
            field.setAccessible(true);
            UNSAFE = (Unsafe) field.get(null);
        } catch (Exception e) {
            throw new RuntimeException("Unable to access Unsafe", e);
        }
    }

    public static void testMicrobenchmark(String rows, String cols, int warmupRuns) {
        createBasePath();
        String outputPath = BASE_PATH + "microbenchmark_" + getTimeStamp() + ".csv";

        int[] rowArr = calculateSizes(rows);
        int[] colArr = calculateSizes(cols);

        testMicrobenchmarkWarmup(rowArr[0], colArr[0], warmupRuns);

        try (FileWriter writer = new FileWriter(outputPath)) {
            // Write CSV header
            writer.append("size,time_ffm1,time_ffm2,time_ffm3,time_bb1,time_bb2,time_bb3,time_unsafe1,time_unsafe2,time_unsafe3,correctness\n");

            for (int row : rowArr) {
                for (int col : colArr) {
                    MatrixBlock mb = MatrixBlock.randOperations(row, col, 0.8, -10, 10, "uniform", 22);
                    double[] values = mb.getDenseBlockValues();
                    int amount = row * col;
                    int size = amount * Double.BYTES;
                    long t11, t12, t13, t21, t22, t23, t31, t32, t33;
                    double avg11 = 0, avg12 = 0, avg13 = 0, avg21 = 0, avg22 = 0, avg23 = 0, avg31 = 0, avg32 = 0, avg33 = 0;

                    double[] resultsFFM = new double[amount];
                    double[] resultsBB = new double[amount];
                    double[] resultsUnsafe = new double[amount];

                    // Allocate Memory with FFM API
                    for (int i = 0; i < 10; i++) {
                        try (Arena arena = Arena.ofConfined()) {
                            t11 = System.nanoTime();
                            MemorySegment segment = arena.allocateArray(ValueLayout.JAVA_DOUBLE, amount);
                            avg11 += (System.nanoTime() - t11) / 1_000_000_000.0;

                            t12 = System.nanoTime();
                            segment.fill((byte) 0);
                            avg12 += (System.nanoTime() - t12) / 1_000_000_000.0;

                            t13 = System.nanoTime();
                            for(int j = 0; j < amount; j++) {
                                segment.setAtIndex(ValueLayout.JAVA_DOUBLE, j, values[j]);
                            }
                            avg13 += (System.nanoTime() - t13) / 1_000_000_000.0;

                            for(int j = 0; j < amount; j++) {
                                resultsFFM[j] = segment.getAtIndex(ValueLayout.JAVA_DOUBLE, j);
                            }
                        }
                    }

                    // Allocate Memory with ByteBuffers
                    for (int i = 0; i < 10; i++) {
                        t21 = System.nanoTime();
                        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(size);
                        avg21 += (System.nanoTime() - t21) / 1_000_000_000.0;

                        t22 = System.nanoTime();
                        for(int j = 0; j < amount; j++) {
                            byteBuffer.putDouble(0);
                        }
                        avg22 += (System.nanoTime() - t22) / 1_000_000_000.0;

                        byteBuffer.clear();

                        t23 = System.nanoTime();
                        for (int j = 0; j < amount; j++) {
                            byteBuffer.putDouble(values[j]);
                        }
                        avg23 += (System.nanoTime() - t23) / 1_000_000_000.0;

                        byteBuffer.clear();

                        for(int j = 0; j < amount; j++) {
                            resultsBB[j] = byteBuffer.getDouble();
                        }
                    }

                    // Allocate Memory with Unsafe
                    for (int i = 0; i < 10; i++) {
                        t31 = System.nanoTime();
                        long memoryAddress = UNSAFE.allocateMemory(size);
                        avg31 += (System.nanoTime() - t31) / 1_000_000_000.0;

                        t32 = System.nanoTime();
                        UNSAFE.setMemory(memoryAddress, size, (byte) 0);
                        avg32 += (System.nanoTime() - t32) / 1_000_000_000.0;

                        t33 = System.nanoTime();
                        for (int j = 0; j < amount; j++) {
                            UNSAFE.putDouble(memoryAddress + j * Double.BYTES, values[j]);
                        }
                        avg33 += (System.nanoTime() - t33) / 1_000_000_000.0;

                        for(int j = 0; j < amount; j++) {
                            resultsUnsafe[j] = UNSAFE.getDouble(memoryAddress + j * Double.BYTES);
                        }
                        UNSAFE.freeMemory(memoryAddress);
                    }

                    avg11 /= 10;
                    avg12 /= 10;
                    avg13 /= 10;
                    avg21 /= 10;
                    avg22 /= 10;
                    avg23 /= 10;
                    avg31 /= 10;
                    avg32 /= 10;
                    avg33 /= 10;

                    boolean equalsFFMBB = true;
                    boolean equalsFFMUnsafe = true;
                    boolean equalsUnsafeBB = true;
                    for(int i = 0; i < amount; i++) {
                        if(resultsFFM[i] != resultsBB[i])
                            equalsFFMBB = false;
                        if(resultsFFM[i] != resultsUnsafe[i])
                            equalsFFMUnsafe = false;
                        if(resultsUnsafe[i] != resultsBB[i])
                            equalsUnsafeBB = false;
                    }

                    System.out.println("Memory allocation for " + row + "x" + col + " doubles");
                    System.out.println("FFM API times: "    + avg11 + " : " + avg12 + " : " + avg13);
                    System.out.println("ByteBuffer times: " + avg21 + " : " + avg22 + " : " + avg23);
                    System.out.println("UNSAFE times: "     + avg31 + " : " + avg32 + " : " + avg33);
                    System.out.println("Results equal:" + (equalsFFMBB && equalsFFMUnsafe && equalsUnsafeBB));

                    writer.append(size + ","
                            + avg11 + "," + avg12 + "," + avg13 + ","
                            + avg21 + "," + avg22 + "," + avg23 + ","
                            + avg31 + "," + avg32 + "," + avg33 + "," + (equalsFFMBB && equalsFFMUnsafe && equalsUnsafeBB) + "\n");

                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void testNativeInvocation(String row1, String cols1, String cols2, int k, int warmupRuns, String dmlPath) {
        createBasePath();
        String outputPath = BASE_PATH + "native_invocation_" + getTimeStamp() + ".csv";

        DMLConfig dmlConfig;
        try {
            dmlConfig = new DMLConfig(String.valueOf(new File(dmlPath)));
            ConfigurationManager.setGlobalConfig(dmlConfig);
        }
        catch(FileNotFoundException e) {
            throw new RuntimeException(e);
        }

        int[] rowArr = calculateSizes(row1);
        int[] col1Arr = calculateSizes(cols1);
        int[] col2Arr = calculateSizes(cols2);

        testNativeInvocationWarmup(rowArr[0], col1Arr[0], col2Arr[0], k, warmupRuns);

        try (FileWriter writer = new FileWriter(outputPath)) {
            // Write CSV header
            writer.append("rows,cols1,cols2,k,time_jni,time_ffm1,time_ffm2,correctness\n");

            for(int row : rowArr) {
                for (int col1 : col1Arr) {
                    for (int col2 : col2Arr) {
                        MatrixBlock mb1 = MatrixBlock.randOperations(row, col1, 0.8, -10, 10, "uniform", 6);
                        MatrixBlock mb2 = MatrixBlock.randOperations(col1, col2, 0.8, -10, 10, "uniform", 8);
                        MatrixBlock retJNI = new MatrixBlock(row, col2, false);
                        MatrixBlock retFFM = new MatrixBlock(row, col2, false);

                        long t1, t21, t22;
                        double avg1 = 0, avg21 = 0, avg22 = 0;

                        for (int i = 0; i < 10; i++) {
                            t1 = System.nanoTime();
                            LibMatrixNative.matrixMult(mb1, mb2, retJNI, k);
                            avg1 += (System.nanoTime() - t1) / 1_000_000_000.0;
                        }

                        LibMatrixNativeFFM.controlArena(true);
                        LibMatrixNativeFFM.m1Segment = LibMatrixNativeFFM.arena.allocateArray(ValueLayout.JAVA_DOUBLE,
                                mb1.getDenseBlockValues());
                        LibMatrixNativeFFM.m2Segment = LibMatrixNativeFFM.arena.allocateArray(ValueLayout.JAVA_DOUBLE,
                                mb2.getDenseBlockValues());
                        int len = mb1.getNumRows() * mb2.getNumColumns();

                        for (int i = 0; i < 10; i++) {
                            t21 = System.nanoTime();
                            LibMatrixNativeFFM.retSegment = LibMatrixNativeFFM.arena.allocateArray(ValueLayout.JAVA_DOUBLE, len);
                            avg21 += (System.nanoTime() - t21) / 1_000_000_000.0;

                            t22 = System.nanoTime();
                            LibMatrixNativeFFM.matrixMult(mb1, mb2, retFFM, k);
                            avg22 += (System.nanoTime() - t22) / 1_000_000_000.0;
                        }

                        LibMatrixNativeFFM.controlArena(false);

                        boolean retEqual = compareResults(retJNI, retFFM, row, col2);
                        System.out.println("Results are equal: " + retEqual);

                        avg1 /= 10;
                        avg21 /= 10;
                        avg22 /= 10;
                        System.out.println(row + "x" + col1 + " " + col1 + "x" + col2);
                        System.out.println("JNI: " + avg1 + " s");
                        System.out.println("FFM: " + avg21 + ", " + avg22 + " s");

                        writer.append(row + "," + col1 + "," + col2 + "," + k + "," + avg1 + "," + avg21 + "," + avg22 + "," + retEqual + "\n");
                    }
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void testMemoryAccessMult(String row1, String cols1, String cols2, int k, int warmupRuns, String dmlPath) {
        createBasePath();
        String outputPath = BASE_PATH + "memaccess_mult_" + getTimeStamp() + ".csv";

        DMLConfig dmlConfig;
        try {
            dmlConfig = new DMLConfig(String.valueOf(new File(dmlPath)));
            ConfigurationManager.setGlobalConfig(dmlConfig);
        }
        catch(FileNotFoundException e) {
            throw new RuntimeException(e);
        }

        int[] rowArr = calculateSizes(row1);
        int[] col1Arr = calculateSizes(cols1);
        int[] col2Arr = calculateSizes(cols2);

        testMemoryAccessMultWarmup(rowArr[0], col1Arr[0], col2Arr[0], k, warmupRuns);

        try (FileWriter writer = new FileWriter(outputPath)) {
            writer.append("rows,cols1,cols2,k,time_ffm1,time_ffm2,time_ffm3,time_scalar,time_simd,time_mkl,correctness\n");

            for(int row : rowArr) {
                for (int col1 : col1Arr) {
                    for (int col2 : col2Arr) {
                        MatrixBlock mb1 = MatrixBlock.randOperations(row, col1, 0.8, -10, 10, "uniform", 6);
                        MatrixBlock mb2 = MatrixBlock.randOperations(col1, col2, 0.8, -10, 10, "uniform", 8);

                        MatrixBlock retFFM = new MatrixBlock(row, col2, false);
                        MatrixBlock retScalar = new MatrixBlock(row, col2, false);

                        DenseBlock m1 = mb1.getDenseBlock();
                        DenseBlock m2 = mb2.getDenseBlock();

                        long t11, t12, t13, t2, t3, t4;
                        double avg11 = 0, avg12 = 0, avg13 = 0, avg2 = 0, avg3 = 0, avg4 = 0;

                        for (int i = 0; i < 10; i++) {
                            Arena arena = Arena.ofShared();

                            // Allocate memory for the matrix values
                            t11 = System.nanoTime();
                            LibMatrixMultFFM.segment1 = arena.allocateArray(ValueLayout.JAVA_DOUBLE, m1.size());
                            LibMatrixMultFFM.segment2 = arena.allocateArray(ValueLayout.JAVA_DOUBLE, m2.size());
                            LibMatrixMultFFM.segmentRet = arena.allocateArray(ValueLayout.JAVA_DOUBLE,row*col2);
                            avg11 += (System.nanoTime() - t11) / 1_000_000_000.0;

                            // Write data into the matrices
                            t12 = System.nanoTime();
                            for(int j = 0; j < m1.size(); j++) {
                                LibMatrixMultFFM.segment1.setAtIndex(ValueLayout.JAVA_DOUBLE, j, mb1.getDenseBlockValues()[j]);
                            }

                            for(int j = 0; j < m2.size(); j++) {
                                LibMatrixMultFFM.segment2.setAtIndex(ValueLayout.JAVA_DOUBLE, j, mb2.getDenseBlockValues()[j]);
                            }
                            avg12 += (System.nanoTime() - t12) / 1_000_000_000.0;

                            // Perform multiplication
                            t13 = System.nanoTime();
                            LibMatrixMultFFM.matrixMult(mb1, mb2, retFFM, k);
                            avg13 += (System.nanoTime() -t13) / 1_000_000_000.0;

							double[] tmp = LibMatrixMultFFM.segmentRet.toArray(ValueLayout.JAVA_DOUBLE);
                            // Write values into retFFM so that I can overwrite them from the memorySegmentRet without getting a NullPointerException
                            retFFM = MatrixBlock.randOperations(row, col2, 0.8, -10, 10, "uniform", 8);
                            long nnz = 0;
							for(int j = 0; j < tmp.length; j++)
                                nnz += (retFFM.getDenseBlockValues()[j] = tmp[j]) != 0 ? 1 : 0;

                            retFFM.setNonZeros(nnz);

                            arena.close();
                        }

                        for (int i = 0; i < 10; i++) {
                            t2 = System.nanoTime();
                            LibMatrixMult.matrixMult(mb1, mb2, retScalar, k);
                            avg2 += (System.nanoTime() - t2) / 1_000_000_000.0;
                        }

                        for (int i = 0; i < 10; i++) {
                            t3 = System.nanoTime();
                            LibMatrixMultSIMD.matrixMult(mb1, mb2, retScalar, k);
                            avg3 += (System.nanoTime() - t3) / 1_000_000_000.0;
                        }

                        for (int i = 0; i < 10; i++) {
                            t4 = System.nanoTime();
                            LibMatrixNative.matrixMult(mb1, mb2, retScalar, k);
                            avg4 += (System.nanoTime() - t4) / 1_000_000_000.0;
                        }

                        boolean retEqual = compareResults(retFFM, retScalar, row, col2);
                        System.out.println("Results are equal: " + retEqual);

                        avg11 /= 10;
                        avg12 /= 10;
                        avg13 /= 10;
                        avg2 /= 10;
                        avg3 /= 10;
                        avg4 /= 10;

                        System.out.println(row + "x" + col1 + " " + col1 + "x" + col2);
                        System.out.println("FFM1: " + avg11 + " s");
                        System.out.println("FFM2: " + avg12 + " s");
                        System.out.println("FFM3: " + avg13 + " s");
                        System.out.println("Scalar: " + avg2 + " s");
                        System.out.println("SIMD: " + avg3 + " s");
                        System.out.println("MKL: " + avg4 + " s");

                        writer.append(row + "," + col1 + "," + col2 + "," + k  + "," + avg11 + "," + avg12 + "," + avg13 + "," + avg2 + "," + avg3 + "," + avg4 + "," + retEqual + "\n");
                    }
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

	public static void testMemoryAccessPower(double[] sparsities, int row, int col, double exponent, int k,
		int warmupRuns) {
		createBasePath();
		String outputPath = BASE_PATH + "memaccess_power_" + getTimeStamp() + ".csv";

		testMemoryAccessPowerWarmup(sparsities[0], row, col, exponent, k, warmupRuns);

		RightScalarOperator powerOpK = new RightScalarOperator(Power.getPowerFnObject(), exponent, k);

		try(FileWriter writer = new FileWriter(outputPath)) {
			writer.append("rows,cols,sparsity,k,time_ffm1,time_ffm2,time_ffm3,time_scalar,time_simd,correctness\n");

			for(double sparsity : sparsities) {
				MatrixBlock mb1 = MatrixBlock.randOperations(row, col, sparsity, -10, 10, "uniform", 11);

				MatrixBlock retFFM = new MatrixBlock(row, col, false);
				MatrixBlock retScalar = new MatrixBlock(row, col, false);
				MatrixBlock retSIMD = new MatrixBlock(row, col, false);

				SparseBlock sparseBlock = mb1.getSparseBlock();
				long t11, t12, t13, t2, t3;
				double avg11 = 0, avg12 = 0, avg13 = 0, avg2 = 0, avg3 = 0;

				for(int i = 0; i < 20; i++) {
					Arena arena = Arena.ofShared();

					// Allocate block-wise
					t11 = System.nanoTime();
					LibMatrixBincellFFM.memorySegments1 = new MemorySegment[sparseBlock.numRows()];
					avg11 += (System.nanoTime() - t11) / 1_000_000_000.0;

					t12 = System.nanoTime();
					for(int j = 0; j < sparseBlock.numRows(); j++) {
						if(sparseBlock.isEmpty(j))
							continue;

						LibMatrixBincellFFM.memorySegments1[j] = arena.allocateArray(ValueLayout.JAVA_DOUBLE,
							sparseBlock.values(j));
					}
					avg12 += (System.nanoTime() - t12) / 1_000_000_000.0;

					t13 = System.nanoTime();
					retFFM = mb1.scalarOperationsFFM(powerOpK, new MatrixBlock());
					avg13 += (System.nanoTime() - t13) / 1_000_000_000.0;

					arena.close();
				}

				for(int i = 0; i < 20; i++) {
					retScalar.reset(row, col, false);
					t2 = System.nanoTime();
					retScalar = mb1.scalarOperations(powerOpK, new MatrixBlock());
					avg2 += (System.nanoTime() - t2) / 1_000_000_000.0;
				}

				for(int i = 0; i < 20; i++) {
					retSIMD.reset(row, col, false);
					t3 = System.nanoTime();
					retSIMD = mb1.scalarOperationsSIMD(powerOpK, new MatrixBlock());
					avg3 += (System.nanoTime() - t3) / 1_000_000_000.0;
				}

				avg11 /= 20;
				avg12 /= 20;
				avg13 /= 20;
				avg2 /= 20;
				avg3 /= 20;

				boolean retEqual = compareResults(retFFM, retScalar, row, col);
				System.out.println("Results are equal: " + retEqual);
				System.out.println(row + "x" + col + " " + row + "x" + col);
				System.out.println("FFM1: " + avg11 + " s");
				System.out.println("FFM2: " + avg12 + " s");
				System.out.println("FFM3: " + avg13 + " s");
				System.out.println("Scalar: " + avg2 + " s");
				System.out.println("SIMD: " + avg3 + " s");

				writer.append(row + "," + col + "," + sparsity + "," + k + "," + avg11 + "," + avg12 + "," + avg13 + "," + avg2 + ","
					+ avg3 + "," + retEqual + "\n");
			}
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
    }

    private static void testMicrobenchmarkWarmup(int row, int col, int warmupRuns) {
        int amount = row * col;
        int size = amount * Double.BYTES;
        MatrixBlock mb = MatrixBlock.randOperations(row, col, 0.8, -10, 10, "uniform", 22);
        double[] values = mb.getDenseBlockValues();

        for(int i = 0; i < warmupRuns; i++) {
            // Allocate Memory with FFM API
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment segment = arena.allocateArray(ValueLayout.JAVA_DOUBLE, amount);
                segment.fill((byte) 0);
                for (int j = 0; j < amount; j++)
                    segment.setAtIndex(ValueLayout.JAVA_DOUBLE, j, values[j]);
            }
        }

        for(int i = 0; i < warmupRuns; i++) {
            // Allocate Memory with ByteBuffer
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(size);
            for (int j = 0; j < amount; j++)
                byteBuffer.putDouble(0);
            byteBuffer.clear();
            for (int j = 0; j < amount; j++)
                byteBuffer.putDouble(values[j]);
        }

        for(int i = 0; i < warmupRuns; i++) {
			// Allocate Memory with Unsafe
			long memoryAddress = UNSAFE.allocateMemory(size);
			UNSAFE.setMemory(memoryAddress, size, (byte) 0);
			for(int j = 0; j < amount; j++)
				UNSAFE.putDouble(memoryAddress + j * Double.BYTES, values[j]);
			UNSAFE.freeMemory(memoryAddress);
        }
    }

    private static void testNativeInvocationWarmup(int row, int col1, int col2, int k, int warmupRuns) {
        MatrixBlock mb1 = MatrixBlock.randOperations(row, col1, 0.8, -10, 10, "uniform", 6);
        MatrixBlock mb2 = MatrixBlock.randOperations(col1, col2, 0.8, -10, 10, "uniform", 8);
        MatrixBlock ret = new MatrixBlock(row, col2, false);

        LibMatrixNativeFFM.controlArena(true);
        LibMatrixNativeFFM.m1Segment = LibMatrixNativeFFM.arena.allocateArray(ValueLayout.JAVA_DOUBLE,
                mb1.getDenseBlockValues());
        LibMatrixNativeFFM.m2Segment = LibMatrixNativeFFM.arena.allocateArray(ValueLayout.JAVA_DOUBLE,
                mb2.getDenseBlockValues());
        LibMatrixNativeFFM.retSegment = LibMatrixNativeFFM.arena.allocateArray(ValueLayout.JAVA_DOUBLE, row*col2);

        for(int i = 0; i < warmupRuns; i++) {
            LibMatrixNative.matrixMult(mb1, mb2, ret, k);
            LibMatrixNativeFFM.matrixMult(mb1, mb2, ret, k);
        }

        LibMatrixNativeFFM.controlArena(false);
    }

    private static void testMemoryAccessMultWarmup(int row, int col1, int col2, int k, int warmupRuns) {
        MatrixBlock mb1 = MatrixBlock.randOperations(row, col1, 0.8, -10, 10, "uniform", 6);
        MatrixBlock mb2 = MatrixBlock.randOperations(col1, col2, 0.8, -10, 10, "uniform", 8);
        MatrixBlock ret = new MatrixBlock(row, col2, false);

        DenseBlock m1 = mb1.getDenseBlock();
        DenseBlock m2 = mb2.getDenseBlock();

        for(int i = 0; i < warmupRuns; i++) {
            Arena arena = Arena.ofShared();

            LibMatrixMultFFM.segment1 = arena.allocateArray(ValueLayout.JAVA_DOUBLE, m1.values(0));
            LibMatrixMultFFM.segment2 = arena.allocateArray(ValueLayout.JAVA_DOUBLE, m2.values(0));
            LibMatrixMultFFM.segmentRet = arena.allocateArray(ValueLayout.JAVA_DOUBLE, row*col2);

            LibMatrixMultFFM.matrixMult(mb1, mb2, ret, k);
            LibMatrixMult.matrixMult(mb1, mb2, ret, k);
            LibMatrixMultSIMD.matrixMult(mb1, mb2, ret, k);
            LibMatrixNative.matrixMult(mb1, mb2, ret, k);

            arena.close();
        }
    }

    private static void testMemoryAccessPowerWarmup(double sparsity, int row, int col, double exponent, int k, int warmupRuns) {
        MatrixBlock mb1 = MatrixBlock.randOperations(row, col, sparsity, -10, 10, "uniform", 6);
        RightScalarOperator powerOpK = new RightScalarOperator(Power.getPowerFnObject(), exponent, k);

        SparseBlock sparseBlock = mb1.getSparseBlock();
        for(int i = 0; i < warmupRuns; i++) {
            Arena arena = Arena.ofShared();

            LibMatrixBincellFFM.memorySegments1 = new MemorySegment[sparseBlock.numRows()];
            for(int j = 0; j < sparseBlock.numRows(); j++) {
                if(sparseBlock.isEmpty(j)) {
                    continue;
                }
                LibMatrixBincellFFM.memorySegments1[j] = arena.allocateArray(ValueLayout.JAVA_DOUBLE, sparseBlock.values(j));
            }

            mb1.scalarOperations(powerOpK, new MatrixBlock());
            mb1.scalarOperationsSIMD(powerOpK, new MatrixBlock());
            mb1.scalarOperationsFFM(powerOpK, new MatrixBlock());

            arena.close();
        }
    }

    private static String getTimeStamp() {
        LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss");
        return now.format(formatter);
    }

    private static boolean compareResults(MatrixBlock mb1, MatrixBlock mb2, int rows, int cols) {
        if(mb1.getNonZeros() != mb2.getNonZeros()) {
            System.out.println("non-zeroes: " + mb1.getNonZeros() + " != " + mb2.getNonZeros());
            return false;
        }

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if(Math.abs(mb1.get(i, j) - mb2.get(i, j)) > EPSILON) {
                    System.out.println("i=" + i + " j=" + j + ":" + mb1.get(i, j) + " is not equals " + mb2.get(i, j));
                    return false;
                }
            }
        }
        return true;
    }

    private static int[] calculateSizes(String stepStr) {
        String[] strArr = stepStr.split("-");
        int mode = 0;

        if (strArr.length == 1) {
            // stepStr only contains 1 value
            return new int[]{Integer.parseInt(strArr[0])};
        }

        // Continue with split string
        String[] strArr2 = null;
        if (strArr[1].contains("#")) {
            strArr2 = strArr[1].split("#");
        } else if (strArr[1].contains(":")) {
            strArr2 = strArr[1].split(":");
            mode = 1;
        }

        int start = Integer.parseInt(strArr[0]);
        int end = Integer.parseInt(strArr2[0]);
        int step = Integer.parseInt(strArr2[1]);

        int n = (mode == 0)
                ? ((end - start) / step) + 1
                : (int) (Math.log10(end / start) / Math.log10(step)) + 1;

        int[] sizes = new int[n];
        int i = 0;

        if (mode == 0) {
            for (; i < sizes.length; i++) {
                sizes[i] = start + step * i;
            }
        } else {
            for (; i < sizes.length; i++) {
                sizes[i] = (int) (start * Math.pow(step, i));
            }
        }
        return sizes;
    }

    private static void createBasePath() {
        try {
            Files.createDirectories(Paths.get(BASE_PATH));
        }
        catch(IOException e) {
            throw new RuntimeException(e);
        }
    }
}
