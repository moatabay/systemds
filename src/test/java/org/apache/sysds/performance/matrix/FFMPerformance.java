package org.apache.sysds.performance.matrix;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import sun.misc.Unsafe;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.util.Arrays;
import java.util.Optional;

public class FFMPerformance {
    /* Matthias:
    first, micro benchmarking of allocations and then data access speed in
    elementwise and matmult operations. Yes, prototype-level or leaving it
    out is fine, if that way you complete the vector API in very good quality.

    Me:
    I looked through the code and I didn't see any spots where off-heap memory
    is allocated or accessed in the context of element-wise or matmult operations.
    I only saw on-heap memory allocations and accesses.

    I just wanted to make sure: The point is to compare the performance between
    the Java Foreign Memory API and the JNI (or ByteBuffer?) in terms of allocation
    and data access (of matrix blocks) from the off-heap memory, right?

    Matthias:
    there are multiple dimensions
    * compare data transfer for native BLAS operations via JNI with an
    alternative implementation with the Foreign memory API
    * implement our Java matrix multiplication and elementwise operations to
    work directly with the Foreign memory API.

    Me:
    As far as I understood, I need to implement three "kinds" of performance tests for the Foreign Memory API, right?

    Microbenchmark of allocations: For example the allocation of a 5000x5000 matrix in the Foreign Memory API versus ByteBuffers (or JNI?).
    Of course I will do it with multiple size.
    The invocation of native BLAS operations (for example "dmmdd") to compare the speed between JNI and the Foreign Memory API.
    And I still didn't really understand this part: "implement our Java matrix multiplication and elementwise operations to
    work directly with the Foreign memory API." because the Foreign Memory API is only there to invoke native functions and allocate off-heap memory.
    So I guess I didn't really understand how the Foreign Memory API can work with matrix multiplication (apart from calling LibMatrixNative)
    and the elementwise operations.

    Matthias:
    ad 3: you could change these operations to get every value over the
    foreign memory API (similar to using Unsafe), right? If so, we would
    want to quantify this overhead.
     */

    private static final Unsafe UNSAFE;

    static {
        try {
            // Use reflection to access the Unsafe instance
            Field field = Unsafe.class.getDeclaredField("theUnsafe");
            field.setAccessible(true);
            UNSAFE = (Unsafe) field.get(null);
        } catch (Exception e) {
            throw new RuntimeException("Unable to access Unsafe", e);
        }
    }

    public static void test() {
        int m = 2; // Rows of A (resulting matrix C)
        int n = 2; // Columns of B (resulting matrix C)
        int k = 3; // Columns of A (must match rows of B)

        // Example matrix data for A (2x3) and B (3x2)
        double[] matrixAData = {
                1.0, 2.0, 3.0,  // Row 1
                4.0, 5.0, 6.0   // Row 2
        };

        double[] matrixBData = {
                7.0, 8.0,       // Row 1
                9.0, 10.0,     // Row 2
                11.0, 12.0     // Row 3
        };

        // Allocate native memory for matrices A, B, and C
        try (Arena arena = Arena.ofConfined()) {
            // Allocate memory segments for A, B, and C
            MemorySegment a = arena.allocateArray(ValueLayout.JAVA_DOUBLE, matrixAData);
            MemorySegment b = arena.allocateArray(ValueLayout.JAVA_DOUBLE, matrixBData);
            MemorySegment c = arena.allocateArray(ValueLayout.JAVA_DOUBLE, new double[m * n]); // C will be 2x2

            // Leading dimensions
            int lda = k; // Leading dimension for A (number of rows of A)
            int ldb = k; // Leading dimension for B (number of rows of B)
            int ldc = n; // Leading dimension for C (number of rows of C)

            double alpha = 1.0; // Scalar for A * B
            double beta = 0.0;  // Scalar for C

            Linker linker = Linker.nativeLinker();
            SymbolLookup lookup = linker.defaultLookup();

            Optional<MethodHandle> addMethodHandle = lookup.find("dmmdd")
                    .map(symbol -> linker.downcallHandle(symbol, FunctionDescriptor.ofVoid(
                            ValueLayout.JAVA_INT,  // Layout for A
                            ValueLayout.JAVA_INT,  // Layout for B
                            ValueLayout.JAVA_INT,  // M
                            ValueLayout.JAVA_INT,  // N
                            ValueLayout.JAVA_INT,  // K
                            ValueLayout.JAVA_DOUBLE, // Alpha
                            ValueLayout.ADDRESS,    // A
                            ValueLayout.JAVA_INT,   // LDA
                            ValueLayout.ADDRESS,    // B
                            ValueLayout.JAVA_INT,   // LDB
                            ValueLayout.JAVA_DOUBLE, // Beta
                            ValueLayout.ADDRESS,    // C
                            ValueLayout.JAVA_INT     // LDC
                    )));

            if(addMethodHandle.isPresent()) {
                addMethodHandle.get().invoke(0, 0, m, n, k, alpha, a.address(),
                        lda,
                        b.address(),
                        ldb,
                        beta,
                        c.address(),
                        ldc);
            } else {
                System.out.println("Didn't find lookup");
            }

            // Read result from native memory
            double[] result = c.toArray(ValueLayout.JAVA_DOUBLE);
            // Process the result as needed
            System.out.println("Result Matrix C:");
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    System.out.print(result[i * n + j] + " ");
                }
                System.out.println();
            }
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public static void memoryAllocationTest(double sparsity, String rows, String cols, int warmupRuns) {
        MatrixBlock mb = MatrixBlock.randOperations(rows, cols, sparsity, -10, 10, "uniform", 93);

        int[] rowArr = calculateSizes(rows);
        int[] colArr = calculateSizes(cols);

        for(int row : rowArr) {
            for(int col : colArr) {

            }
        }

        int size = rows*cols*Double.BYTES;
        long t1, t2, t3;
        double avg1 = 0, avg2 = 0, avg3 = 0;

        // Allocate Memory in FFM API
        for(int i = 0; i < warmupRuns; i++) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment segment = arena.allocateArray(ValueLayout.JAVA_DOUBLE, mb.getDenseBlockValues());
            }
        }

        for(int i = 0; i < 10; i++) {
            t1 = System.nanoTime();
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment segment = arena.allocateArray(ValueLayout.JAVA_DOUBLE, mb.getDenseBlockValues());
            }
            avg1 += System.nanoTime() - t1;
        }


        // Allocate Memory in ByteBuffers
        for(int i = 0; i < warmupRuns; i++) {
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(size);
            DoubleBuffer doubleBuffer = byteBuffer.asDoubleBuffer();
            doubleBuffer.put(mb.getDenseBlockValues());
        }

        for(int i = 0; i < 10; i++) {
            t2 = System.nanoTime();
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(size);
            DoubleBuffer doubleBuffer = byteBuffer.asDoubleBuffer();
            doubleBuffer.put(mb.getDenseBlockValues());
            avg2 += System.nanoTime() - t2;
        }

        // Allocate Memory in Unsafe
        for(int i = 0; i < warmupRuns; i++) {
            long memoryAddress = UNSAFE.allocateMemory(size);

            for(int j = 0; j < rows*cols; j++) {
                UNSAFE.putDouble(memoryAddress + j * Double.BYTES, mb.getDenseBlockValues()[j]);
            }

            UNSAFE.freeMemory(memoryAddress);
        }

        for(int i = 0; i < 10; i++) {
            t3 = System.nanoTime();
            long memoryAddress = UNSAFE.allocateMemory(size);

            for(int j = 0; j < rows*cols; j++) {
                UNSAFE.putDouble(memoryAddress + j * Double.BYTES, mb.getDenseBlockValues()[j]);
            }
            avg3 += System.nanoTime() - t3;
            UNSAFE.freeMemory(memoryAddress);
        }

        avg1 /= 10;
        avg2 /= 10;
        avg3 /= 10;

        System.out.println("FFM API: " + avg1);
        System.out.println("ByteBuffer: " + avg2);
        System.out.println("UNSAFE: " + avg3);
    }

    public static void test2() {
        MemoryLayout POINT_2D = MemoryLayout.structLayout(
                ValueLayout.JAVA_DOUBLE.withName("x"),
                ValueLayout.JAVA_DOUBLE.withName("y")
        );

        VarHandle xHandle = POINT_2D.varHandle(MemoryLayout.PathElement.groupElement("x"));
        VarHandle yHandle = POINT_2D.varHandle(MemoryLayout.PathElement.groupElement("y"));

        try(Arena arena = Arena.ofConfined()) {
            MemorySegment point = arena.allocate(POINT_2D);
            xHandle.set(point, 0, 3d);
            yHandle.set(point, 0, 4d);

            System.out.println(xHandle.toString());
            System.out.println(yHandle.toString());

        }

    }

    private static int[] calculateSizes(String stepStr) {
        String[] strArr = stepStr.split("-");

        if(strArr.length == 1) {
            // stepStr only contains 1 value
            return new int[] {Integer.parseInt(strArr[0])};
        }

        // Continue with split string
        String[] strArr2 = strArr[1].split("#");
        int start = Integer.valueOf(strArr[0]);
        int end = Integer.valueOf(strArr2[0]);
        int step = Integer.valueOf(strArr2[1]);

        int n = ((int) (end - start) / step) + 1;

        int[] sizes = new int[n];
        for(int i = 0; i < sizes.length; i++) {
            sizes[i] = start + step * i;
        }
        return sizes;
    }
}
