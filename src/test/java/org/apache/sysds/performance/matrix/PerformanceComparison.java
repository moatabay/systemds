package org.apache.sysds.performance.matrix;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;

public class PerformanceComparison {

//    static {
//        // Load the native library for JNI
//        System.loadLibrary("native");
//    }
//
//    // Declare the native method for JNI
//    public native int addJNI(int a, int b);
//
//    public static void main(String[] args) throws Throwable {
//        int a = 10, b = 20;
//        int iterations = 1000000; // Number of iterations for performance testing
//
//        PerformanceComparison pc = new PerformanceComparison();
//
//        // Measure time for JNI
//        long startTime = System.nanoTime();
//        for (int i = 0; i < iterations; i++) {
//            pc.addJNI(a, b);
//        }
//        long endTime = System.nanoTime();
//        long jniTime = endTime - startTime;
//
//        // Measure time for Foreign Memory and Function API
//        startTime = System.nanoTime();
//        try (MemorySession session = MemorySession.openConfined()) {
//            Linker linker = Linker.nativeLinker();
//            SymbolLookup lookup = linker.defaultLookup();
//
//            MethodHandle addMethod = linker.downcallHandle(
//                    lookup.find("add").orElseThrow(),
//                    MethodType.methodType(int.class, int.class, int.class),
//                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)
//            );
//
//            for (int i = 0; i < iterations; i++) {
//                addMethod.invoke(a, b);
//            }
//        }
//        endTime = System.nanoTime();
//        long foreignAPITime = endTime - startTime;
//
//        // Print performance comparison
//        System.out.println("JNI Time: " + jniTime + " ns");
//        System.out.println("Foreign API Time: " + foreignAPITime + " ns");
//    }
}
