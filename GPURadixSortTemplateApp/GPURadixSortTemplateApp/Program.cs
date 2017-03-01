using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GPGPUCollisionDetection;
using OpenCL.Net;

namespace GPURadixSortTemplateApp {

    class Program {
        private static OpenCL.Net.ErrorCode error;
        private static Device _device;

        private static Context cxGPUContext;
        private static CommandQueue cqCommandQueue;
        private static GPURadixSort sort;

        private static void ContextNotify(string errInfo, byte[] data, IntPtr cb, IntPtr userData) {
            Console.WriteLine("OpenCL Notification: " + errInfo);
        }

        static void CheckErr(ErrorCode err, string text) {
            Debug.WriteLine(err.ToString() + " Text: " + text);
        }

        public static void initOpenCL() {
            var platforms = Cl.GetPlatformIDs(out error);
            List<Device> devicesList = new List<Device>();

            CheckErr(error, "Cl.GetPlatformIDs");

            foreach (Platform platform in platforms) {
                string platformName = Cl.GetPlatformInfo(platform, PlatformInfo.Name, out error).ToString();
                Console.WriteLine("Platform: " + platformName);
                CheckErr(error, "Cl.GetPlatformInfo");
                //We will be looking only for GPU devices
                foreach (Device device in Cl.GetDeviceIDs(platform, DeviceType.Gpu, out error)) {
                    CheckErr(error, "Cl.GetDeviceIDs");

                    var vendor = Cl.GetDeviceInfo(device, DeviceInfo.Vendor, out error);
                    var name = Cl.GetDeviceInfo(device, DeviceInfo.Name, out error);
                    var worksize = Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkGroupSize, out error);
                    Console.WriteLine("Vendor: " + vendor + " , " + name);

                    Console.WriteLine("Device: " + device.GetType());
                    Console.WriteLine("Workgroupsize: " + worksize.CastTo<long>());
                    devicesList.Add(device);
                }
            }

            if (devicesList.Count <= 0) {
                Console.WriteLine("No devices found.");
                return;
            }

            _device = devicesList[0];

            if (Cl.GetDeviceInfo
                (_device,
                    DeviceInfo.ImageSupport,
                    out
                        error).CastTo<Bool>() == Bool.False) {
                Console.WriteLine("No image support.");
                return;
            }
            cxGPUContext = Cl.CreateContext(null, 1, new[] {_device}, ContextNotify, IntPtr.Zero, out error); //Second parameter is amount of devices
            CheckErr(error, "Cl.CreateContext");

            cqCommandQueue = Cl.CreateCommandQueue(cxGPUContext, _device, (CommandQueueProperties) 0, out error);
            CheckErr(error, "Cl.CreateCommandQueue");

            sort = new GPURadixSort(cqCommandQueue, cxGPUContext, _device);
        }


        static void Main(string[] args) {
            initOpenCL();


            // Simple sortTest
            List<int> numOfSortValues = new List<int>();
            for (int i = 0; i < 2; i++) {
                numOfSortValues.Add(50000000);
            }

            foreach (var numElements in numOfSortValues) {
                uint[] testKeys = new uint[numElements];
                ulong[] testValues = new ulong[numElements];
                uint[] testKeysOutput = new uint[numElements];
               // ulong[] testValuesOutput = new ulong[numElements];
                Random rnd = new Random();
                Console.WriteLine("\n New Sort test with {0} random values", numElements);
                Console.WriteLine("Init:");
                for (uint i = 0; i < numElements; i++) {
                    uint tmp = (uint) rnd.Next(10000);
                    testKeys[i] = tmp;
                    testValues[i] = tmp;
                    // Console.WriteLine("Key: {0} value: {1}", testKeys[i], testValues[i]);
                }


                var before = DateTime.Now;
                // create buffers 
                Console.WriteLine("Start creating Buffers for {0} values", numElements);
                // Create Buffers
                IMem cl_KeyMem = Cl.CreateBuffer
                    (cxGPUContext,
                        MemFlags.ReadWrite,
                        (IntPtr) (testKeys.Length*4),
                        testKeys,
                        out error);
                CheckErr(error, "Createbuffer");


                IMem cl_ValueMem = Cl.CreateBuffer
                    (cxGPUContext,
                        MemFlags.ReadWrite,
                        (IntPtr) (testKeys.Length*8),
                        testValues,
                        out error);
                CheckErr(error, "Createbuffer");

                IMem cl_KeyOutput = Cl.CreateBuffer
                    (cxGPUContext,
                        MemFlags.ReadWrite,
                        (IntPtr) (testKeys.Length*4),
                        testKeys,
                        out error);
                CheckErr(error, "Createbuffer");



                IMem cl_ValueOutput;


                Event eve;
                error = Cl.EnqueueWriteBuffer
                    (cqCommandQueue,
                        cl_KeyMem,
                        Bool.True,
                        IntPtr.Zero,
                        (IntPtr) (testKeys.Length*4),
                        testKeys,
                        0,
                        null,
                        out eve);
                CheckErr(error, "EnqBuffer");


                error = Cl.EnqueueWriteBuffer
                    (cqCommandQueue,
                        cl_ValueMem,
                        Bool.True,
                        IntPtr.Zero,
                        (IntPtr) (testKeys.Length*8),
                        testValues,
                        0,
                        null,
                        out eve);
                CheckErr(error, "EnqBuffer");


                error = Cl.Finish(cqCommandQueue);
                CheckErr(error, "Cl.Finish");
                Console.WriteLine
                    ("finished creating Buffers for {0} values after {1} ms -> Started sorting", numElements, (DateTime.Now - before).TotalMilliseconds);


                //                sort.sortKeysValue(cl_KeyMem, cl_ValueMem, testKeys.Length);
                sort.sortKeysOnly(cl_KeyMem, cl_KeyOutput, testKeys.Length);
                Console.WriteLine("finished sort for {0} values after {1} ms", numElements, (DateTime.Now - before).TotalMilliseconds);

                error = Cl.EnqueueReadBuffer
                    (cqCommandQueue,
                        cl_KeyMem,
                        Bool.True,
                        IntPtr.Zero,
                        (IntPtr) (testKeys.Length*4),
                        testKeysOutput,
                        0,
                        null,
                        out eve);
                CheckErr(error, "Cl.EnqueueReadBuffer");

                Console.WriteLine("Total sort time {0} ms", (DateTime.Now - before).TotalMilliseconds);


//                error = Cl.EnqueueReadBuffer(cqCommandQueue, cl_ValueMem, Bool.True, IntPtr.Zero, (IntPtr)(testKeys.Length * 8),
//                testValuesOutput, 0, null, out eve);
//                CheckErr(error, "Cl.EnqueueReadBuffer");


//                Array.Sort(testKeys);
//                Console.WriteLine("Sort finished");
//                for (uint i = 0; i < testKeys.Length; i++) {
//                    if (testKeysOutput[i] != testKeys[i]) {
//                        Console.WriteLine("keys not sorted");
//                    }
//                    ;
//                    //  Assert.True(testValuesOutput[i] == (uint)testKeys[i], "values not sorted");
//                }
                // Assert.False(testKeys == testValues, "did not work");
                Cl.ReleaseMemObject(cl_KeyMem);
                Cl.ReleaseMemObject(cl_ValueMem);
                Console.WriteLine("Run with {0} random numbers finished without errors", numElements);

                Cl.ReleaseMemObject(cl_KeyOutput);

//             Cl.ReleaseMemObject(cl_ValueOutput);
            }
        }
    }

}