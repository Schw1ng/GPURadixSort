using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using OpenCL.Net;
using Environment = System.Environment;

namespace GPGPUCollisionDetection
{
    public class GPURadixSort
    {
        private static int Log_Idx = 0;
        private const bool DEBUG = false;
        private const bool DEBUG_CONSOLE_OUTPUT = false;
        private readonly string debugLog = Path.Combine(Environment.CurrentDirectory , "OpenCLDebugLog.txt");
        private static string sortLog = Path.Combine(Environment.CurrentDirectory, "sortLog");
        private string programPath = Path.Combine(Environment.CurrentDirectory, @"..\..\..\GPURadixSort\RadixSort.cl");

        public struct GPUConstants
        {
             public int numRadices;
             public int numBlocks;
             public int numGroupsPerBlock;
             public int R;
             public int numThreadsPerGroup;
             public int numElementsPerGroup;
			 public int numRadicesPerBlock;
             public int bitMask;
             public int L;
             public int numThreadsPerBlock;
            public int numTotalElements;
        }


        private Context cxGPUContext { get; set; }
        private CommandQueue cqCommandQueue { get; set; }
        private Device _device { get; set; }

        private GPUConstants gpuConstants;


        private IMem mInputBuff;
        private IMem mCounters;
        private IMem mRadixPrefixes;
        private IMem mOutputBuff;
        //private IMem mBlockOffsets;

        private const int numCounters = num_Radices*NumGroupsPerBlock*numBlocks;

        // Anzahl an Bits die als Buckets für jeden radix durchlauf verwendet werden
        private const int radix_BitsL = 4;
        private const int num_Radices = 1 << radix_BitsL;
       
        // Auch Thread Blocks unter CUDA -> Gruppe von Threads mit gemeinsamen shared memory.
        private const int numBlocks = 16;

        // Anzahl von WorkItems / Threads, die sich in einer Work-Group befinden
        private const int numThreadsPerBlock = 512;

        private const int R = 8;

        private const int NumGroupsPerBlock = numThreadsPerBlock / R;
        
        private const int BIT_MASK_START = 0xF;

        int[] counters = new int[numCounters];




        Kernel ckSetupAndCount; // OpenCL kernels
        Kernel ckSumIt;
        Kernel ckReorderingKeysOnly;
        Kernel ckReorderingKeyValue;
        //Kernel ckReorderDataKeysOnly;


        private int maxElements;
        private int[] debugRead;




        /// <summary>
        /// 
        /// This method is used to check the opencl return values
        /// 
        /// </summary>
        /// <param name="err"></param>
        /// <param name="name"></param>
         
        private void CheckErr(ErrorCode err, string name)
        {
            StackFrame frame = new StackFrame(1);
            var method = frame.GetMethod();
            var type = method.DeclaringType;
            var nameCaller = method.Name;
            if (err != ErrorCode.Success)
            {
                Debug.WriteLine(err.ToString() + " Text: " + name+" from : "+type +"."+nameCaller);
                if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine(err.ToString() + " Text: " + name+" from : "+type +"."+nameCaller);
//                if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("error: " + name + " (" + err + ")");
                using (StreamWriter sw = File.AppendText(debugLog))
                {
                    sw.WriteLine("error: " + name + " (" + err + ")");
                }
            }
            //Guid id;
            //id.
        }

        public GPURadixSort(
             CommandQueue commandQue,
             Context context,
            Device device
            )
        {
          
            gpuConstants = new GPUConstants();
            gpuConstants.L = radix_BitsL;
            gpuConstants.numGroupsPerBlock = NumGroupsPerBlock;
            gpuConstants.R = R;
            gpuConstants.numThreadsPerGroup = numThreadsPerBlock/NumGroupsPerBlock;
            gpuConstants.numThreadsPerBlock = numThreadsPerBlock;
            gpuConstants.numBlocks = numBlocks;
            gpuConstants.numRadices = num_Radices;
            gpuConstants.numRadicesPerBlock = num_Radices/numBlocks;
            gpuConstants.bitMask = BIT_MASK_START;
            counters.Initialize();
            OpenCL.Net.ErrorCode error;
            cxGPUContext = context;
            cqCommandQueue = commandQue;
            _device = device;
            //Create a command queue, where all of the commands for execution will be added
            /*cqCommandQueue = Cl.CreateCommandQueue(cxGPUContext, _device, (CommandQueueProperties)0, out  error);
            CheckErr(error, "Cl.CreateCommandQueue");*/
            string programSource = System.IO.File.ReadAllText(programPath);
             IntPtr[] progSize = new IntPtr[] { (IntPtr)programSource.Length };
            OpenCL.Net.Program clProgramRadix = Cl.CreateProgramWithSource(cxGPUContext, 1, new[] { programSource },progSize,
                out error);
            CheckErr(error,"createProgramm");
            string flags = "-cl-fast-relaxed-math";

                error = Cl.BuildProgram(clProgramRadix, 1, new[] { _device }, flags, null, IntPtr.Zero);
            CheckErr(error, "Cl.BuildProgram");
            //Check for any compilation errors
            if (Cl.GetProgramBuildInfo(clProgramRadix, _device, ProgramBuildInfo.Status, out error).CastTo<BuildStatus>()
                != BuildStatus.Success)
            {
                CheckErr(error, "Cl.GetProgramBuildInfo");
                if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("Cl.GetProgramBuildInfo != Success");
                if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine(Cl.GetProgramBuildInfo(clProgramRadix, _device, ProgramBuildInfo.Log, out error));
                return;
            }
            int ciErrNum;


            ckSetupAndCount = Cl.CreateKernel(clProgramRadix, "SetupAndCount", out error);
            CheckErr(error, "Cl.CreateKernel");
            ckSumIt = Cl.CreateKernel(clProgramRadix, "SumIt", out error);
            CheckErr(error, "Cl.CreateKernel");
            ckReorderingKeysOnly = Cl.CreateKernel(clProgramRadix, "ReorderingKeysOnly", out error);
            CheckErr(error, "Cl.CreateKernel");
            ckReorderingKeyValue = Cl.CreateKernel(clProgramRadix, "ReorderingKeyValue", out error);
            CheckErr(error, "Cl.CreateKernel");
        }






       public void sortKeysOnly(IMem input, IMem output,
                    int numElements)
        {
            debugRead = new int[Math.Max(numElements,numCounters)];
            OpenCL.Net.ErrorCode error;
           Event eve;

           mCounters = Cl.CreateBuffer(cxGPUContext, MemFlags.ReadWrite, gpuConstants.numGroupsPerBlock * gpuConstants.numRadices * gpuConstants.numBlocks * sizeof(int),
            out error);
           CheckErr(error, "Cl.CreateBuffer");

           mRadixPrefixes = Cl.CreateBuffer(cxGPUContext, MemFlags.ReadWrite, gpuConstants.numRadices * sizeof(int), out error);
           CheckErr(error, "Cl.CreateBuffer");

           gpuConstants.numElementsPerGroup = (numElements/(gpuConstants.numBlocks*gpuConstants.numGroupsPerBlock)) +1 ;
           gpuConstants.numTotalElements = numElements;

           if (DEBUG) {
                Cl.EnqueueReadBuffer(cqCommandQueue, input, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4), debugRead, 0,
                    null, out eve);
                CheckErr(error, "Cl.EnqueueReadBuffer");
                PrintAsArray(debugRead, gpuConstants.numTotalElements);
            }
            int i;
           for (i = 0; i < 8; i++)
           {
               error = Cl.EnqueueWriteBuffer(cqCommandQueue, mCounters, Bool.True, IntPtr.Zero, (IntPtr)(numCounters * 4),
                counters, 0, null, out eve);
               CheckErr(error, "Cl.EnqueueWriteBuffer Counter initialize");
               if (i%2 == 0)
               {
                   DateTime before = DateTime.Now;
                   SetupAndCount(input, 4 * i);
                   if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("Setup and Count =" + (DateTime.Now - before).TotalMilliseconds);

                   before = DateTime.Now;
                   SumIt(input, 4 * i);
                   if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("SumIt =" + (DateTime.Now - before).TotalMilliseconds);
                   
                   before = DateTime.Now;
                   ReorderingKeysOnly(input, output, 4 * i);
                   if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("Reorder =" + (DateTime.Now - before).TotalMilliseconds);

               }
               else
               {
                   SetupAndCount(output, 4 * i);
                   SumIt(output, 4 * i);
                   ReorderingKeysOnly(output, input, 4 * i);
               }

           }
           if (i%2 != 0)
           {
               error= Cl.EnqueueCopyBuffer(cqCommandQueue, input, output, IntPtr.Zero, IntPtr.Zero, (IntPtr)(numElements * 4), 0, null, out eve);
               CheckErr(error, "Cl.EnqueueCopyBuffer");
               error = Cl.Finish(cqCommandQueue);
               CheckErr(error, "Cl.Finish Copybuffer");
           }
           error = Cl.ReleaseMemObject(mRadixPrefixes);
           CheckErr(error, "Cl.ReleaseMemObj");
           error = Cl.ReleaseMemObject(mCounters);
           CheckErr(error, "Cl.ReleaseMemObj");
           Log_Idx++;

       }


       public void sortKeysValue(IMem inputKey, IMem outputKey,IMem inputValue, IMem outputValue,
                int numElements)
       {
           debugRead = new int[Math.Max(numElements, numCounters)];
           OpenCL.Net.ErrorCode error;
           Event eve;
           mCounters = Cl.CreateBuffer(cxGPUContext, MemFlags.ReadWrite, gpuConstants.numGroupsPerBlock * gpuConstants.numRadices * gpuConstants.numBlocks * sizeof(int),
                out error);
           CheckErr(error, "Cl.CreateBuffer");

           mRadixPrefixes = Cl.CreateBuffer(cxGPUContext, MemFlags.ReadWrite, gpuConstants.numRadices * sizeof(int), out error);
           CheckErr(error, "Cl.CreateBuffer");

            /*
                       error = Cl.EnqueueReadBuffer(cqCommandQueue, input, Bool.True, IntPtr.Zero, (IntPtr)(numElements * 4),
                debugRead, 0, null, out eve);
                       CheckErr(error, "Cl.EnqueueReadBuffer");
            */
            if (DEBUG)
            {
                Cl.EnqueueReadBuffer(cqCommandQueue, inputKey, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4), debugRead, 0,
                    null, out eve);
                CheckErr(error, "Cl.EnqueueReadBuffer");
                PrintAsArray(debugRead, gpuConstants.numTotalElements);
            }
            gpuConstants.numElementsPerGroup = (numElements / (gpuConstants.numBlocks * gpuConstants.numGroupsPerBlock)) + 1;
           gpuConstants.numTotalElements = numElements;
           int i;
           for (i = 0; i < 8; i++)
           {
               error = Cl.EnqueueWriteBuffer(cqCommandQueue, mCounters, Bool.True, IntPtr.Zero, (IntPtr)(numCounters * 4),
                counters, 0, null, out eve);
               CheckErr(error, "Cl.EnqueueWriteBuffer Counter initialize");
               if (i % 2 == 0)
               {
                   SetupAndCount(inputKey, 4 * i);
                   SumIt(inputKey, 4 * i);
                   ReorderingKeyValue(inputKey, outputKey,inputValue,outputValue, 4 * i);
               }
               else
               {
                   SetupAndCount(outputKey, 4 * i);
                   SumIt(outputKey, 4 * i);
                   ReorderingKeyValue(outputKey, inputKey,outputValue,inputValue, 4 * i);
               }

           }
           if (i % 2 != 0)
           {
               error = Cl.EnqueueCopyBuffer(cqCommandQueue, inputKey, outputKey, IntPtr.Zero, IntPtr.Zero, (IntPtr)(numElements * 4), 0, null, out eve);
               CheckErr(error, "Cl.EnqueueCopyBuffer");
               error = Cl.Finish(cqCommandQueue);
               CheckErr(error, "Cl.Finish Copybuffer");
               error = Cl.EnqueueCopyBuffer(cqCommandQueue, inputValue, outputValue, IntPtr.Zero, IntPtr.Zero, (IntPtr)(numElements * 8), 0, null, out eve);
               CheckErr(error, "Cl.EnqueueCopyBuffer");
               error = Cl.Finish(cqCommandQueue);
               CheckErr(error, "Cl.Finish Copybuffer");
           }
           error = Cl.ReleaseMemObject(mRadixPrefixes);
           CheckErr(error, "Cl.ReleaseMemObj");
           error = Cl.ReleaseMemObject(mCounters);
           CheckErr(error, "Cl.ReleaseMemObj");

            Log_Idx++;

        }


        public void sortKeysValue(IMem key, IMem value,  
              int numElements)
       {
           debugRead = new int[Math.Max(numElements, numCounters)];
           OpenCL.Net.ErrorCode error;
           Event eve;
           /*
                      error = Cl.EnqueueReadBuffer(cqCommandQueue, input, Bool.True, IntPtr.Zero, (IntPtr)(numElements * 4),
               debugRead, 0, null, out eve);
                      CheckErr(error, "Cl.EnqueueReadBuffer");
           */

           mCounters = Cl.CreateBuffer(cxGPUContext, MemFlags.ReadWrite, gpuConstants.numGroupsPerBlock * gpuConstants.numRadices * gpuConstants.numBlocks * sizeof(int),
                out error);
           CheckErr(error, "Cl.CreateBuffer");

           mRadixPrefixes = Cl.CreateBuffer(cxGPUContext, MemFlags.ReadWrite, gpuConstants.numRadices * sizeof(int), out error);
           CheckErr(error, "Cl.CreateBuffer");
            IMem outputValue = Cl.CreateBuffer(cxGPUContext, MemFlags.ReadWrite, (IntPtr)(8 * numElements),
                out error);
            CheckErr(error, "Cl.CreateBuffer");
            IMem outputKey = Cl.CreateBuffer(cxGPUContext, MemFlags.ReadWrite, (IntPtr)(4 * numElements),
                out error);
           CheckErr(error, "Cl.CreateBuffer");


           gpuConstants.numElementsPerGroup = (numElements / (gpuConstants.numBlocks * gpuConstants.numGroupsPerBlock)) + 1;
           gpuConstants.numTotalElements = numElements;
           int i;
           for (i = 0; i < 8; i++)
           {
               error = Cl.EnqueueWriteBuffer(cqCommandQueue, mCounters, Bool.True, IntPtr.Zero, (IntPtr)(numCounters * 4),
                counters, 0, null, out eve);
               CheckErr(error, "Cl.EnqueueWriteBuffer Counter initialize");
               if (i % 2 == 0)
               {
                   SetupAndCount(key, 4 * i);
                   SumIt(key, 4 * i);
                   ReorderingKeyValue(key, outputKey, value, outputValue, 4 * i);
               }
               else
               {
                   SetupAndCount(outputKey, 4 * i);
                   SumIt(outputKey, 4 * i);
                   ReorderingKeyValue(outputKey, key, outputValue, value, 4 * i);
               }

           }
           if (i % 2 == 0)
           {
               error = Cl.EnqueueCopyBuffer(cqCommandQueue, outputKey, key, IntPtr.Zero, IntPtr.Zero, (IntPtr)(numElements * 4), 0, null, out eve);
               CheckErr(error, "Cl.EnqueueCopyBuffer");
               error = Cl.Finish(cqCommandQueue);
               CheckErr(error, "Cl.Finish Copybuffer");
               error = Cl.EnqueueCopyBuffer(cqCommandQueue, outputValue, value, IntPtr.Zero, IntPtr.Zero, (IntPtr)(numElements * 8), 0, null, out eve);
               CheckErr(error, "Cl.EnqueueCopyBuffer");
               error = Cl.Finish(cqCommandQueue);
               CheckErr(error, "Cl.Finish Copybuffer");
           }


           error = Cl.ReleaseMemObject(outputKey);
           CheckErr(error, "Cl.ReleaseMemObj");
           error = Cl.ReleaseMemObject(outputValue);
           CheckErr(error, "Cl.ReleaseMemObj");
           error = Cl.ReleaseMemObject(mRadixPrefixes);
           CheckErr(error, "Cl.ReleaseMemObj");
           error = Cl.ReleaseMemObject(mCounters);
           CheckErr(error, "Cl.ReleaseMemObj");
            Log_Idx++;

        }




        private void ReorderingKeysOnly(IMem input, IMem output, int bitOffset)
       {
           OpenCL.Net.ErrorCode error;
           IntPtr agentPtrSize = (IntPtr)0;
           agentPtrSize = (IntPtr)Marshal.SizeOf(typeof(IntPtr));
           var ptrSize = (IntPtr)Marshal.SizeOf(typeof(Mem));


           int globalWorkSize = gpuConstants.numThreadsPerBlock * gpuConstants.numBlocks;
           int localWorkSize = gpuConstants.numThreadsPerBlock;

           IntPtr[] workGroupSizePtr = new IntPtr[] { (IntPtr)globalWorkSize };
           IntPtr[] localWorkGroupSizePtr = new IntPtr[] { (IntPtr)localWorkSize };
           Event clevent;
           error = Cl.SetKernelArg(ckReorderingKeysOnly, 0, ptrSize, input);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckReorderingKeysOnly, 1, ptrSize, output);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckReorderingKeysOnly, 2, ptrSize, mCounters);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckReorderingKeysOnly, 3, ptrSize, mRadixPrefixes);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckReorderingKeysOnly, 4, (IntPtr)(gpuConstants.numGroupsPerBlock * gpuConstants.numBlocks * gpuConstants.numRadicesPerBlock * 4), null);
            CheckErr(error, "Cl.SetKernelArg");
            error = Cl.SetKernelArg(ckReorderingKeysOnly, 5, (IntPtr)(gpuConstants.numRadices * 4), null);
            CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckReorderingKeysOnly, 6, (IntPtr)(Marshal.SizeOf(typeof(GPUConstants))), gpuConstants);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckReorderingKeysOnly, 7, (IntPtr)4, bitOffset);
           CheckErr(error, "Cl.SetKernelArg");

           error = Cl.EnqueueNDRangeKernel(cqCommandQueue, ckReorderingKeysOnly, 1, null, workGroupSizePtr, localWorkGroupSizePtr, 0, null, out clevent);
           CheckErr(error, "Cl.EnqueueNDRangeKernel");

           error = Cl.Finish(cqCommandQueue);
           CheckErr(error, "Cl.Finish ReorderKeysOnly");
           if (DEBUG)
           {
                
               //if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("-------------------------------Reordering-------------------------------------------------");
               Event eve;
               
               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("              Input                ");
               Cl.EnqueueReadBuffer(cqCommandQueue, input, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4), debugRead, 0,
                   null, out eve);
               CheckErr(error, "Cl.EnqueueReadBuffer");
               PrintElementBuffer(debugRead, gpuConstants.numTotalElements, "Reordering -> Input -> bitoffset = " + bitOffset);

                if (DEBUG_CONSOLE_OUTPUT) Console.WriteLine("              Counters                ");
               Cl.EnqueueReadBuffer(cqCommandQueue, mCounters, Bool.True, IntPtr.Zero, (IntPtr)(numCounters * sizeof(int)), debugRead, 0,
                   null, out eve);
               CheckErr(error, "Cl.EnqueueReadBuffer");
               PrintCounterBuffer(debugRead, "Reordering -> bitoffset = " + bitOffset);

               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("              Counters                ");
               Cl.EnqueueReadBuffer(cqCommandQueue, mRadixPrefixes, Bool.True, IntPtr.Zero, (IntPtr)( gpuConstants.numRadices * sizeof(int)), debugRead, 0,
                   null, out eve);
               CheckErr(error, "Cl.EnqueueReadBuffer");
               PrintElementBuffer(debugRead,gpuConstants.numRadices, "Reordering -> RadixPrefixe -> bitoffset = " + bitOffset);



               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("              Output                ");
               Cl.EnqueueReadBuffer(cqCommandQueue, output, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4), debugRead, 0,
                   null, out eve);
               CheckErr(error, "Cl.EnqueueReadBuffer");
               PrintElementBuffer(debugRead, gpuConstants.numTotalElements, "Reordering -> Output -> bitoffset = " + bitOffset);


               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("Reordering -> bitoffset = " + bitOffset);
               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine();
           };

        }

        private void ReorderingKeyValue(IMem inputKey, IMem outputKey,IMem inputValue, IMem outputValue, int bitOffset)
       {
           OpenCL.Net.ErrorCode error;
           IntPtr agentPtrSize = (IntPtr)0;
           agentPtrSize = (IntPtr)Marshal.SizeOf(typeof(IntPtr));
           var ptrSize = (IntPtr)Marshal.SizeOf(typeof(Mem));


           int globalWorkSize = gpuConstants.numThreadsPerBlock * gpuConstants.numBlocks;
           int localWorkSize = gpuConstants.numThreadsPerBlock;

           IntPtr[] workGroupSizePtr = new IntPtr[] { (IntPtr)globalWorkSize };
           IntPtr[] localWorkGroupSizePtr = new IntPtr[] { (IntPtr)localWorkSize };
           Event clevent;
           error = Cl.SetKernelArg(ckReorderingKeyValue, 0, ptrSize, inputKey);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckReorderingKeyValue, 1, ptrSize, outputKey);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckReorderingKeyValue, 2, ptrSize, inputValue);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckReorderingKeyValue, 3, ptrSize, outputValue);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckReorderingKeyValue, 4, ptrSize, mCounters);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckReorderingKeyValue, 5, ptrSize, mRadixPrefixes);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckReorderingKeyValue, 6, (IntPtr)(gpuConstants.numGroupsPerBlock * gpuConstants.numBlocks * gpuConstants.numRadicesPerBlock * 4), null);
           CheckErr(error, "Cl.SetKernelArg");
            error = Cl.SetKernelArg(ckReorderingKeyValue, 7, (IntPtr)(gpuConstants.numRadices * 4), null);
            CheckErr(error, "Cl.SetKernelArg");
            error = Cl.SetKernelArg(ckReorderingKeyValue, 8, (IntPtr)(Marshal.SizeOf(typeof(GPUConstants))), gpuConstants);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckReorderingKeyValue, 9, (IntPtr)4, bitOffset);
           CheckErr(error, "Cl.SetKernelArg");

           error = Cl.EnqueueNDRangeKernel(cqCommandQueue, ckReorderingKeyValue, 1, null, workGroupSizePtr, localWorkGroupSizePtr, 0, null, out clevent);
           CheckErr(error, "Cl.EnqueueNDRangeKernel");

           error = Cl.Finish(cqCommandQueue);
           CheckErr(error, "Cl.Finish");
           if (DEBUG)
           {
               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("-------------------------------Reordering-------------------------------------------------");
               Event eve;

               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("              Input                ");
               Cl.EnqueueReadBuffer(cqCommandQueue, inputKey, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4), debugRead, 0,
                   null, out eve);
               CheckErr(error, "Cl.EnqueueReadBuffer");
               PrintElementBuffer(debugRead, gpuConstants.numTotalElements, "Reordering -> Input -> bitoffset = " + bitOffset);
              
               Cl.EnqueueReadBuffer(cqCommandQueue, inputValue, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4), debugRead, 0,
                   null, out eve);
               CheckErr(error, "Cl.EnqueueReadBuffer");
               PrintElementBuffer(debugRead, gpuConstants.numTotalElements, "Reordering -> InputValues -> bitoffset = " + bitOffset);

               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("              Counters                ");
               Cl.EnqueueReadBuffer(cqCommandQueue, mCounters, Bool.True, IntPtr.Zero, (IntPtr)(numCounters * sizeof(int)), debugRead, 0,
                   null, out eve);
               CheckErr(error, "Cl.EnqueueReadBuffer");
               PrintCounterBuffer(debugRead, "Reordering -> bitoffset = " + bitOffset);

               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("              Counters                ");
               Cl.EnqueueReadBuffer(cqCommandQueue, mRadixPrefixes, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numRadices * sizeof(int)), debugRead, 0,
                   null, out eve);
               CheckErr(error, "Cl.EnqueueReadBuffer");
               PrintElementBuffer(debugRead, gpuConstants.numRadices, "Reordering -> RadixPrefixe -> bitoffset = " + bitOffset);



               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("              Output                ");
               Cl.EnqueueReadBuffer(cqCommandQueue, outputKey, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4), debugRead, 0,
                   null, out eve);
               CheckErr(error, "Cl.EnqueueReadBuffer");
               PrintElementBuffer(debugRead, gpuConstants.numTotalElements, "Reordering -> Output -> bitoffset = " + bitOffset);

               Cl.EnqueueReadBuffer(cqCommandQueue, outputValue, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4), debugRead, 0,
                   null, out eve);
               CheckErr(error, "Cl.EnqueueReadBuffer");
               PrintElementBuffer(debugRead, gpuConstants.numTotalElements, "Reordering -> OutputValue -> bitoffset = " + bitOffset);

               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("Reordering -> bitoffset = " + bitOffset);
               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine();
           };

        }
        private void SumIt(IMem input, int bitOffset)
       {
           OpenCL.Net.ErrorCode error;
           IntPtr agentPtrSize = (IntPtr)0;
           agentPtrSize = (IntPtr)Marshal.SizeOf(typeof(IntPtr));
           var ptrSize = (IntPtr)Marshal.SizeOf(typeof(Mem));


           int globalWorkSize = gpuConstants.numThreadsPerBlock * gpuConstants.numBlocks;
           int localWorkSize = gpuConstants.numThreadsPerBlock;

           IntPtr[] workGroupSizePtr = new IntPtr[] { (IntPtr)globalWorkSize };
           IntPtr[] localWorkGroupSizePtr = new IntPtr[] { (IntPtr)localWorkSize };
           Event clevent;
           error = Cl.SetKernelArg(ckSumIt, 0, ptrSize, input);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckSumIt, 1, ptrSize, mCounters);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckSumIt, 2, ptrSize, mRadixPrefixes);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckSumIt, 3, (IntPtr)(Marshal.SizeOf(typeof(GPUConstants))), gpuConstants);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckSumIt, 4, (IntPtr)4, bitOffset);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckSumIt, 5, (IntPtr)(4* gpuConstants.numBlocks*gpuConstants.numGroupsPerBlock), null);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.EnqueueNDRangeKernel(cqCommandQueue, ckSumIt, 1, null, workGroupSizePtr, localWorkGroupSizePtr, 0, null, out clevent);
           CheckErr(error, "Cl.EnqueueNDRangeKernel");
           
           error = Cl.Finish(cqCommandQueue);
           CheckErr(error, "Cl.Finish");
           if (DEBUG)
           {
               Event eve;

               error = Cl.EnqueueReadBuffer(cqCommandQueue, input, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4),
                debugRead, 0, null, out eve);
                CheckErr(error, "Cl.EnqueueReadBuffer");

                PrintElementBuffer(debugRead, gpuConstants.numTotalElements, "SumIt -> Input -> bitoffset = " + bitOffset);

               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("              Counters                ");
               Cl.EnqueueReadBuffer(cqCommandQueue, mCounters, Bool.True, IntPtr.Zero, (IntPtr)(numCounters * sizeof(int)), debugRead, 0,
                   null, out eve);
               CheckErr(error, "Cl.EnqueueReadBuffer");
               PrintCounterBuffer(debugRead, "SumIt -> bitoffset = " + bitOffset);
               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("SumIt -> bitoffset = " + bitOffset);
                Cl.EnqueueReadBuffer(cqCommandQueue, mRadixPrefixes, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numRadices * sizeof(int)), debugRead, 0,
                    null, out eve);
                CheckErr(error, "Cl.EnqueueReadBuffer");
                PrintElementBuffer(debugRead, gpuConstants.numRadices, "SumIt -> RadixPrefixe -> bitoffset = " + bitOffset);
                if (DEBUG_CONSOLE_OUTPUT) Console.WriteLine();    
           };
       }

       private void SetupAndCount(IMem input, int bitOffset)
       {
           OpenCL.Net.ErrorCode error;
           IntPtr agentPtrSize = (IntPtr)0;
           agentPtrSize = (IntPtr)Marshal.SizeOf(typeof(IntPtr));
           var ptrSize = (IntPtr)Marshal.SizeOf(typeof(Mem));


           int globalWorkSize = gpuConstants.numThreadsPerBlock * gpuConstants.numBlocks;
           int localWorkSize = gpuConstants.numThreadsPerBlock;

           IntPtr[] workGroupSizePtr = new IntPtr[] { (IntPtr)globalWorkSize };
           IntPtr[] localWorkGroupSizePtr = new IntPtr[] { (IntPtr)localWorkSize };
           Event clevent;
           error = Cl.SetKernelArg(ckSetupAndCount, 0, ptrSize, input);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckSetupAndCount, 1, ptrSize, mCounters);
           CheckErr(error, "Cl.SetKernelArg");
           //if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine((Marshal.SizeOf(typeof(GPUConstants))));
           error = Cl.SetKernelArg(ckSetupAndCount, 2, (IntPtr)(Marshal.SizeOf(typeof(GPUConstants))), gpuConstants);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.SetKernelArg(ckSetupAndCount, 3, (IntPtr)4, bitOffset);
           CheckErr(error, "Cl.SetKernelArg");
           error = Cl.EnqueueNDRangeKernel(cqCommandQueue, ckSetupAndCount, 1, null, workGroupSizePtr, localWorkGroupSizePtr, 0, null, out clevent);
           CheckErr(error, "Cl.EnqueueNDRangeKernel");

           error = Cl.Finish(cqCommandQueue);
           CheckErr(error, "Cl.Finish");
           if (DEBUG)
           {
               Event eve;
               Cl.EnqueueReadBuffer(cqCommandQueue, input, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4), debugRead, 0,
                   null, out eve);
               CheckErr(error, "Cl.EnqueueReadBuffer");
               PrintElementBuffer(debugRead, gpuConstants.numTotalElements, "Setup and Count -> Input  -> bitoffset = " + bitOffset);              

               Cl.EnqueueReadBuffer(cqCommandQueue, mCounters, Bool.True, IntPtr.Zero, (IntPtr)(numCounters * sizeof(int)), debugRead, 0,
                   null, out eve);
               CheckErr(error, "Cl.EnqueueReadBuffer");
               PrintCounterBuffer(debugRead,"Setup and Count -> bitoffset = "+bitOffset);
               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("Setup and Count -> bitoffset = " + bitOffset);

               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine();    
           }
           
       }

        private void PrintElementBuffer(int[] printData , int count, string caption)
        {
            String output = caption;
            output += "\n";
            output += "----------------------------------------------------------------------------------------------------------------------------------------------------------";
            for (int i = 0; i < count; i++)
            {
                if (i%20 == 0) output += "\n";
                output += String.Format("{0,5:x} ", printData[i]);
                
            }
            output += "\n";
            output += "----------------------------------------------------------------------------------------------------------------------------------------------------------\n";
            using (StreamWriter sw = File.AppendText(sortLog + Log_Idx + ".txt"))
            {
                sw.WriteLine(output);
            }	
        }


        private void PrintCounterBuffer(int[] printData, string caption)
        {
            String output = caption;
            output += "\n";
            output += "----------------------------------------------------------------------------------------------------------------------------------------------------------\n";
            
            output += String.Format("{0,10}","");
            for (int j = 0; j < numBlocks; j++)
            {
                output += String.Format("{0,29}", "Block "+j);
            }
            output += "\n";
            output += "----------------------------------------------------------------------------------------------------------------------------------------------------------\n";
            for (int i = 0; i < num_Radices; i++)
            {
                output += String.Format(" {0,15}    ","Radix "+ i);
                for (int j = 0; j < numBlocks; j++)
                {
                    output += String.Format("{0,5}", "");
                    for (int k = 0; k < NumGroupsPerBlock; k++)
                    {
                        output += String.Format("{0,5:x} ", printData[i * numBlocks * NumGroupsPerBlock + j * NumGroupsPerBlock + k]);
                        
                    }
                }
                output += "\n";
            }
            using (StreamWriter sw = File.AppendText(sortLog+Log_Idx+".txt"))
            {
                sw.WriteLine(output);
            }	
        }

        public static void PrintAsArray(int[] values, int count)
        {
            string output = "static ulong[] data= new ulong[]{ ";
            for (int i = 0; i < count - 1; i++)
            {
                output += string.Format("0x{0:x} ,", values[i]);
            }
            output += string.Format("0x{0:x} }};", values[values.Length - 1]);

            using (StreamWriter sw = File.AppendText(sortLog + Log_Idx + ".txt"))
            {
                sw.WriteLine(output);
            }
        }


    }
}