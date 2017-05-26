using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.Remoting.Contexts;
using Cloo;
using Cloo.Bindings;
using Environment = System.Environment;

namespace GPGPUCollisionDetection
{
    public class GPURadixSort
    {
        private static int Log_Idx = 0;
        private const bool DEBUG = true;
        private const bool DEBUG_CONSOLE_OUTPUT = false;
        private readonly string debugLog = Path.Combine(Directory.GetCurrentDirectory() , "OpenCLDebugLog.txt");
        private static string sortLog = Path.Combine(Directory.GetCurrentDirectory(), "sortLog");
        private string programPath = Path.Combine(Directory.GetCurrentDirectory(), @"..\..\..\GPURadixSort\RadixSort.cl");

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


        private CLContextHandle cxGPUContext { get; set; }
        private CLCommandQueueHandle cqCommandQueue { get; set; }
        private CLDeviceHandle _device { get; set; }

        private GPUConstants gpuConstants;


        private CLMemoryHandle mInputBuff;
        private CLMemoryHandle mCounters;
        private CLMemoryHandle mRadixPrefixes;
        private CLMemoryHandle mOutputBuff;
        //private CLMemoryHandle mBlockOffsets;

        private const int numCounters = num_Radices*NumGroupsPerBlock*numBlocks;

        // Anzahl an Bits die als Buckets für jeden radix durchlauf verwendet werden
        private const int radix_BitsL = 4;
        private const int num_Radices = 1 << radix_BitsL;
       
        // Auch Thread Blocks unter CUDA -> Gruppe von Threads mit gemeinsamen shared memory.
        private const int numBlocks = 4;

        // Anzahl von WorkItems / Threads, die sich in einer Work-Group befinden
        private const int numThreadsPerBlock = 32;

        private const int R = 8;

        private const int NumGroupsPerBlock = numThreadsPerBlock / R;
        
        private const int BIT_MASK_START = 0xF;

        int[] counters = new int[numCounters];




        CLKernelHandle ckSetupAndCount; // OpenCL kernels
        CLKernelHandle ckSumIt;
        CLKernelHandle ckReorderingKeysOnly;
        CLKernelHandle ckReorderingKeyValue;
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
         
        private void CheckErr(ComputeErrorCode err, string name)
        {

            if (err != ComputeErrorCode.Success)
            {
                Debug.WriteLine(err.ToString() + " Text: " + name+" from : ");
                if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine(err.ToString() + " Text: " + name+" from : ");
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
             ComputeCommandQueue commandQue,
             ComputeContext context,
            ComputeDevice device
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
            ComputeErrorCode error;
            cxGPUContext = context.Handle;
            cqCommandQueue = commandQue.Handle;
            _device = device.Handle;
            //Create a command queue, where all of the commands for execution will be added
            /*cqCommandQueue = CL10.CreateCommandQueue(cxGPUContext, _device, (CommandQueueProperties)0, out  error);
            CheckErr(error, "CL10.CreateCommandQueue");*/
            string programSource = System.IO.File.ReadAllText(programPath);
            IntPtr[] progSize = new IntPtr[] { (IntPtr)programSource.Length };
            string flags = "-cl-fast-relaxed-math";

            ComputeProgram prog = new ComputeProgram(context, programSource);
            prog.Build(new List<ComputeDevice>(){ device },flags,null, IntPtr.Zero );


            if (prog.GetBuildStatus(device) != ComputeProgramBuildStatus.Success)
            {
                Debug.WriteLine(prog.GetBuildLog(device));
                throw new ArgumentException("UNABLE to build programm");
            }
            //            ComputeProgram clProgramRadix = CL10.CreateProgramWithSource(cxGPUContext, 1, new[] { programSource },progSize,
            //                out error);

            CLProgramHandle clProgramRadix = prog.Handle;



            ckSetupAndCount = CL10.CreateKernel(clProgramRadix, "SetupAndCount", out error);
            CheckErr(error, "CL10.CreateKernel");
            ckSumIt = CL10.CreateKernel(clProgramRadix, "SumIt", out error);
            CheckErr(error, "CL10.CreateKernel");
            ckReorderingKeysOnly = CL10.CreateKernel(clProgramRadix, "ReorderingKeysOnly", out error);
            CheckErr(error, "CL10.CreateKernel");
            ckReorderingKeyValue = CL10.CreateKernel(clProgramRadix, "ReorderingKeyValue", out error);
            CheckErr(error, "CL10.CreateKernel");
        }






       public void sortKeysOnly(CLMemoryHandle input, CLMemoryHandle output,
                    int numElements)
        {
            debugRead = new int[Math.Max(numElements,numCounters)];
            ComputeErrorCode error;
            Compute ComputeEvent eve;

           mCounters = CL10.CreateBuffer(cxGPUContext, ComputeMemoryFlags.ReadWrite, gpuConstants.numGroupsPerBlock * gpuConstants.numRadices * gpuConstants.numBlocks * sizeof(int),
            out error);
           CheckErr(error, "CL10.CreateBuffer");

           mRadixPrefixes = CL10.CreateBuffer(cxGPUContext, ComputeMemoryFlags.ReadWrite, gpuConstants.numRadices * sizeof(int), out error);
           CheckErr(error, "CL10.CreateBuffer");

           gpuConstants.numElementsPerGroup = (numElements/(gpuConstants.numBlocks*gpuConstants.numGroupsPerBlock)) +1 ;
           gpuConstants.numTotalElements = numElements;

           if (DEBUG) {
                CL10.EnqueueReadBuffer(cqCommandQueue, input, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4), debugRead, 0,
                    null, out eve);
                CheckErr(error, "CL10.EnqueueReadBuffer");
                PrintAsArray(debugRead, gpuConstants.numTotalElements);
            }
            int i;
           for (i = 0; i < 8; i++)
           {
               error = CL10.EnqueueWriteBuffer(cqCommandQueue, mCounters, true, IntPtr.Zero, (IntPtr)(numCounters * 4),
                counters, 0, null, out eve);
               CheckErr(error, "CL10.EnqueueWriteBuffer Counter initialize");
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
               error= CL10.EnqueueCopyBuffer(cqCommandQueue, input, output, IntPtr.Zero, IntPtr.Zero, (IntPtr)(numElements * 4), 0, null, out eve);
               CheckErr(error, "CL10.EnqueueCopyBuffer");
               error = CL10.Finish(cqCommandQueue);
               CheckErr(error, "CL10.Finish Copybuffer");
           }
           error = CL10.ReleaseMemObject(mRadixPrefixes);
           CheckErr(error, "CL10.ReleaseMemObj");
           error = CL10.ReleaseMemObject(mCounters);
           CheckErr(error, "CL10.ReleaseMemObj");
           Log_Idx++;

       }


       public void sortKeysValue(CLMemoryHandle inputKey, CLMemoryHandle outputKey,CLMemoryHandle inputValue, CLMemoryHandle outputValue,
                int numElements)
       {
           debugRead = new int[Math.Max(numElements, numCounters)];
           ComputeErrorCode error;
            ComputeEvent eve;
           mCounters = CL10.CreateBuffer(cxGPUContext, ComputeMemoryFlags.ReadWrite,(IntPtr) (gpuConstants.numGroupsPerBlock * gpuConstants.numRadices * gpuConstants.numBlocks * sizeof(int)),debugRead,
                out error);
           CheckErr(error, "CL10.CreateBuffer");

           mRadixPrefixes = CL10.CreateBuffer(cxGPUContext, ComputeMemoryFlags.ReadWrite, gpuConstants.numRadices * sizeof(int), out error);
           CheckErr(error, "CL10.CreateBuffer");

            /*
                       error = CL10.EnqueueReadBuffer(cqCommandQueue, input, Bool.True, IntPtr.Zero, (IntPtr)(numElements * 4),
                debugRead, 0, null, out eve);
                       CheckErr(error, "CL10.EnqueueReadBuffer");
            */
            if (DEBUG)
            {
                CL10.EnqueueReadBuffer(cqCommandQueue, inputKey, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4), debugRead, 0,
                    null, out eve);
                CheckErr(error, "CL10.EnqueueReadBuffer");
                PrintAsArray(debugRead, gpuConstants.numTotalElements);
            }
            gpuConstants.numElementsPerGroup = (numElements / (gpuConstants.numBlocks * gpuConstants.numGroupsPerBlock)) + 1;
           gpuConstants.numTotalElements = numElements;
           int i;
           for (i = 0; i < 8; i++)
           {
               error = CL10.EnqueueWriteBuffer(cqCommandQueue, mCounters, Bool.True, IntPtr.Zero, (IntPtr)(numCounters * 4),
                counters, 0, null, out eve);
               CheckErr(error, "CL10.EnqueueWriteBuffer Counter initialize");
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
               error = CL10.EnqueueCopyBuffer(cqCommandQueue, inputKey, outputKey, IntPtr.Zero, IntPtr.Zero, (IntPtr)(numElements * 4), 0, null, out eve);
               CheckErr(error, "CL10.EnqueueCopyBuffer");
               error = CL10.Finish(cqCommandQueue);
               CheckErr(error, "CL10.Finish Copybuffer");
               error = CL10.EnqueueCopyBuffer(cqCommandQueue, inputValue, outputValue, IntPtr.Zero, IntPtr.Zero, (IntPtr)(numElements * 8), 0, null, out eve);
               CheckErr(error, "CL10.EnqueueCopyBuffer");
               error = CL10.Finish(cqCommandQueue);
               CheckErr(error, "CL10.Finish Copybuffer");
           }
           error = CL10.ReleaseMemObject(mRadixPrefixes);
           CheckErr(error, "CL10.ReleaseMemObj");
           error = CL10.ReleaseMemObject(mCounters);
           CheckErr(error, "CL10.ReleaseMemObj");

            Log_Idx++;

        }


        public void sortKeysValue(CLMemoryHandle key, CLMemoryHandle value,  
              int numElements)
       {
           debugRead = new int[Math.Max(numElements, numCounters)];
           ComputeErrorCode error;
            ComputeEvent eve;
           /*
                      error = CL10.EnqueueReadBuffer(cqCommandQueue, input, Bool.True, IntPtr.Zero, (IntPtr)(numElements * 4),
               debugRead, 0, null, out eve);
                      CheckErr(error, "CL10.EnqueueReadBuffer");
           */

           mCounters = CL10.CreateBuffer(cxGPUContext,  ComputeMemoryFlags.ReadWrite, gpuConstants.numGroupsPerBlock * gpuConstants.numRadices * gpuConstants.numBlocks * sizeof(int),
                out error);
           CheckErr(error, "CL10.CreateBuffer");

           mRadixPrefixes = CL10.CreateBuffer(cxGPUContext, ComputeMemoryFlags.ReadWrite, gpuConstants.numRadices * sizeof(int), out error);
           CheckErr(error, "CL10.CreateBuffer");
            CLMemoryHandle outputValue = CL10.CreateBuffer(cxGPUContext, ComputeMemoryFlags.ReadWrite, (IntPtr)(8 * numElements),
                out error);
            CheckErr(error, "CL10.CreateBuffer");
            CLMemoryHandle outputKey = CL10.CreateBuffer(cxGPUContext, ComputeMemoryFlags.ReadWrite, (IntPtr)(4 * numElements),
                out error);
           CheckErr(error, "CL10.CreateBuffer");


           gpuConstants.numElementsPerGroup = (numElements / (gpuConstants.numBlocks * gpuConstants.numGroupsPerBlock)) + 1;
           gpuConstants.numTotalElements = numElements;
           int i;
           for (i = 0; i < 8; i++)
           {
               error = CL10.EnqueueWriteBuffer(cqCommandQueue, mCounters, Bool.True, IntPtr.Zero, (IntPtr)(numCounters * 4),
                counters, 0, null, out eve);
               CheckErr(error, "CL10.EnqueueWriteBuffer Counter initialize");
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
               error = CL10.EnqueueCopyBuffer(cqCommandQueue, outputKey, key, IntPtr.Zero, IntPtr.Zero, (IntPtr)(numElements * 4), 0, null, out eve);
               CheckErr(error, "CL10.EnqueueCopyBuffer");
               error = CL10.Finish(cqCommandQueue);
               CheckErr(error, "CL10.Finish Copybuffer");
               error = CL10.EnqueueCopyBuffer(cqCommandQueue, outputValue, value, IntPtr.Zero, IntPtr.Zero, (IntPtr)(numElements * 8), 0, null, out eve);
               CheckErr(error, "CL10.EnqueueCopyBuffer");
               error = CL10.Finish(cqCommandQueue);
               CheckErr(error, "CL10.Finish Copybuffer");
           }


           error = CL10.ReleaseMemObject(outputKey);
           CheckErr(error, "CL10.ReleaseMemObj");
           error = CL10.ReleaseMemObject(outputValue);
           CheckErr(error, "CL10.ReleaseMemObj");
           error = CL10.ReleaseMemObject(mRadixPrefixes);
           CheckErr(error, "CL10.ReleaseMemObj");
           error = CL10.ReleaseMemObject(mCounters);
           CheckErr(error, "CL10.ReleaseMemObj");
            Log_Idx++;

        }




        private void ReorderingKeysOnly(CLMemoryHandle input, CLMemoryHandle output, int bitOffset)
       {
           ComputeErrorCode error;
           IntPtr agentPtrSize = (IntPtr)0;
           agentPtrSize = (IntPtr)Marshal.SizeOf(typeof(IntPtr));
           var ptrSize = (IntPtr)Marshal.SizeOf(typeof(Mem));


           int globalWorkSize = gpuConstants.numThreadsPerBlock * gpuConstants.numBlocks;
           int localWorkSize = gpuConstants.numThreadsPerBlock;

           IntPtr[] workGroupSizePtr = new IntPtr[] { (IntPtr)globalWorkSize };
           IntPtr[] localWorkGroupSizePtr = new IntPtr[] { (IntPtr)localWorkSize };
            ComputeEvent clevent;
           error = CL10.SetKernelArg(ckReorderingKeysOnly, 0, ptrSize, input.Value);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckReorderingKeysOnly, 1, ptrSize, output.Value);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckReorderingKeysOnly, 2, ptrSize, mCounters.Value);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckReorderingKeysOnly, 3, ptrSize, mRadixPrefixes.Value);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckReorderingKeysOnly, 4, (IntPtr)(gpuConstants.numGroupsPerBlock * gpuConstants.numBlocks * gpuConstants.numRadicesPerBlock * 4), IntPtr.Zero);
            CheckErr(error, "CL10.SetKernelArg");
            error = CL10.SetKernelArg(ckReorderingKeysOnly, 5, (IntPtr)(gpuConstants.numRadices * 4), null);
            CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckReorderingKeysOnly, 6, (IntPtr)(Marshal.SizeOf(typeof(GPUConstants))), gpuConstants);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckReorderingKeysOnly, 7, (IntPtr)4, bitOffset);
           CheckErr(error, "CL10.SetKernelArg");

           error = CL10.EnqueueNDRangeKernel(cqCommandQueue, ckReorderingKeysOnly, 1, null, workGroupSizePtr, localWorkGroupSizePtr, 0, null, out clevent);
           CheckErr(error, "CL10.EnqueueNDRangeKernel");

           error = CL10.Finish(cqCommandQueue);
           CheckErr(error, "CL10.Finish ReorderKeysOnly");
           if (DEBUG)
           {
                
               //if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("-------------------------------Reordering-------------------------------------------------");
                ComputeEvent eve;
               
               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("              Input                ");
               CL10.EnqueueReadBuffer(cqCommandQueue, input, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4), debugRead, 0,
                   null, out eve);
               CheckErr(error, "CL10.EnqueueReadBuffer");
               PrintElementBuffer(debugRead, gpuConstants.numTotalElements, "Reordering -> Input -> bitoffset = " + bitOffset);

                if (DEBUG_CONSOLE_OUTPUT) Console.WriteLine("              Counters                ");
               CL10.EnqueueReadBuffer(cqCommandQueue, mCounters, Bool.True, IntPtr.Zero, (IntPtr)(numCounters * sizeof(int)), debugRead, 0,
                   null, out eve);
               CheckErr(error, "CL10.EnqueueReadBuffer");
               PrintCounterBuffer(debugRead, "Reordering -> bitoffset = " + bitOffset);

               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("              Counters                ");
               CL10.EnqueueReadBuffer(cqCommandQueue, mRadixPrefixes, Bool.True, IntPtr.Zero, (IntPtr)( gpuConstants.numRadices * sizeof(int)), debugRead, 0,
                   null, out eve);
               CheckErr(error, "CL10.EnqueueReadBuffer");
               PrintElementBuffer(debugRead,gpuConstants.numRadices, "Reordering -> RadixPrefixe -> bitoffset = " + bitOffset);



               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("              Output                ");
               CL10.EnqueueReadBuffer(cqCommandQueue, output, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4), debugRead, 0,
                   null, out eve);
               CheckErr(error, "CL10.EnqueueReadBuffer");
               PrintElementBuffer(debugRead, gpuConstants.numTotalElements, "Reordering -> Output -> bitoffset = " + bitOffset);


               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("Reordering -> bitoffset = " + bitOffset);
               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine();
           };

        }

        private void ReorderingKeyValue(CLMemoryHandle inputKey, CLMemoryHandle outputKey,CLMemoryHandle inputValue, CLMemoryHandle outputValue, int bitOffset)
       {
           ComputeErrorCode error;
           IntPtr agentPtrSize = (IntPtr)0;
           agentPtrSize = (IntPtr)Marshal.SizeOf(typeof(IntPtr));
           var ptrSize = (IntPtr)Marshal.SizeOf(typeof(Mem));


           int globalWorkSize = gpuConstants.numThreadsPerBlock * gpuConstants.numBlocks;
           int localWorkSize = gpuConstants.numThreadsPerBlock;

           IntPtr[] workGroupSizePtr = new IntPtr[] { (IntPtr)globalWorkSize };
           IntPtr[] localWorkGroupSizePtr = new IntPtr[] { (IntPtr)localWorkSize };
            ComputeEvent clevent;
           error = CL10.SetKernelArg(ckReorderingKeyValue, 0, ptrSize, inputKey);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckReorderingKeyValue, 1, ptrSize, outputKey);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckReorderingKeyValue, 2, ptrSize, inputValue);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckReorderingKeyValue, 3, ptrSize, outputValue);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckReorderingKeyValue, 4, ptrSize, mCounters);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckReorderingKeyValue, 5, ptrSize, mRadixPrefixes);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckReorderingKeyValue, 6, (IntPtr)(gpuConstants.numGroupsPerBlock * gpuConstants.numBlocks * gpuConstants.numRadicesPerBlock * 4), null);
           CheckErr(error, "CL10.SetKernelArg");
            error = CL10.SetKernelArg(ckReorderingKeyValue, 7, (IntPtr)(gpuConstants.numRadices * 4), null);
            CheckErr(error, "CL10.SetKernelArg");
            error = CL10.SetKernelArg(ckReorderingKeyValue, 8, (IntPtr)(Marshal.SizeOf(typeof(GPUConstants))), gpuConstants);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckReorderingKeyValue, 9, (IntPtr)4, bitOffset);
           CheckErr(error, "CL10.SetKernelArg");

           error = CL10.EnqueueNDRangeKernel(cqCommandQueue, ckReorderingKeyValue, 1, null, workGroupSizePtr, localWorkGroupSizePtr, 0, null, out clevent);
           CheckErr(error, "CL10.EnqueueNDRangeKernel");

           error = CL10.Finish(cqCommandQueue);
           CheckErr(error, "CL10.Finish");
           if (DEBUG)
           {
               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("-------------------------------Reordering-------------------------------------------------");
                ComputeEvent eve;

               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("              Input                ");
               CL10.EnqueueReadBuffer(cqCommandQueue, inputKey, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4), debugRead, 0,
                   null, out eve);
               CheckErr(error, "CL10.EnqueueReadBuffer");
               PrintElementBuffer(debugRead, gpuConstants.numTotalElements, "Reordering -> Input -> bitoffset = " + bitOffset);
              
               CL10.EnqueueReadBuffer(cqCommandQueue, inputValue, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4), debugRead, 0,
                   null, out eve);
               CheckErr(error, "CL10.EnqueueReadBuffer");
               PrintElementBuffer(debugRead, gpuConstants.numTotalElements, "Reordering -> InputValues -> bitoffset = " + bitOffset);

               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("              Counters                ");
               CL10.EnqueueReadBuffer(cqCommandQueue, mCounters, Bool.True, IntPtr.Zero, (IntPtr)(numCounters * sizeof(int)), debugRead, 0,
                   null, out eve);
               CheckErr(error, "CL10.EnqueueReadBuffer");
               PrintCounterBuffer(debugRead, "Reordering -> bitoffset = " + bitOffset);

               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("              Counters                ");
               CL10.EnqueueReadBuffer(cqCommandQueue, mRadixPrefixes, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numRadices * sizeof(int)), debugRead, 0,
                   null, out eve);
               CheckErr(error, "CL10.EnqueueReadBuffer");
               PrintElementBuffer(debugRead, gpuConstants.numRadices, "Reordering -> RadixPrefixe -> bitoffset = " + bitOffset);



               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("              Output                ");
               CL10.EnqueueReadBuffer(cqCommandQueue, outputKey, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4), debugRead, 0,
                   null, out eve);
               CheckErr(error, "CL10.EnqueueReadBuffer");
               PrintElementBuffer(debugRead, gpuConstants.numTotalElements, "Reordering -> Output -> bitoffset = " + bitOffset);

               CL10.EnqueueReadBuffer(cqCommandQueue, outputValue, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4), debugRead, 0,
                   null, out eve);
               CheckErr(error, "CL10.EnqueueReadBuffer");
               PrintElementBuffer(debugRead, gpuConstants.numTotalElements, "Reordering -> OutputValue -> bitoffset = " + bitOffset);

               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("Reordering -> bitoffset = " + bitOffset);
               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine();
           };

        }
        private void SumIt(CLMemoryHandle input, int bitOffset)
       {
           ComputeErrorCode error;
           IntPtr agentPtrSize = (IntPtr)0;
           agentPtrSize = (IntPtr)Marshal.SizeOf(typeof(IntPtr));
           var ptrSize = (IntPtr)Marshal.SizeOf(typeof(Mem));


           int globalWorkSize = gpuConstants.numThreadsPerBlock * gpuConstants.numBlocks;
           int localWorkSize = gpuConstants.numThreadsPerBlock;

           IntPtr[] workGroupSizePtr = new IntPtr[] { (IntPtr)globalWorkSize };
           IntPtr[] localWorkGroupSizePtr = new IntPtr[] { (IntPtr)localWorkSize };
            ComputeEvent clevent;
           error = CL10.SetKernelArg(ckSumIt, 0, ptrSize, input);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckSumIt, 1, ptrSize, mCounters);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckSumIt, 2, ptrSize, mRadixPrefixes);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckSumIt, 3, (IntPtr)(Marshal.SizeOf(typeof(GPUConstants))), gpuConstants);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckSumIt, 4, (IntPtr)4, bitOffset);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckSumIt, 5, (IntPtr)(4* gpuConstants.numBlocks*gpuConstants.numGroupsPerBlock), null);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.EnqueueNDRangeKernel(cqCommandQueue, ckSumIt, 1, null, workGroupSizePtr, localWorkGroupSizePtr, 0, null, out clevent);
           CheckErr(error, "CL10.EnqueueNDRangeKernel");
           
           error = CL10.Finish(cqCommandQueue);
           CheckErr(error, "CL10.Finish");
           if (DEBUG)
           {
                ComputeEvent eve;

               error = CL10.EnqueueReadBuffer(cqCommandQueue, input, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4),
                debugRead, 0, null, out eve);
                CheckErr(error, "CL10.EnqueueReadBuffer");

                PrintElementBuffer(debugRead, gpuConstants.numTotalElements, "SumIt -> Input -> bitoffset = " + bitOffset);

               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("              Counters                ");
               CL10.EnqueueReadBuffer(cqCommandQueue, mCounters, Bool.True, IntPtr.Zero, (IntPtr)(numCounters * sizeof(int)), debugRead, 0,
                   null, out eve);
               CheckErr(error, "CL10.EnqueueReadBuffer");
               PrintCounterBuffer(debugRead, "SumIt -> bitoffset = " + bitOffset);
               if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine("SumIt -> bitoffset = " + bitOffset);
                CL10.EnqueueReadBuffer(cqCommandQueue, mRadixPrefixes, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numRadices * sizeof(int)), debugRead, 0,
                    null, out eve);
                CheckErr(error, "CL10.EnqueueReadBuffer");
                PrintElementBuffer(debugRead, gpuConstants.numRadices, "SumIt -> RadixPrefixe -> bitoffset = " + bitOffset);
                if (DEBUG_CONSOLE_OUTPUT) Console.WriteLine();    
           };
       }

       private void SetupAndCount(CLMemoryHandle input, int bitOffset)
       {
           ComputeErrorCode error;
           IntPtr agentPtrSize = (IntPtr)0;
           agentPtrSize = (IntPtr)Marshal.SizeOf(typeof(IntPtr));
           var ptrSize = (IntPtr)Marshal.SizeOf(typeof(Mem));


           int globalWorkSize = gpuConstants.numThreadsPerBlock * gpuConstants.numBlocks;
           int localWorkSize = gpuConstants.numThreadsPerBlock;

           IntPtr[] workGroupSizePtr = new IntPtr[] { (IntPtr)globalWorkSize };
           IntPtr[] localWorkGroupSizePtr = new IntPtr[] { (IntPtr)localWorkSize };
            ComputeEvent clevent;
           error = CL10.SetKernelArg(ckSetupAndCount, 0, ptrSize, input);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckSetupAndCount, 1, ptrSize, mCounters);
           CheckErr(error, "CL10.SetKernelArg");
           //if(DEBUG_CONSOLE_OUTPUT) Console.WriteLine((Marshal.SizeOf(typeof(GPUConstants))));
           error = CL10.SetKernelArg(ckSetupAndCount, 2, (IntPtr)(Marshal.SizeOf(typeof(GPUConstants))), gpuConstants);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.SetKernelArg(ckSetupAndCount, 3, (IntPtr)4, bitOffset);
           CheckErr(error, "CL10.SetKernelArg");
           error = CL10.EnqueueNDRangeKernel(cqCommandQueue, ckSetupAndCount, 1, null, workGroupSizePtr, localWorkGroupSizePtr, 0, null, out clevent);
           CheckErr(error, "CL10.EnqueueNDRangeKernel");

           error = CL10.Finish(cqCommandQueue);
           CheckErr(error, "CL10.Finish");
           if (DEBUG)
           {
                ComputeEvent eve;
               CL10.EnqueueReadBuffer(cqCommandQueue, input, Bool.True, IntPtr.Zero, (IntPtr)(gpuConstants.numTotalElements * 4), debugRead, 0,
                   null, out eve);
               CheckErr(error, "CL10.EnqueueReadBuffer");
               PrintElementBuffer(debugRead, gpuConstants.numTotalElements, "Setup and Count -> Input  -> bitoffset = " + bitOffset);              

               CL10.EnqueueReadBuffer(cqCommandQueue, mCounters, Bool.True, IntPtr.Zero, (IntPtr)(numCounters * sizeof(int)), debugRead, 0,
                   null, out eve);
               CheckErr(error, "CL10.EnqueueReadBuffer");
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