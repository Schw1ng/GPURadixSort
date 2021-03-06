

typedef struct GPUConstants
{
     int numRadices;
     int numBlocks;
     int numGroupsPerBlock;
     int R;
     int numThreadsPerGroup;
     int numElementsPerGroup;
	 int numRadicesPerBlock;
     int bitMask;
     int L;
     int numThreadsPerBlock;
     int numTotalElements;
}Constants;


inline void PrefixLocal(__local uint* inout, int p_length, int numThreads){
    __private uint glocalID = get_local_id(0);
    __private int inc = 2;

    // reduce
    while(inc <= p_length){
        
        for(int i = ((inc>>1) - 1) + (glocalID * inc)  ; (i + inc) < p_length ; i+= numThreads*inc){
            inout[i + (inc>>1)] = inout[i] + inout[i + (inc>>1)];
        }
        inc = inc <<1;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Downsweep
    inout[p_length-1] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    while(inc >=2){
        for (int i = ((inc>>1) - 1) + (glocalID * inc)  ; (i + (inc>>1)) <= p_length ; i+= numThreads*inc)
        {
            uint tmp = inout[i + (inc >>1)];
            inout[i + (inc >>1)] = inout[i] + inout[i + (inc >>1 )];
            inout[i] = tmp;
        }
        inc = inc>>1;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

inline void PrefixGlobal(__global uint* inout, int p_length, int numThreads){
    __private uint glocalID = get_local_id(0);
    __private int inc = 2;

    // reduce
    while(inc <= p_length){
        
        for(int i = ((inc>>1) - 1) + (glocalID * inc)  ; (i + inc) < p_length ; i+= numThreads*inc){
            inout[i + (inc>>1)] = inout[i] + inout[i + (inc>>1)];
        }
        inc = inc <<1;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Downsweep
    inout[p_length-1] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    while(inc >=2){
        for (int i = ((inc>>1) - 1) + (glocalID * inc)  ; (i + (inc>>1)) <= p_length ; i+= numThreads*inc)
        {
            uint tmp = inout[i + (inc >>1)];
            inout[i + (inc >>1)] = inout[i] + inout[i + (inc >>1 )];
            inout[i] = tmp;
        }
        inc = inc>>1;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}



__kernel void SetupAndCount(	__global  uint* cellIdIn, 
							__global volatile uint* counters,
							Constants dConst, 
							uint bitOffset)
{
	__private uint gLocalId = get_local_id(0);
	__private uint gBlockId = get_group_id(0);
	
    // Current threadGroup -> is based of the localId(Block internal)
	__private uint threadGroup = gLocalId / dConst.R;

    // Startindex of the datablock that corresponds to the Threadblock
    __private int actBlock = gBlockId * dConst.numGroupsPerBlock * dConst.numElementsPerGroup ;
    
    // Offset inside the block for the threadgroup of the current thread
    __private int actGroup = (gLocalId / dConst.R ) * dConst.numElementsPerGroup;

    // Startindex for the current thread
    __private uint idx = actBlock + actGroup + gLocalId % dConst.R;
    
    // Set the boarder
    __private int boarder = actBlock +actGroup + dConst.numElementsPerGroup;
    boarder = (boarder > dConst.numTotalElements)? dConst.numTotalElements : boarder;
    
    // Number of counters for each radix
    __private uint countersPerRadix = dConst.numBlocks * dConst.numGroupsPerBlock;
    // Each Threadgroup has its own counter for each radix -> Calculating offset based on current block
    __private uint counterGroupOffset = gBlockId * dConst.numGroupsPerBlock;

	for(;idx < boarder; idx += dConst.numThreadsPerGroup){
		__private uint actRadix = (cellIdIn[idx] >> bitOffset) & dConst.bitMask;
        // The following code ensures that the counters of each Threadgroup are sequentially incremented
         for(uint tmpIdx = 0 ; tmpIdx < dConst.R; tmpIdx++){
            if(gLocalId % dConst.R == tmpIdx){
                counters[ (actRadix * countersPerRadix)  +counterGroupOffset+ threadGroup ]++;
            }
      barrier(CLK_GLOBAL_MEM_FENCE);
        }
	}
}

__kernel void SumIt(	__global uint* cellIdIn, 
                            __global volatile uint* counters,
							__global uint* radixPrefixes,
							 Constants dConst, 
							uint bitOffset,
							__local uint* groupcnt)
{
	__private uint globalId = get_global_id(0);
	__private uint gLocalId = get_local_id(0);
	__private uint gBlockId = get_group_id(0);
	

    __private uint countersPerRadix = dConst.numBlocks * dConst.numGroupsPerBlock;


	__private uint actRadix = dConst.numRadicesPerBlock * gBlockId;

    for(int i = 0 ; i< dConst.numRadicesPerBlock; i++){
        // The Num_Groups counters of the radix are read from global memory to shared memory.
        // Jeder Thread liest die Counter basierend auf der localid aus
        int numIter = 0;
        uint boarder = ((actRadix+1) * countersPerRadix);
       // boarder = (boarder > dConst.numBlocks * dConst.numGroupsPerBlock)? dConst.numBlocks * dConst.numGroupsPerBlock : boarder;
        for(int j = (actRadix * countersPerRadix) + gLocalId ; j < boarder; j+= dConst.numThreadsPerBlock){
            groupcnt[gLocalId + dConst.numThreadsPerBlock * numIter++] = counters[j];
            //numIter += dConst.numThreadsPerBlock;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Die einzelnen RadixCounter sind nun in dem groupcnt local Memory
        //prefixSum(&counters[actRadix * dConst.numBlocks * dConst.numGroupsPerBlock], groupcnt, tmpPrefix,dConst.numBlocks * dConst.numGroupsPerBlock);
    	PrefixLocal(groupcnt, countersPerRadix ,dConst.numThreadsPerBlock );
        
        // PrefixSum wurde gebildet..
        barrier(CLK_LOCAL_MEM_FENCE);

        // Gesamtprefix für den aktuellen radix berechnen
        if(gLocalId == 1 ){
            radixPrefixes[actRadix] = groupcnt[(countersPerRadix) -1] + counters[((actRadix+1) * countersPerRadix)-1];
        }

        // Errechnete Prefixsumme zurück in den global memory schreiben
        barrier(CLK_GLOBAL_MEM_FENCE);
        numIter = 0;
        for(int j = (actRadix * countersPerRadix) + gLocalId ; j < ((actRadix+1) * countersPerRadix); j+= dConst.numThreadsPerBlock){
            counters[j] = groupcnt[gLocalId + dConst.numThreadsPerBlock * numIter++];
            //numIter += dConst.numThreadsPerBlock;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        actRadix++;
    }
}



__kernel void ReorderingKeysOnly(    __global uint* cellIdIn, 
                            __global uint* cellIdOut,
                            __global uint* counters,
                            __global uint* radixPrefixes,
                            __local uint* localCounters,
                            __local uint* localPrefix,
                             Constants dConst, 
                            uint bitOffset)
{
    int globalId = get_global_id(0);
    __private uint gLocalId = get_local_id(0);
    const uint gBlockId = get_group_id(0);
    const uint threadGroup= gLocalId / dConst.R;
    const uint threadGroupId= gLocalId % dConst.R;
    const uint actRadix = dConst.numRadicesPerBlock * gBlockId;
    const uint countersPerRadix = dConst.numGroupsPerBlock * dConst.numBlocks;


    __private int radixCounterOffset = actRadix * countersPerRadix;

    // erst abschließen der radix summierung
    __private  uint blockidx ; 


    // Read radix prefixes to localMemory
    for(int i = gLocalId ; i< dConst.numRadices ; i+= dConst.numThreadsPerBlock){
        localPrefix[i] = radixPrefixes[i];
    }
    // Präfixsumme über die RadixCounter bilden.
    barrier(CLK_LOCAL_MEM_FENCE);
    PrefixLocal(localPrefix, dConst.numRadices, dConst.numThreadsPerBlock);
    barrier(CLK_LOCAL_MEM_FENCE);



    // Load (groups per block * radices) counters, i.e., the block column
  for (uint i = threadGroupId; i < dConst.numRadices; i+= dConst.numThreadsPerGroup) {
    localCounters[threadGroup+ dConst.numGroupsPerBlock * i] = counters[countersPerRadix * i + gBlockId * dConst.numGroupsPerBlock + threadGroup] + localPrefix[i];
    
  }
  /*  for(int i = 0 ; i< dConst.numRadicesPerBlock ; i++){
        for(uint   blockidx =gLocalId;  
                    blockidx <  countersPerRadix ; 
                     blockidx+= dConst.numThreadsPerBlock){
            // The Num_Groups counters of the radix are read from global memory to shared memory.
            // Jeder Thread liest die Counter basierend auf der groupId aus
            localCounters[ i* countersPerRadix  +  blockidx] = counters[radixCounterOffset +  i* countersPerRadix  +  blockidx];
        }
    }
    // Die Präfixsumme des Radixe auf alle subcounter der radixes addieren
  for(int i = 0 ; i< dConst.numRadicesPerBlock ; i++){
        int numIter = 0;
        for(int j =  gLocalId ; j < countersPerRadix; j+= dConst.numThreadsPerBlock){
            localCounters[ i* countersPerRadix  +  j] += localPrefix[actRadix+i];

        }
       // barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Zurückschreiben der Radixe mit entsprechedem offset.
    for(int i = 0 ; i< dConst.numRadicesPerBlock ; i++){
        for(uint   blockidx =gLocalId;  
                    blockidx <  countersPerRadix; 
                     blockidx+= dConst.numThreadsPerBlock){
            // The Num_Groups counters of the radix are read from global memory to shared memory.
            // Jeder Thread liest die Counter basierend auf der groupId aus
            counters[radixCounterOffset +  i* countersPerRadix  +  blockidx] = localCounters[ i* countersPerRadix  +  blockidx];
        }
    }

*/

    barrier(CLK_LOCAL_MEM_FENCE);
    __private int actBlock = gBlockId * dConst.numGroupsPerBlock * dConst.numElementsPerGroup ;
    __private int actBlockCounter = gBlockId * dConst.numGroupsPerBlock  ;
    __private int actGroup = (gLocalId / dConst.R ) * dConst.numElementsPerGroup;

    __private  uint idx = actBlock + actGroup +  gLocalId % dConst.R;
    int boundary = actBlock+ actGroup +  dConst.numElementsPerGroup;
    boundary = (idx +  dConst.numElementsPerGroup < dConst.numTotalElements)? boundary : dConst.numTotalElements;
    for(;idx <   boundary ; idx += dConst.numThreadsPerGroup){
        uint tmpRdx = (cellIdIn[idx] >> bitOffset) & dConst.bitMask;
        //uint outputIdx = counters[actRadix * countersPerRadix + gBlockId * dConst.numGroupsPerBlock + threadGroup]++;
        for(uint tmpIdx = 0 ; tmpIdx < dConst.R; tmpIdx++){
            if(threadGroupId == tmpIdx){

                cellIdOut[localCounters[tmpRdx * dConst.numGroupsPerBlock + threadGroup]] = cellIdIn[idx];
                localCounters[tmpRdx * dConst.numGroupsPerBlock + threadGroup];
                //cellIdOut[idx] = gLocalId+1;

            }
            barrier(CLK_LOCAL_MEM_FENCE);

        }

    }


}








__kernel void ReorderingKeyValue(    __global uint* cellIdIn, 
                            __global uint* cellIdOut,
                            __global ulong* valueIn,
                            __global ulong* valueOut,
                            __global uint* counters,
                            __global uint* radixPrefixes,
                            __local uint* localCounters,
                            __local uint* localPrefix,
                             Constants dConst, 
                            uint bitOffset)
{
  int globalId = get_global_id(0);
    __private uint gLocalId = get_local_id(0);
    __private uint gBlockId = get_group_id(0);


    __private uint threadGroup= gLocalId / dConst.R;


    __private uint actRadix = dConst.numRadicesPerBlock * gBlockId;
    __private uint countersPerRadix = dConst.numGroupsPerBlock * dConst.numBlocks;


    __private int radixCounterOffset = actRadix * countersPerRadix;

    // erst abschließen der radix summierung
    __private  uint blockidx ; 
    for(int i = 0 ; i< dConst.numRadicesPerBlock ; i++){
        for(uint   blockidx =gLocalId;  
                    blockidx <  countersPerRadix ; 
                     blockidx+= dConst.numThreadsPerBlock){
            // The Num_Groups counters of the radix are read from global memory to shared memory.
            // Jeder Thread liest die Counter basierend auf der groupId aus
            localCounters[ i* countersPerRadix  +  blockidx] = counters[radixCounterOffset +  i* countersPerRadix  +  blockidx];
        }
    }

    // Read radix prefixes to localMemory
    for(int i = gLocalId ; i< dConst.numRadices ; i+= dConst.numThreadsPerBlock){
        localPrefix[i] = radixPrefixes[i];
    }

    // Präfixsumme über die RadixCounter bilden.
    barrier(CLK_LOCAL_MEM_FENCE);
    PrefixLocal(localPrefix, dConst.numRadices, dConst.numThreadsPerBlock);
    barrier(CLK_LOCAL_MEM_FENCE);
 



    // Die Präfixsumme des Radixe auf alle subcounter der radixes addieren
  for(int i = 0 ; i< dConst.numRadicesPerBlock ; i++){
        int numIter = 0;
        //for(int j = ((actRadix+i) * dConst.numBlocks * dConst.numGroupsPerBlock) + localId ; j < ((dConst.numRadicesPerBlock * gBlockId + 1) * dConst.numBlocks * dConst.numGroupsPerBlock); j+= dConst.numThreadsPerBlock){
        for(int j =  gLocalId ; j < countersPerRadix; j+= dConst.numThreadsPerBlock){
            //groupcnt[gLocalId + dConst.numThreadsPerBlock * numIter++] = counters[j];
            //if(gLocalId == 0 && gBlockId ==0 && i==0 && j == gLocalId )
            localCounters[ i* countersPerRadix  +  j] += localPrefix[actRadix+i];

        }
       // barrier(CLK_LOCAL_MEM_FENCE);
    }
    



    // Zurückschreiben der Radixe mit entsprechedem offset.
    for(int i = 0 ; i< dConst.numRadicesPerBlock ; i++){
        for(uint   blockidx =gLocalId;  
                    blockidx <  countersPerRadix; 
                     blockidx+= dConst.numThreadsPerBlock){
            // The Num_Groups counters of the radix are read from global memory to shared memory.
            // Jeder Thread liest die Counter basierend auf der groupId aus
            counters[radixCounterOffset +  i* countersPerRadix  +  blockidx] = localCounters[ i* countersPerRadix  +  blockidx];
        }
    }



    barrier(CLK_LOCAL_MEM_FENCE);



    __private int actBlock = gBlockId * dConst.numGroupsPerBlock * dConst.numElementsPerGroup ;
    __private int actBlockCounter = gBlockId * dConst.numGroupsPerBlock  ;

    __private int actGroup = (gLocalId / dConst.R ) * dConst.numElementsPerGroup;

    __private  uint idx = actBlock + actGroup +  gLocalId % dConst.R;
    int boundary = actBlock+ actGroup +  dConst.numElementsPerGroup;
    boundary = (actBlock+ actGroup +  dConst.numElementsPerGroup < dConst.numTotalElements)? boundary : dConst.numTotalElements;
    for(;idx <   boundary ; idx += dConst.numThreadsPerGroup){
        uint tmpRdx = (cellIdIn[idx] >> bitOffset) & dConst.bitMask;
        //uint outputIdx = counters[actRadix * countersPerRadix + gBlockId * dConst.numGroupsPerBlock + threadGroup]++;
        for(uint tmpIdx = 0 ; tmpIdx < dConst.R; tmpIdx++){
            if(gLocalId % dConst.R == tmpIdx){

                cellIdOut[counters[tmpRdx * countersPerRadix + actBlockCounter+ threadGroup]] = cellIdIn[idx];
                valueOut[counters[tmpRdx * countersPerRadix + actBlockCounter+ threadGroup]++] = valueIn[idx];
                //cellIdOut[idx] = gLocalId+1;

            }
            barrier(CLK_LOCAL_MEM_FENCE);


        }

    }

/*

    __private  uint idx = (gBlockId * dConst.numGroupsPerBlock + gLocalId / dConst.R) * dConst.numElementsPerGroup + gLocalId % dConst.R;

    for(;idx < (gBlockId * dConst.numGroupsPerBlock + threadGroup+1) * dConst.numElementsPerGroup; idx += dConst.numThreadsPerGroup){
        uint actRadix = (cellIdIn[idx] >> bitOffset) & dConst.bitMask;
        //uint outputIdx = counters[actRadix * countersPerRadix + gBlockId * dConst.numGroupsPerBlock + threadGroup]++;
        for(uint tmpIdx = 0 ; tmpIdx < dConst.R; tmpIdx++){
            if(gLocalId % dConst.R == tmpIdx){
                cellIdOut[counters[actRadix * countersPerRadix + gBlockId * dConst.numGroupsPerBlock + threadGroup]] = cellIdIn[idx];
                valueOut[counters[actRadix * countersPerRadix + gBlockId * dConst.numGroupsPerBlock + threadGroup]++] = valueIn[idx];
            }
            barrier(CLK_LOCAL_MEM_FENCE);


        }

    }
*/

}


