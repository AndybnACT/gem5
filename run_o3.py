from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory.single_channel import SingleChannelDDR3_1600
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.processors.cpu_types import CPUTypes
from gem5.resources.resource import  BinaryResource
from gem5.simulate.simulator import Simulator
from gem5.isas import ISA
# from m5.objects import LTAGE, TAGE_SC_L_64KB, BranchPredictor, MultiperspectivePerceptron64KB
import os
from pathlib import Path
from gem5.resources.resource import BinaryResource
import m5
from m5.objects import *

# 1. Setup the Cache Hierarchy
# The O3 CPU requires a cache hierarchy. We use a standard Private L1, Private L2.
cache_hierarchy = PrivateL1PrivateL2CacheHierarchy(
    l1d_size="32kB",
    l1i_size="32kB",
    l2_size="256kB",
)

# 2. Setup the Memory System
memory = SingleChannelDDR3_1600("1GiB")

# 3. Setup the Processor (Crucial Step)
# This uses CPUTypes.O3, which maps directly to the "DerivO3CPU" you need.
processor = SimpleProcessor(
    cpu_type=CPUTypes.O3,  # <--- This enables the Out-of-Order Core
    isa=ISA.X86,
    num_cores=1,
)

# 4. Setup the Board
# The SimpleBoard connects the CPU, Cache, and Memory together.
board = SimpleBoard(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)


class BPU(BranchPredictor):
    instShiftAmt = 0
    speculativeHistUpdate = True
    conditionalBranchPred = MultiperspectivePerceptronTAGE192KB()
    requiresBTBHit = True
    takenOnlyHistory = True


# Loop through all cores (even if just 1) and swap the predictor
for core in processor.get_cores():
    # get_simobject() returns the actual C++ object (DerivO3CPU)
    cxx_core = core.get_simobject()
    # "branchQuad" is the internal name for the BP in the O3 CPU
    # We replace the default TournamentBP with LTAGE
    cxx_core.branchPred = BPU()

    # Optional: If you want to configure TAGE parameters
    # cxx_core.branchQuad.numThreads = 1
    # cxx_core.branchQuad.tableSize = 8192


# 5. Set the Workload
# This points to a standard "Hello World" binary provided by gem5 resources.
# For your project, you will replace this with your compiled SPEC CPU binary path.

board.set_se_binary_workload(
    BinaryResource(local_path="/telecomm/gsm/bin/untoast"),
    arguments=["-fps", "-c", "/telecomm/gsm/data/small.au.run.gsm"],
    stdout_file=Path("./output_large.decode.run")
)

# 6. Run the Simulation
print("Beginning simulation with O3 CPU...")
simulator = Simulator(board=board)
simulator.run()

print(f"Exiting @ tick {simulator.get_current_tick()}.")
