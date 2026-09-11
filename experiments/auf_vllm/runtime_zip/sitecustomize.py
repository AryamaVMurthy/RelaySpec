import os
if os.environ.get('TRANSFER_CAPTURE')=='1':
 import capture_runtime
 capture_runtime.install()
if os.environ.get('TRANSFER_MAPPED')=='1':
 import mapper_runtime
 mapper_runtime.install()

if os.environ.get('CROSS_BRIDGE')=='1':
 from experiments.handoff_transfer.cross.runtime import install
 install()
