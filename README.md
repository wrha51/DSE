## DSE tool for AI Accelerators
Extension of zigzag-v1 https://github.com/ZigZag-Project/zigzag-v1

## Configuration


### Hardware Configuration
**Supported HW**|**PE Dimension**|**Global Memory**|**GM Bandwidth**|**GM Area**
:---:|:---:|:---:|:---:|:---:
Edge|32x32|33554432|512|27.0113
Mobile|64x64|67108864|2048|51.9235
Cloud|128x128|134217728|8192|103.323

### Benchmark


ARVRA : ResNet50, UNet, MobileNetv2


ARVRB : ResNet50, UNet, MobileNetv2, BRQ, DepthNet


MLPerf: ResNet50, MobileNetv1, SSDResNet34, SSDMobileNetv1, GNMT


## Quickstart
Change the ```run_batch.py``` lines 55~73 to configure HW/SW settings

To run the framework
```
python3 run_batch.py \
```

