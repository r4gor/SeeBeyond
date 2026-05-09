17 keypoints including nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles

• Input shape: [1, 3, 288, 512]
• Output shapes: [[1, 6, 36, 64], [1, 6, 18, 32], [1, 6, 9, 16], [1, 51, 2304], [1, 51, 576], [1, 51, 144]]

6 channels:
4 (xywh bbox) + 1 (objectness) + 1 (person class)


 samplefor a 512×288 image with one person
 0 0.523 0.401 0.312 0.748 0.498 0.142 2 0.521 0.118 2 0.476 0.121 1 0.543 0.134 0 0.461 0.139 1 0.571 0.298 2 0.441 0.302 2 0.598 0.445 2 0.419 0.451 2 0.612 0.578 2 0.401 0.582 2 0.559 0.561 2 0.452 0.558 2 0.563 0.721 2 0.448 0.718 2 0.568 0.871 2 0.444 0.868 2

 class   bbox_cx  bbox_cy  bbox_w   bbox_h
0       0.523    0.401    0.312    0.748

kp        x        y       v
nose      0.498    0.142   2   ✅ visible
l_eye     0.521    0.118   2   ✅ visible
r_eye     0.476    0.121   1   🟡 occluded
l_ear     0.543    0.134   0   ❌ not labeled
r_ear     0.461    0.139   1   🟡 occluded
l_sho     0.571    0.298   2   ✅ visible
r_sho     0.441    0.302   2   ✅ visible
l_elb     0.598    0.445   2   ✅ visible
r_elb     0.419    0.451   2   ✅ visible
l_wri     0.612    0.578   2   ✅ visible
r_wri     0.401    0.582   2   ✅ visible
l_hip     0.559    0.561   2   ✅ visible
r_hip     0.452    0.558   2   ✅ visible
l_kne     0.563    0.721   2   ✅ visible
r_kne     0.448    0.718   2   ✅ visible
l_ank     0.568    0.871   2   ✅ visible
r_ank     0.444    0.868   2   ✅ visible


The bbox tells the model where the person is, and the** keypoints are predicted relative to or within that region**. Without it, the model wouldn't know how to separate two people standing close together — the bbox is what isolates each individual before estimating their pose.
So in the annotation file, both are required for training. At inference time you typically only care about the keypoints, but the bbox is still used internally to localize each person first.