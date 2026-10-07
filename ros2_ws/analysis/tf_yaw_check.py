import sys, math, rosbag2_py
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage
uri=sys.argv[1]; r=rosbag2_py.SequentialReader(); r.open(rosbag2_py.StorageOptions(uri=uri,storage_id=""),rosbag2_py.ConverterOptions("",""))
ys=[]
while r.has_next():
    tp,d,t=r.read_next()
    if tp=="/tf":
        for tr in deserialize_message(d,TFMessage).transforms:
            if tr.header.frame_id=="map" and tr.child_frame_id=="odom":
                q=tr.transform.rotation; ys.append(math.degrees(math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))))
print(len(ys), "map->odom yaw deg min/max", round(min(ys),1), round(max(ys),1))
