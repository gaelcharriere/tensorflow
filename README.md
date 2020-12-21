# tensorflow

gRPC client detecting only the following objects on images: (person=1, bicycle=2, car=3, motorcycle=4, bus=6, truck=8). The other objects detected by teh model are filtered out.
Images are loaded locally and sent to the gRPC server to be analyzed. The gRPC server returns an output containing:
* classes: The type of image detected. Classes are used to filter out the objects that we are not interested in.
* boxes: The box coordinates surrounding the object used by the client to draw the boxes around the detected object.
* scores: The score of the detection. Below a defined minimumm score we can filter out the object detected.
