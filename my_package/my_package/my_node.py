import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

# bridge side of the roslibpy_bridge to test with ros2 humble
# publishes every second on to the /chatter topic

# class BridgeNode(Node):
#    def __init__(self):
#        super().__init__('bridge_node')
#        self.publisher = self.create_publisher(String, '/chatter', 10)
#        self.timer = self.create_timer(1.0, self.timer_callback)
#
#    def timer_callback(self):
#        msg = String()
#        msg.data = 'Hello from the bridge!'
#        self.publisher.publish(msg)
#        self.get_logger().info('Publishing: "%s"' % msg.data)
#

# publishes an image once from the path /msak_0.png on the topic /camera/image
# class BridgeNode(Node):
#     def __init__(self):
#         super().__init__('bridge_node')
#         self.publisher = self.create_publisher(Image, '/camera/image', 10)
#         self.timer = self.create_timer(1.0, self.timer_callback)
#         self.bridge = CvBridge()

#     def timer_callback(self):
#         image_path = '/media/vincent/more/bpc_teamname/mask_0.png'
#         # Load the image from the specified path
#         cv_image = cv2.imread(image_path)

#         if cv_image is None:
#             self.get_logger().error(f"Failed to load image from {image_path}")
#             return

#         # Convert the OpenCV image to a ROS Image message
#         msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
#         msg.header.stamp = self.get_clock().now().to_msg()
#         msg.header.frame_id = 'camera_frame'

#         self.publisher.publish(msg)
#         self.get_logger().info('Publishing image from: "%s"' % image_path)


# def main(args=None):
#     rclpy.init(args=args)
#     node = BridgeNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()


import rclpy
from rclpy.node import Node
import base64
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

#! from ibpc_interfaces.srv import GetPoseEstimates
from ibpc_interfaces.srv import TestService


class ImageProcessingClient(Node):
    def __init__(self):
        super().__init__("image_processing_client")

        # Create a client for the /process_image service
        #! self.client = self.create_client(GetPoseEstimates, '/process_image')
        self.client = self.create_client(TestService, "/process_image")

        # Wait for the service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /process_image service...")

        # Load the random image
        image_path = "/media/vincent/more/bpc_teamname/mask_0.png"
        self.get_logger().info(f"Loading image from {image_path}")
        image = cv2.imread(image_path)

        if image is None:
            self.get_logger().error("Failed to load image. Exiting...")
            return

        # Encode the image to base64
        _, buffer = cv2.imencode(".png", image)
        base64_image = base64.b64encode(buffer).decode("ascii")

        # Create the service request
        request = TestService.Request()
        bridge = CvBridge()
        img_msg = bridge.cv2_to_imgmsg(
            image, encoding="bgr8"
        )  # Convert OpenCV image to ROS Image message
        request.rgb = img_msg

        # Call the service
        self.get_logger().info("Calling /process_image service...")
        future = self.client.call_async(request)
        future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        try:
            response = future.result()
            self.get_logger().info("Service call completed.")
            if response is None:
                self.get_logger().error("Service call failed: No response received.")
                return
            self.get_logger().info(f"Service response: {response}")
            if response.success:
                self.get_logger().info("Image processed successfully.")
                # Decode the processed image from base64
                processed_image_bytes = base64.b64decode(response.processed_image)
                np_array = np.frombuffer(processed_image_bytes, dtype=np.uint8)
                processed_image = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

                # Display the processed image
                cv2.imshow("Processed Image", processed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                self.get_logger().error(f"Failed to process image: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Error while calling service: {e}")
        finally:
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessingClient()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
